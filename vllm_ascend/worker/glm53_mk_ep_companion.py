"""Run the local experts for an idle rank of a single-origin TP1 B1 MK decode step."""

import builtins
from dataclasses import dataclass

import torch

from blockrt.models.glm_5_3_flash import mxfp4
from blockrt.runtime.tracer import DAGCapture


_SEND_CAPACITY = 8
_HIDDEN_SIZE = 4096
_EXPERTS = 288
_COMM_OPS = ("moe_metadata_exchange", "moe_dispatch", "moe_combine")


@dataclass(frozen=True)
class LayerReceive:
    layer_id: int
    receive_rows: int


class Glm53MKEPCompanion:
    """Capture one graph per sparse layer using the model's bound expert weights.

    The caller owns the SHMEM runtime and coordinates graph execution across
    ranks. Build only binds the existing symmetric stage. Each call to run()
    must pair with one B1 target replay from the rank that received the request.
    """

    def __init__(self, model, *, ep_rank: int, device_id: int):
        config = model.model_args
        size = config.ep_size
        if ((config.num_hidden_layers, size) not in ((8, 2), (45, 6), (45, 8)) or not 0 <= ep_rank < size
                or (model.tp_size, model.tp_rank) != (1, 0)
                or config.ep_rank != ep_rank
                or config.attention_impl != "mk_decode" or config.routed_expert_format != mxfp4.FORMAT
                or (config.hidden_size, config.n_routed_experts, config.num_experts_per_tok)
                != (_HIDDEN_SIZE, _EXPERTS, _SEND_CAPACITY)):
            raise ValueError("MK companion requires GLM5.3 single-origin TP1 EP2/EP6/EP8 W4A4 decode")
        if model.device_id != device_id or torch.npu.current_device() != device_id:
            raise ValueError("MK companion requires the model's current NPU device")
        stage = getattr(builtins, "_mk_shmem_ptr", None)
        if stage is None or not stage.data_ptr():
            raise RuntimeError("Initialize MK SHMEM before building companion graphs")

        layers = model.model.model.layers
        self.layer_ids = tuple(block.layer_id for block in layers if block.is_layer_sparse)
        expected = tuple(range(config.first_k_dense_replace, config.num_hidden_layers))
        if (config.first_k_dense_replace != 3 or len(layers) != config.num_hidden_layers
                or not expected or self.layer_ids != expected):
            raise ValueError(f"Unexpected MK sparse layer sequence: {self.layer_ids}, expected {expected}")

        device = torch.device("npu", device_id)
        partition = mxfp4.ExpertPartition(_EXPERTS, size, ep_rank)
        self._local_experts = partition.count
        for layer_id in self.layer_ids:
            moe = layers[layer_id].mlp
            if not moe._mxfp4 or moe.expert_partition != partition:
                raise ValueError(f"Layer {layer_id} has the wrong MK expert partition for rank {ep_rank}")
            for name in ("w13_gate", "w13_up", "experts_w2"):
                for tensor in (getattr(moe, name), getattr(moe, name + "_mx_scale")):
                    if tensor.shape[0] != partition.count or tensor.device != device or tensor.dtype != torch.uint8:
                        raise ValueError(f"Layer {layer_id} has invalid rank {ep_rank} local {name} storage")

        self._stage = stage
        self._device_id = device_id
        self._failed = False
        with torch.inference_mode(False), torch.no_grad():
            self._expanded = torch.zeros((_SEND_CAPACITY, _HIDDEN_SIZE), dtype=torch.bfloat16, device=device)
            self._cumsum = torch.zeros((_EXPERTS,), dtype=torch.int64, device=device)
            self._graphs = tuple(self._capture_layer(layer_id, layers[layer_id].mlp)
                                 for layer_id in self.layer_ids)

    def _capture_layer(self, layer_id, moe):
        with DAGCapture() as capture:
            returned = mxfp4.grouped(moe, self._expanded, self._cumsum)
            comm = [node for node in capture.toposort() if node.op_name in _COMM_OPS]
            if tuple(node.op_name for node in comm) != _COMM_OPS:
                raise RuntimeError(f"Unexpected MK EP communication order in layer {layer_id}")
            local_cumsum = comm[1].outputs[2]
            if local_cumsum.shape != (self._local_experts,) or local_cumsum.dtype != torch.int64:
                raise RuntimeError(f"Unexpected MK EP dispatch cumsum in layer {layer_id}")
            capture.retain_output(returned)
            capture.retain_output(local_cumsum)
        graph = capture.build(device_id=self._device_id, num_experts=0, skip_steps=0,
                              callback="libdefault_callback.so", enable_static_fusion=False)
        return graph, local_cumsum

    def check_ready(self):
        """Check local bindings before either rank enters a collective."""
        if self._failed:
            raise RuntimeError("MK EP companion cannot replay after a failed step")
        if getattr(builtins, "_mk_shmem_ptr", None) is not self._stage:
            raise RuntimeError("MK SHMEM stage changed after companion capture")
        if torch.npu.current_device() != self._device_id:
            raise RuntimeError("MK EP companion replay is on the wrong NPU device")

    def run(self) -> tuple[LayerReceive, ...]:
        """Replay in target layer order and return actual received expert rows."""
        self.check_ready()
        receives = []
        try:
            for layer_id, (graph, local_cumsum) in zip(self.layer_ids, self._graphs):
                graph.run()
                torch.npu.synchronize()
                rows = int(local_cumsum[-1].item())
                if not 0 <= rows <= _SEND_CAPACITY:
                    raise RuntimeError(f"Layer {layer_id} received {rows} rows from one B1 source")
                receives.append(LayerReceive(layer_id, rows))
        except Exception:
            self._failed = True
            raise
        return tuple(receives)
