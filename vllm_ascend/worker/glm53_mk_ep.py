"""Coordinated GLM5.3 MK decode at VA's worker forward boundary."""

import builtins
import os
from dataclasses import dataclass
from itertools import chain
from struct import Struct

import torch
from vllm.distributed.parallel_state import get_dp_group, get_ep_group, get_tp_group
from vllm.logger import logger
from vllm_ascend.worker.glm53_mk_hook import GLM53MKHook


@dataclass(frozen=True)
class LayerPlan:
    depth: int
    size: int
    kda: tuple
    mla: tuple
    moe: tuple

    @classmethod
    def for_size(cls, size, *, tp_size=1, depth=None, native_experts=False):
        depths = {2: 8, 6: 45, 8: 45}
        if tp_size in (2, 8) and tp_size == size:
            _require(depth in (4, 8) or depth == 45 and native_experts,
                     f"45-layer TP{tp_size} requires shared native routed weights")
        else:
            _require(tp_size == 1 and size in depths, f"unsupported TP{tp_size}/EP{size}")
            depth = depths[size]
        return cls(depth, size, tuple(i for i in range(depth) if i % 4 != 3),
                   tuple(range(3, depth, 4)), tuple(range(3, depth)))


def _require(ok, message):
    if not ok:
        raise RuntimeError(f"GLM53 MK EP: {message}")


def _log_hbm(rank, phase):
    free, total = torch.npu.mem_get_info()
    logger.info("GLM53 MK EP rank=%d phase=%s hbm_free_gib=%.3f hbm_total_gib=%.3f",
                rank, phase, free / 2**30, total / 2**30)


def _topology(runner):
    cfg = runner.parallel_config
    return (cfg.tensor_parallel_size, cfg.pipeline_parallel_size, cfg.data_parallel_size,
            cfg.prefill_context_parallel_size, cfg.decode_context_parallel_size)


def _group_rows(group, size, value):
    rows = [None] * size
    torch.distributed.all_gather_object(rows, value, group=group)
    return rows


class RoundGate:
    def __init__(self, rank, exchange, size=2):
        _require(size in (2, 6, 8) and 0 <= rank < size, "invalid DP group size/rank")
        self.rank, self.exchange, self.size, self.sequence = rank, exchange, size, 0

    def gather(self, phase, payload):
        rows = self.exchange((self.rank, self.sequence, phase, payload))
        _require([(r, n, p) for r, n, p, _ in rows] == [(i, self.sequence, phase) for i in range(self.size)],
                 f"rank/order mismatch at round {self.sequence} {phase}")
        return [data for _, _, _, data in rows]

    def begin(self, role, request_id, position, error=None):
        self.sequence += 1
        rows = self.gather("select", (role, request_id, position, error))
        _require(all(not r[3] for r in rows), f"eligibility errors {rows}")
        roles = [r[0] for r in rows]
        _require(all(r in ("native", "real", "idle") for r in roles), "unknown rank role")
        if roles.count("real") != 1 or roles.count("idle") != self.size - 1:
            return None
        owner = roles.index("real")
        return owner, rows[owner][1], rows[owner][2]

    def begin_tp(self, request_id, position, token, error=None):
        return self._decide_tp("select", request_id, position, token, error)

    def admit_tp(self, request_id, position, token, error=None):
        return self._decide_tp("admit", request_id, position, token, error)

    def _decide_tp(self, phase, request_id, position, token, error):
        self.sequence += 1
        rows = self.gather(phase, (request_id, position, token, error))
        _require(all(row[3] is None for row in rows), f"TP {phase} eligibility errors {rows}")
        _require(all(row == rows[0] for row in rows), f"TP workers disagree on the decode token {rows}")
        return None if request_id is None else (0, request_id, position)

    def check(self, phase, error=None):
        errors = self.gather(phase, error)
        _require(not any(errors), f"round {self.sequence} {phase} errors {errors}")


_TP_PHASES = {"select": 0, "ready": 1, "complete": 2, "admit": 3}
_TP_REQUEST_BYTES = 96
_TP_NO_REQUEST = -1
_TP_NEEDS_DETAILS = -2
# Rank, sequence, phase, request length or sentinel, position, token, request bytes.
_TP_MESSAGE = Struct(f"<qqBhqq{_TP_REQUEST_BYTES}s")


class _TPRoundGate(RoundGate):
    def __init__(self, rank, group, size):
        super().__init__(rank, lambda value: _group_rows(group, size, value), size)
        self._group = group
        self._local_control = None
        self._send_bytes = bytearray(_TP_MESSAGE.size)
        self._recv_bytes = [bytearray(_TP_MESSAGE.size) for _ in range(size)]
        # Gloo fills the receive tensors on its worker threads.
        with torch.inference_mode(False):
            self._send = torch.frombuffer(self._send_bytes, dtype=torch.uint8)
            self._recv = [torch.frombuffer(buffer, dtype=torch.uint8) for buffer in self._recv_bytes]

    def enable_local_control(self):
        from vllm_ascend.worker.glm53_mk_tp_control import LocalTPControl

        self._local_control = LocalTPControl.connect(self, _TP_MESSAGE.size)

    def close_local_control(self):
        if self._local_control is not None:
            self._local_control.close()

    def gather(self, phase, payload):
        phase_id = _TP_PHASES.get(phase)
        if phase_id is None:
            return super().gather(phase, payload)

        length, position, token, request = _TP_NO_REQUEST, 0, 0, b""
        if phase in ("select", "admit"):
            request_id, pos, tok, error = payload
            if error is not None:
                length = _TP_NEEDS_DETAILS
            elif request_id is not None or pos is not None or tok is not None:
                length = _TP_NEEDS_DETAILS
                if (isinstance(request_id, str) and type(pos) is int and type(tok) is int
                        and -(1 << 63) <= pos < 1 << 63 and -(1 << 63) <= tok < 1 << 63):
                    encoded = request_id.encode("utf-8", errors="surrogatepass")
                    if len(encoded) <= _TP_REQUEST_BYTES:
                        length, position, token, request = len(encoded), pos, tok, encoded
        elif payload:
            length = _TP_NEEDS_DETAILS

        # Every runtime phase must start with the same collective shape, so
        # workers can report a phase mismatch before any rank exchanges details.
        _TP_MESSAGE.pack_into(self._send_bytes, 0, self.rank, self.sequence, phase_id,
                              length, position, token, request)
        if self._local_control is None:
            torch.distributed.all_gather(self._recv, self._send, group=self._group)
            rows = [_TP_MESSAGE.unpack_from(buffer) for buffer in self._recv_bytes]
        else:
            gathered = self._local_control.all_gather(self._send_bytes)
            rows = [_TP_MESSAGE.unpack_from(gathered, rank * _TP_MESSAGE.size) for rank in range(self.size)]
        _require(all(row[:3] == (rank, self.sequence, phase_id) for rank, row in enumerate(rows)),
                 f"rank/order mismatch at round {self.sequence} {phase}")
        if any(row[3] == _TP_NEEDS_DETAILS for row in rows):
            return super().gather(phase, payload)
        if phase not in ("select", "admit"):
            return [None] * self.size
        return [(request[:length].decode("utf-8", errors="surrogatepass"), pos, tok, None)
                if length >= 0 else (None, None, None, None)
                for _, _, _, length, pos, tok, request in rows]


@dataclass
class _Round:
    owner: int
    request: str
    position: int
    admission_token: int | None = None
    readiness_started: bool = False
    launch_allowed: bool = False


class _CoordinatedDecodeState:
    """Delay the actual state.run until the peer's companion also passed preparation."""

    def __init__(self, state, hook):
        self.state, self.hook = state, hook

    def __getattr__(self, name):
        return getattr(self.state, name)

    @property
    def require_cached_graph(self):
        active = self.hook.active
        return active is not None and active.admission_token is not None

    def run(self, graph):
        if graph is not self.hook.verified_graph:
            _require(not self.require_cached_graph, "TP admission requires the previous MK graph")
            ordinary_ops = ("moe_metadata_exchange", "moe_dispatch", "moe_combine")
            typed_owner = self.hook.replicated_owner and os.getenv("MK_GLM53_MOE_DIRECT_OWNER_PLAN") == "1"
            owner_plan_op = "moe_gating_topk_owner_plan"
            owner_ops = (owner_plan_op if typed_owner else "moe_owner_cumsum", "moe_combine")
            expected_ops = owner_ops if self.hook.replicated_owner else ordinary_ops
            expected = [(f"layer{layer}", op) for layer in self.hook.plan.moe
                        for op in expected_ops]
            sequence = [(node.layer_id, node.op_name) for node in graph.nodes
                        if node.op_name in (*ordinary_ops, "moe_owner_cumsum", owner_plan_op)]
            _require(sequence == expected, f"full decoder EP collective order {sequence}")
            if self.hook.replicated_owner:
                _require(all(node.attrs.get("replicated_owner") == 1
                             and node.attrs.get("replicated_owner_endpoints", 0) == int(typed_owner)
                             for node in graph.nodes if node.op_name == "moe_combine"),
                         "full decoder MoE owner combine layout differs")
            self.hook.verified_graph = graph
            _log_hbm(self.hook.gate.rank, "full_graph_built")
        # The state manager also validates its KDA lifecycle in begin_step.
        # Place the rendezvous at graph.run, after that final local check.
        hook = self.hook

        class ReadyGraph:
            def run(self):
                hook.ready_to_launch()
                return graph.run()

        self.state.run(ReadyGraph())


class GLM53MKEPHook(GLM53MKHook):
    def __init__(self, bridge, text, names, model, runtime, companion, gate, idle_hidden, *, plan=None,
                 fuse_tp_admission=False, replicated_owner=False):
        super().__init__(bridge, text, names)
        self.runtime, self.companion, self.gate = runtime, companion, gate
        self.plan = plan or LayerPlan.for_size(gate.size)
        self.tp_size = model.tp_size
        self.idle_hidden = idle_hidden
        self.verified_graph = self.active = self.result = None
        self.invalid = False
        self._fuse_tp_admission = fuse_tp_admission
        self.replicated_owner = replicated_owner
        self.native_builder_deferral = os.getenv("MK_GLM53_NATIVE_BUILDER_DEFERRAL", "off")
        self._address_builder = None
        self._address_builder_unavailable = False
        self._address_plan_reported = False
        self._address_miss_reported = False
        self._report_address_steps = os.getenv("MK_GLM53_NATIVE_BUILDER_TRACE") == "1"
        self._last_completed_round = None
        # VAMKBridge prepares the original state. Model forward reaches this proxy
        # after it validates its inputs, builds its graph and checks bound addresses.
        model._mk_decode_state = _CoordinatedDecodeState(model._mk_decode_state, self)

    def prepare_cpu_addresses(self, runner, output):
        if (self.native_builder_deferral not in ("shadow", "on") or self.tp_size != 8
                or not self._fuse_tp_admission or self._last_completed_round is None
                or self.invalid or self.active is not None or self.verified_graph is None
                or self._address_builder_unavailable):
            return None
        parallel = runner.parallel_config
        if (parallel.data_parallel_size, parallel.pipeline_parallel_size,
                parallel.prefill_context_parallel_size, parallel.decode_context_parallel_size) != (1, 1, 1, 1):
            return None
        if (runner.speculative_config is not None or runner.use_async_scheduling
                or runner.use_async_spec_decode or runner.use_dcp or runner.dynamic_eplb
                or (runner.device_metadata_executor is not None and (
                    runner.device_metadata_executor.submission_in_flight
                    or runner.device_metadata_executor.capture_producers))
                or runner.long_seq_metadata is not None
                or runner.sparse_kv_offload_enabled or runner.is_mm_prefix_lm
                or runner.kvpp.scheduler is not None or runner.cache_config.enable_prefix_caching
                or runner.cache_config.mamba_cache_mode != "none" or not self.replicated_owner):
            return None
        scheduled = self._schedule(runner, output)
        if scheduled is None:
            return None
        rid, position, prompt, count = scheduled
        if (count != 1 or not prompt < position < self.bridge.max_context_tokens
                or (rid, position) != (self.request_id, self.next_position)
                or (rid, position - 1) != self._last_completed_round
                or (rid, position - 1) != (self.bridge._request, self.bridge._position)):
            return None
        if os.getenv("MK_GLM53_NATIVE_BUILDER_MISS") == f"{self.gate.rank}:{position - prompt}":
            logger.info("GLM53 native builder deferral forced local miss rank=%d position=%d", self.gate.rank, position)
            return None
        try:
            from vllm_ascend.worker.glm53_mk_step_address_plan import (
                CpuStepAddressBuilder, UnsupportedAddressPlan, early_indexer_cache,
            )
        except ImportError:
            logger.exception("GLM53 native builder deferral unavailable on rank=%d", self.gate.rank)
            self._address_builder_unavailable = True
            return None
        try:
            if self._address_builder is None:
                self._address_builder = CpuStepAddressBuilder(runner)
            kda, mla, indexer = self._cache_descriptors(runner, bound_indexer=early_indexer_cache)
            plan = self._address_builder.read_checked(runner, position, kda, mla, indexer, scheduler_output=output)
            if (plan.request_id, plan.position) != (rid, position):
                return None
            addresses = self.bridge.checked_cpu_addresses(position, kda, mla, indexer, plan)
        except (AttributeError, IndexError, KeyError, UnsupportedAddressPlan) as exc:
            if self._report_address_steps and not self._address_miss_reported:
                logger.info("GLM53 native builder deferral local miss rank=%d position=%d reason=%s",
                            self.gate.rank, position, exc)
                self._address_miss_reported = True
            return None
        if addresses is None:
            return None
        return (rid, position), (kda, mla, indexer), addresses

    @staticmethod
    def _early_cache_bindings_match(runner, descriptors):
        from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend

        kda, mla, indexer = descriptors
        registry = runner.compilation_config.static_forward_context
        bound = Glm5NextKPoolIndexerBackend._bound_cache
        try:
            for cache in kda.values():
                owner = registry[cache.prefix]
                conv, recurrent = owner.kv_cache
                if (conv is not cache.conv or recurrent is not cache.recurrent
                        or owner._conv_state_dim_first != cache.conv_dim_first):
                    return False
            for cache in mla.values():
                latent, rope = registry[cache.prefix].kv_cache
                if latent is not cache.latent or rope.shape[-1] != 0:
                    return False
            return all(bound(registry[cache.pool_prefix]) is cache.pool
                       and bound(registry[cache.tail_prefix]) is cache.tail for cache in indexer.values())
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            return False

    def _cache_descriptors(self, runner, *, bound_indexer=None):
        from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
        from vllm_ascend.worker.va_mk_bridge import IndexerCache, KDACache, MLACache

        registry = runner.compilation_config.static_forward_context
        kda, mla, indexer = {}, {}, {}
        for i in self.plan.kda:
            name = self.names[i]
            owner = registry[name]
            conv, recurrent = owner.kv_cache
            kda[i] = KDACache(name, conv, recurrent, owner._conv_state_dim_first)
        bound = Glm5NextKPoolIndexerBackend._bound_cache if bound_indexer is None else bound_indexer
        for i in self.plan.mla:
            mn, pn, tn = self.names[i]
            latent, rope = registry[mn].kv_cache
            _require(rope.shape[-1] == 0, "native MLA cache has nonempty rotary head")
            mla[i] = MLACache(mn, latent)
            indexer[i] = IndexerCache(pn, tn, bound(registry[pn]), bound(registry[tn]))
        return kda, mla, indexer

    def ready_to_launch(self):
        active = self.active
        _require(active is not None and not active.readiness_started, "duplicate or unscheduled EP launch")
        # Complete any local metadata, captured graph, or cache-import copies
        # before telling the other rank that SHMEM operators may execute.
        torch.npu.synchronize()
        self._check_ready(active)
        active.launch_allowed = True
        self.runtime.barrier()

    def _check_ready(self, active, error=None):
        active.readiness_started = True
        if active.admission_token is None:
            self.gate.check("ready", error)
        else:
            self.gate.admit_tp(active.request, active.position, active.admission_token, error)

    def _vote(self, runner, output, idle, context, num_tokens):
        if self.invalid or self.active is not None:
            raise RuntimeError("previous EP generation failed or is still active")
        _require(idle == (output is None), "ambiguous schedule and explicit runtime-idle marker")
        if num_tokens != 1 or context.cudagraph_runtime_mode.name != "NONE" or getattr(context, "capturing", False):
            return "native", None, None
        if idle:
            return "idle", None, None
        scheduled = self._schedule(runner, output)
        if scheduled is not None:
            rid, pos, prompt, count = scheduled
            if ((rid, pos) == (self.request_id, self.next_position) and count == 1
                    and 0 < pos < getattr(self.bridge, "max_context_tokens", 64) and pos >= prompt):
                return "real", rid, pos
        return "native", None, None

    def forward_round(self, runner, output, idle, context, num_tokens, input_ids, positions):
        self.result = None
        previous_round, self._last_completed_round = self._last_completed_round, None
        row = ("native", None, None)
        error, token = None, None
        tp_size = self.tp_size
        try:
            prepare_error = getattr(context, "glm53_metadata_prepare_error", None)
            if prepare_error is not None:
                raise prepare_error
            row = self._vote(runner, output, idle, context, num_tokens)
            if row[0] != "real" and getattr(context, "glm53_cpu_addresses", None) is not None:
                materialize = getattr(context, "glm53_materialize_native_attention_metadata", None)
                if materialize is not None:
                    materialize()
                context.glm53_cpu_addresses = None
            if tp_size in (2, 8):
                _require(not idle, f"TP{tp_size} requires the real request on every worker")
                if row[0] == "real":
                    inputs = runner.input_ids.gpu[:1] if input_ids is None else input_ids
                    _require(inputs.shape == (1,), f"TP{tp_size} requires one token on each worker")
                    token = int(inputs.item())
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        if previous_round is not None:
            # All ranks arm admission after the same successful completion.
            # A rank choosing Native or seeing an error still joins that phase.
            request, position = previous_round
            if row[0] == "real" and error is None and (row[1], row[2]) != (request, position + 1):
                error = "TP admission requires the next position of the completed request"
            if row[0] == "real" and error is None:
                decision = (0, row[1], row[2])
                self.active = _Round(*decision, admission_token=token)
            else:
                try:
                    decision = self.gate.admit_tp(row[1], row[2], token, error)
                except Exception:
                    self.invalid = True
                    raise
        elif tp_size in (2, 8):
            decision = self.gate.begin_tp(row[1], row[2], token, error=error)
        else:
            decision = self.gate.begin(*row, error=error)
        if decision is None:
            logger.info("GLM53 EP round=%d rank=%d local=%s backend=native",
                        self.gate.sequence, self.gate.rank, row[0])
            return None
        if self.active is None:
            self.active = _Round(*decision)
        active = self.active
        receipt = None
        try:
            if idle:
                self.companion.check_ready()
                self.ready_to_launch()
                receipt = self.companion.run()
                _require(tuple(report.layer_id for report in receipt) == self.plan.moe, "companion layer mismatch")
                result = self.idle_hidden
            else:
                self.result = self.try_decode(runner, output, context, positions, input_ids)
                _require(self.result is not None and active.launch_allowed, "scheduled full MK launch did not run")
                result = self.result.hidden_states
            torch.npu.synchronize()
        except Exception as exc:
            self.invalid, self.result = True, None
            detail = f"{type(exc).__name__}: {exc}"
            if not active.readiness_started:
                self._check_ready(active, detail)
            elif active.launch_allowed:
                self.gate.check("complete", detail)
            raise
        finally:
            self.active = None
        try:
            self.gate.check("complete")
            self.runtime.barrier()
        except Exception:
            self.invalid, self.result = True, None
            raise
        if self._fuse_tp_admission:
            self._last_completed_round = (active.request, active.position)
        logger.info("GLM53 EP round=%d rank=%d origin_rank=%d request=%s position=%d role=%s "
                    "origin_tokens=%d backend=mk cache_writeback=%s layer_receive=%s tp_size=%d admission=%s",
                    self.gate.sequence, self.gate.rank, active.owner, active.request, active.position,
                    "companion" if idle else "real", 0 if idle else 1, "none" if idle else "complete",
                    [(r.layer_id, r.receive_rows) for r in receipt] if idle else None, tp_size,
                    "fused" if active.admission_token is not None else "separate")
        return result

    def take_step(self):
        result, self.result = self.result, None
        return result

    def close(self):
        try:
            self.runtime.close()
        finally:
            if isinstance(self.gate, _TPRoundGate):
                self.gate.close_local_control()


def _check_native_experts(parallel, layers, plan, rank):
    local = 288 // plan.size
    _require(parallel.expert_placement_strategy == "linear" and not parallel.enable_eplb,
             "requires contiguous native expert ownership with static placement")
    for i in plan.moe:
        moe = layers[i].mlp
        _require((moe.ep_rank, moe.ep_size, moe.n_logical_experts, moe.n_redundant_experts,
                  moe.n_local_physical_experts, moe.physical_expert_start, moe.physical_expert_end)
                 == (rank, plan.size, 288, 0, local, rank * local, (rank + 1) * local),
                 f"native expert ownership differs from MK on layer {i}, rank {rank}")


def _weight_source(loader, plan, tp_size=1, tp_rank=0, *, native_experts=False, capture_lm_head=True):
    if tp_size == 1 and plan.size == 2:
        weights = loader.load()
        return {"weight_dict": weights}, loader.kda_quant_config_by_layer
    kwargs = {"native_experts": True} if native_experts else {}
    if not capture_lm_head:
        kwargs["include_lm_head"] = False
    batches = iter(loader.iter_decode_weights(tp_size=tp_size, tp_rank=tp_rank, **kwargs))
    first = next(batches)
    _require(first[0] is None, "streamed checkpoint must begin with text globals")
    return {"decode_weight_batches": chain((first,), batches)}, loader.kda_quant_config_by_layer


def _construct(runner, rank, runtime, gate, *, fuse_tp_admission=False, replicated_owner=False):
    from blockrt.models.glm_5_3_flash.glm5_3_flash_megakernel import Glm53FlashMegaKernel
    from blockrt.models.glm_5_3_flash.model_args import ModelArgsGLM5_3Flash
    from blockrt.models.glm_5_3_flash.weight_loader import WeightLoader
    from vllm_ascend.worker.glm53_mk_ep_companion import Glm53MKEPCompanion
    from vllm_ascend.worker.va_mk_bridge import VAMKBridge

    cfg = runner.vllm_config
    _require(runner.model_config.enforce_eager and runner.scheduler_config.max_num_seqs == 1
             and runner.speculative_config is None and not runner.broadcast_pp_output
             and not runner.use_aux_hidden_state_outputs and not runner.use_async_scheduling
             and not runner.dynamic_eplb and cfg.kv_transfer_config is None
             and getattr(cfg, "ec_transfer_config", None) is None,
             "requires synchronous eager single-request VA with MTP, EPLB and KV transfer disabled")
    tp_size = _topology(runner)[0]
    tp_rank = rank if tp_size in (2, 8) else 0
    capacity = int(os.environ.get("VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS", "64"))
    _require(capacity == 64 or tp_size in (2, 8) and 64 < capacity <= 4095 * 64,
             "extended context requires TP2/EP2 or TP8/EP8 and at most 262080 tokens")
    text = getattr(runner.model, "language_model", runner.model)
    depth = text.config.num_hidden_layers
    native_experts = tp_size in (2, 8) and (depth == 45
                                      or os.environ.get("VLLM_ASCEND_GLM53_MK_NATIVE_EXPERTS") == "1")
    capture_lm_head = not (native_experts and depth == 45)
    plan = LayerPlan.for_size(gate.size, tp_size=tp_size, depth=depth, native_experts=native_experts)
    fine = runner.ascend_config.finegrained_tp_config
    _require(all(size <= 1 for size in (fine.oproj_tensor_parallel_size, fine.lmhead_tensor_parallel_size,
                                        fine.embedding_tensor_parallel_size, fine.mlp_tensor_parallel_size)),
             "requires local LM head and local dense/embedding weights")
    args, native = ModelArgsGLM5_3Flash(), text.config
    args.num_hidden_layers, args.num_nextn_predict_layers = plan.depth, 0
    args.ep_rank, args.ep_size = rank, plan.size
    args.attention_impl, args.routed_expert_format = "mk_decode", "W4A4_MXFP4"
    args.mxfp8_decode, args.indexer_skip_rope = True, True
    layers = text.model.layers
    _require(native.model_type == "glm5_next_text" and native.num_hidden_layers == len(layers) == plan.depth and native.qk_rope_head_dim == 0
             and native.index_kpool == 4 and native.index_topk == 2048 and native.index_kpool_compress
             and native.index_kpool_always_select_tail and native.vocab_size == args.vocab_size
             and native.hidden_size == 4096 and native.n_routed_experts == 288,
             f"requires the {plan.depth}-layer NoPE/KPool GLM-Next checkpoint")
    _require(tuple(i for i, l in enumerate(layers) if l.layer_kind == "kda") == plan.kda
             and tuple(i for i, l in enumerate(layers) if l.layer_kind == "mla") == plan.mla,
             "unexpected native KDA/MLA layer order")
    _require(tuple(args.linear_layer_ids) == plan.kda and tuple(args.full_attention_layer_ids) == plan.mla
             and args.first_k_dense_replace == 3, "MK layer schedule differs from native checkpoint")
    _check_native_experts(runner.parallel_config, layers, plan, rank)
    owners = ({i: layers[i].mlp.experts.routed_experts for i in plan.moe} if native_experts else None)
    keys = {id(owner): name for name, owner in runner.compilation_config.static_forward_context.items()}
    names = {i: keys[id(layers[i].self_attn)] for i in plan.kda}
    for i in plan.mla:
        attn = layers[i].self_attn
        names[i] = tuple(keys[id(owner)] for owner in
                         (attn.mla_attn.mla_attn, attn.indexer.k_cache, attn.indexer.tail_cache))
    path = os.environ.get("VLLM_ASCEND_GLM53_MK_CHECKPOINT") or runner.model_config.model
    with torch.inference_mode(False), torch.no_grad():
        _log_hbm(rank, "before_mk_weights")
        loader = WeightLoader(args, path)
        source, schemes = _weight_source(loader, plan, tp_size, tp_rank, native_experts=native_experts,
                                         capture_lm_head=capture_lm_head)
        _require(set(schemes) == set(plan.kda) and
                 all(len(p) == 9 and set(p.values()) == {"W8A8_MXFP8"} for p in schemes.values()),
                 f"requires all {9 * len(plan.kda)} KDA MXFP8 projections")
        rope = torch.zeros((64, 64), dtype=torch.bfloat16, device=runner.device)
        rope[:, :32].fill_(1)
        model = Glm53FlashMegaKernel(args, tp_size=tp_size, tp_rank=tp_rank, index_rope_cache=rope,
                                     native_expert_owners=owners, capture_lm_head=capture_lm_head,
                                     kda_quant_config_by_layer=schemes, **source)
        del source
        _log_hbm(rank, "mk_weights_loaded")
        if native_experts:
            logger.info("GLM53 MK EP rank=%d native_expert_layers=%d aliased=true",
                        rank, len(model.native_routed_layers))
        if not capture_lm_head:
            logger.info("GLM53 MK EP rank=%d lm_head=va-native", rank)
        _require(model.device_id == torch.npu.current_device(), "MK and VA devices differ")
        bridge_options = {"max_context_tokens": capacity} if capacity != 64 else {}
        if tp_size == 8 and os.environ.get("MK_GLM53_REUSE_INDEXER_TAIL") == "1":
            bridge_options["reuse_indexer_tail"] = True
        if tp_size == 8 and os.environ.get("MK_GLM53_REUSE_KDA_VIEWS") == "1":
            bridge_options["reuse_kda_views"] = True
        if tp_size == 8 and os.environ.get("MK_GLM53_REUSE_PAGED_METADATA") == "1":
            bridge_options["reuse_paged_metadata"] = True
        if tp_size == 8 and os.environ.get("MK_GLM53_BATCH_PREFIX_COPIES") == "1":
            bridge_options["batch_prefix_copies"] = True
        if tp_size == 8 and os.environ.get("MK_GLM53_REUSE_VA_PAGE_VALIDATION") == "1":
            bridge_options["reuse_va_page_validation"] = True
        if tp_size == 8 and os.environ.get("MK_GLM53_BATCH_ATTENTION_WRITEBACK") == "1":
            bridge_options["batch_attention_writeback"] = True
        if tp_size == 8 and os.environ.get("MK_GLM53_BATCH_KDA_WRITEBACK") == "1":
            bridge_options["batch_kda_writeback"] = True
        bridge = VAMKBridge(model, ep_size=plan.size, **bridge_options)
        companion = idle_hidden = None
        if tp_size == 1:
            companion = Glm53MKEPCompanion(model, ep_rank=rank, device_id=model.device_id)
            _log_hbm(rank, "companion_graphs_built")
            _require(tuple(companion.layer_ids) == plan.moe, "companion omitted routed layers")
            idle_hidden = torch.zeros((1, 4096), dtype=torch.bfloat16, device=runner.device)
    return GLM53MKEPHook(bridge, text, names, model, runtime, companion, gate, idle_hidden, plan=plan,
                         fuse_tp_admission=fuse_tp_admission, replicated_owner=replicated_owner)


def load_ep_hook(runner):
    enabled = os.environ.get("VLLM_ASCEND_GLM53_MK_EP") == "1"
    topology = _topology(runner)
    tp_size = topology[0]
    size = tp_size if tp_size in (2, 8) else topology[2]
    supported = ((1, 1, 2, 1, 1), (1, 1, 6, 1, 1), (1, 1, 8, 1, 1), (2, 1, 1, 1, 1), (8, 1, 1, 1, 1))
    if topology not in supported or not runner.parallel_config.enable_expert_parallel:
        _require(not enabled, "requires TP1 with DP/EP2, 6 or 8, or TP2/DP1/EP2 or TP8/DP1/EP8 with PP/CP1")
        return None
    group = get_tp_group() if tp_size in (2, 8) else get_dp_group()
    rank = group.rank_in_group
    if enabled and tp_size in (2, 8):
        gate = _TPRoundGate(rank, group.cpu_group, size)
    else:
        gate = RoundGate(rank, lambda item: _group_rows(group.cpu_group, size, item), size)
    endpoint = os.environ.get("VLLM_ASCEND_GLM53_MK_EP_ENDPOINT")
    path = os.environ.get("VLLM_ASCEND_GLM53_MK_CHECKPOINT") or runner.model_config.model
    other = any(os.environ.get(k) == "1" for k in ("VLLM_ASCEND_GLM53_MK_DECODE", "VLLM_ASCEND_GLM53_MK_MTP"))
    shared = os.environ.get("VLLM_ASCEND_GLM53_MK_NATIVE_EXPERTS") == "1"
    capacity = os.environ.get("VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS", "64") if enabled else None
    single_origin = enabled and tp_size == 8 and os.environ.get("MK_GLM53_MOE_SINGLE_ORIGIN") == "1"
    replicated_owner = enabled and tp_size == 8 and os.environ.get("MK_GLM53_MOE_REPLICATED_OWNER") == "1"
    reuse_indexer_tail = enabled and tp_size == 8 and os.environ.get("MK_GLM53_REUSE_INDEXER_TAIL") == "1"
    reuse_kda_views = enabled and tp_size == 8 and os.environ.get("MK_GLM53_REUSE_KDA_VIEWS") == "1"
    pack_kda_qkv = enabled and tp_size == 8 and os.environ.get("MK_GLM53_KDA_PACK_QKV") == "1"
    packed_swiglu = enabled and os.environ.get("MK_GLM53_MOE_PACKED_SWIGLU") == "1"
    fuse_tp_admission = enabled and tp_size == 8 and os.environ.get("MK_GLM53_FUSE_TP_ADMISSION") == "1"
    reuse_paged_metadata = (enabled and tp_size == 8
                            and os.environ.get("MK_GLM53_REUSE_PAGED_METADATA") == "1")
    batch_prefix_copies = enabled and tp_size == 8 and os.environ.get("MK_GLM53_BATCH_PREFIX_COPIES") == "1"
    reuse_va_page_validation = enabled and tp_size == 8 and os.environ.get("MK_GLM53_REUSE_VA_PAGE_VALIDATION") == "1"
    local_tp_control = enabled and tp_size == 8 and os.environ.get("MK_GLM53_LOCAL_TP_CONTROL") == "1"
    batch_attention = enabled and tp_size == 8 and os.environ.get("MK_GLM53_BATCH_ATTENTION_WRITEBACK") == "1"
    batch_kda = enabled and tp_size == 8 and os.environ.get("MK_GLM53_BATCH_KDA_WRITEBACK") == "1"
    voted = gate.gather("enable", (enabled, endpoint if enabled else None, path if enabled else None,
                                   other, shared, capacity, single_origin, reuse_indexer_tail,
                                   reuse_kda_views, pack_kda_qkv, packed_swiglu, fuse_tp_admission,
                                   reuse_paged_metadata, batch_prefix_copies, reuse_va_page_validation,
                                   local_tp_control, replicated_owner, batch_attention, batch_kda))
    _require(all(v == voted[0] for v in voted), f"inconsistent worker configuration {voted}")
    _require(not (single_origin and replicated_owner), "single origin and replicated MoE owner are exclusive")
    if not enabled:
        return None
    ep = get_ep_group()
    local_error = None
    if (not endpoint or other or ep.world_size != size or group.world_size != size
            or ep.rank_in_group != rank or runner.dp_rank != (0 if tp_size in (2, 8) else rank)
            or tuple(ep.ranks) != tuple(group.ranks)):
        local_error = "dedicated MK endpoint, exclusive feature and matching ordered EP/worker groups required"
    try:
        from blockrt.dist.ep_runtime import ExpertParallelRuntime
        _require(getattr(builtins, "_mk_shmem_ptr", None) is None, "another MK SHMEM runtime owns this process")
        library = ExpertParallelRuntime.load_bootstrap()
        for symbol in ("mk_ep_init", "mk_ep_pointer", "mk_ep_barrier", "mk_ep_close"):
            getattr(library, symbol)
    except Exception as exc:
        local_error = f"{type(exc).__name__}: {exc}"
    gate.check("bootstrap", local_error)
    if local_tp_control:
        gate.enable_local_control()
    try:
        runtime = ExpertParallelRuntime(rank, size, endpoint)
    except Exception:
        if local_tp_control:
            gate.close_local_control()
        raise
    hook, error = None, None
    try:
        hook = _construct(runner, rank, runtime, gate, fuse_tp_admission=fuse_tp_admission,
                          replicated_owner=replicated_owner)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    try:
        gate.check("loaded", error)
    except Exception:
        try:
            runtime.close()
        finally:
            if local_tp_control:
                gate.close_local_control()
        raise
    local_experts = 288 // size
    logger.info("GLM53 MK EP loaded rank=%d/%d depth=%d experts=[%d,%d) moe_layers=%s tp_size=%d",
                rank, size, hook.plan.depth, rank * local_experts, (rank + 1) * local_experts, hook.plan.moe, tp_size)
    return hook
