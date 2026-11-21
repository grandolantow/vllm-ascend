"""Eager GLM-5.3 four-layer, single-request MK decode in the Ascend v1 runner."""

import os
from copy import copy
from dataclasses import dataclass

import torch
from vllm.logger import logger as _LOG


@dataclass
class _Step:
    request_id: str
    position: int
    hidden_states: torch.Tensor
    raw_logits: torch.Tensor | None
    logits_consumed: bool = False


class GLM53MKHook:
    def __init__(self, bridge, text, names):
        self.bridge = bridge
        self.text = text
        self.names = names
        # Preserve the platform-selected processor class and all native options.
        self.logits_processor = copy(text.logits_processor)
        self.logits_processor.logits_as_input = True
        self.request_id = None
        self.next_position = None

    @classmethod
    def from_runner(cls, runner):
        from blockrt.models.glm_5_3_flash.glm5_3_flash_megakernel import Glm53FlashMegaKernel
        from blockrt.models.glm_5_3_flash.model_args import ModelArgsGLM5_3Flash
        from blockrt.models.glm_5_3_flash.weight_loader import WeightLoader
        from vllm_ascend.worker.va_mk_bridge import VAMKBridge

        vcfg = runner.vllm_config
        parallel = vcfg.parallel_config
        topology = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size,
                    parallel.prefill_context_parallel_size, parallel.decode_context_parallel_size)
        if (topology != (1, 1, 1, 1, 1) or not runner.model_config.enforce_eager
                or runner.speculative_config is not None or runner.scheduler_config.max_num_seqs != 1
                or runner.broadcast_pp_output or runner.use_aux_hidden_state_outputs
                or vcfg.kv_transfer_config is not None or getattr(vcfg, "ec_transfer_config", None) is not None):
            raise ValueError("GLM53 MK decode requires eager single-worker TP1/EP1, max_num_seqs=1, no MTP or KV transfer")
        text = getattr(runner.model, "language_model", runner.model)
        cfg = text.config
        layers = text.model.layers
        if (cfg.model_type != "glm5_next_text" or len(layers) != 4 or cfg.qk_rope_head_dim != 0
                or cfg.index_kpool != 4 or cfg.index_topk != 2048 or not cfg.index_kpool_compress
                or not cfg.index_kpool_always_select_tail
                or tuple(layer.layer_kind for layer in layers) != ("kda", "kda", "kda", "mla")):
            raise ValueError("GLM53 MK decode requires the four-layer NoPE/KPool GLM-Next text checkpoint")
        if getattr(text.logits_processor, "head_dtype", None) not in (None, torch.bfloat16):
            raise ValueError("GLM53 MK decode requires the BF16 LM head")

        sparse = layers[3].self_attn
        owners = [layers[i].self_attn for i in (0, 1, 2)]
        owners += [sparse.mla_attn.mla_attn, sparse.indexer.k_cache, sparse.indexer.tail_cache]
        registry = runner.compilation_config.static_forward_context
        by_id = {id(layer): name for name, layer in registry.items()}
        names = tuple(by_id[id(owner)] for owner in owners)
        for index in (0, 1, 2, 4, 5):
            if owners[index].prefix != names[index]:
                raise ValueError(f"GLM53 cache owner prefix differs from runtime metadata key {names[index]}")

        args = ModelArgsGLM5_3Flash()
        args.num_hidden_layers = 4
        args.num_nextn_predict_layers = 0
        args.attention_impl = "mk_decode"
        args.mxfp8_decode = True
        args.indexer_skip_rope = True
        args.routed_expert_format = "W4A4_MXFP4"
        if (cfg.vocab_size, cfg.hidden_size) != (args.vocab_size, args.hidden_size):
            raise ValueError("GLM53 checkpoint vocabulary or hidden size differs from the MK model")
        path = os.environ.get("VLLM_ASCEND_GLM53_MK_CHECKPOINT") or runner.model_config.model
        _LOG.info("GLM53 MK loading four layers TP1/EP1 checkpoint=%s cache_keys=%s", path, names)
        with torch.inference_mode(False), torch.no_grad():
            loader = WeightLoader(args, path)
            weights = loader.load()
            schemes = loader.kda_quant_config_by_layer
            if (set(schemes) != {0, 1, 2} or
                    any(len(p) != 9 or set(p.values()) != {"W8A8_MXFP8"} for p in schemes.values())):
                raise ValueError("GLM53 MK requires nine W8A8_MXFP8 KDA projections on each of layers 0, 1 and 2")
            rope = torch.zeros((64, 64), dtype=torch.bfloat16, device=runner.device)
            rope[:, :32].fill_(1)
            model = Glm53FlashMegaKernel(args, weight_dict=weights, tp_size=1, tp_rank=0,
                                         index_rope_cache=rope, kda_quant_config_by_layer=schemes)
            del weights
            bridge = VAMKBridge(model)
        _LOG.info("GLM53 MK loaded four layers TP1/EP1; VA owns prefill and sampling")
        return cls(bridge, text, names)

    def _release(self, reason):
        if self.request_id is not None:
            self.bridge.release(self.request_id)
            _LOG.info("GLM53 MK release request=%s next_position=%s reason=%s",
                      self.request_id, self.next_position, reason)
        self.request_id = None
        self.next_position = None

    def update_requests(self, output):
        if self.request_id is None:
            return
        new_ids = {entry.req_id for entry in output.scheduled_new_reqs}
        if (self.request_id in output.finished_req_ids or self.request_id in new_ids
                or self.request_id in output.scheduled_cached_reqs.resumed_req_ids
                or self.request_id in (getattr(output, "preempted_req_ids", None) or ())):
            self._release("finished, new lifetime or preemption")

    @staticmethod
    def _schedule(runner, output):
        batch = runner.input_batch
        if batch.num_reqs != 1 or len(output.num_scheduled_tokens) != 1:
            return None
        rid = batch.req_ids[0]
        count = output.num_scheduled_tokens.get(rid)
        if count is None or count <= 0 or count != output.total_num_scheduled_tokens:
            return None
        req = runner.requests[rid]
        if (req.mm_features or req.prompt_embeds is not None or req.lora_request is not None
                or req.sampling_params is None or req.sampling_params.prompt_logprobs is not None
                or output.scheduled_spec_decode_tokens):
            return None
        return rid, int(batch.num_computed_tokens_cpu[0]), int(batch.num_prompt_tokens[0]), int(count)

    def observe_native(self, runner, output):
        """Record only successfully executed native prefill, including consecutive chunks."""
        scheduled = self._schedule(runner, output)
        if scheduled is None:
            self._release("outside single text request")
            return
        rid, start, prompt, count = scheduled
        end = start + count
        if start == 0 and self.request_id is not None:
            self._release("new native prefill")
        if start < prompt and end <= prompt and (
            start == 0 or (self.request_id, self.next_position) == (rid, start)
        ):
            self.request_id, self.next_position = rid, end
            _LOG.info("GLM53 VA native prefill request=%s positions=[%d,%d) prompt=%d", rid, start, end, prompt)
        else:
            self._release("native decode or discontinuous prefix")

    def _cache_descriptors(self, runner):
        from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend
        from vllm_ascend.worker.va_mk_bridge import IndexerCache, KDACache, MLACache

        registry = runner.compilation_config.static_forward_context
        kda = {}
        for index, name in enumerate(self.names[:3]):
            owner = registry[name]
            conv, recurrent = owner.kv_cache
            kda[index] = KDACache(name, conv, recurrent, owner._conv_state_dim_first)
        mla_name, pool_name, tail_name = self.names[3:]
        latent, rope = registry[mla_name].kv_cache
        if rope.shape[-1] != 0:
            raise ValueError("GLM53 MK requires a zero-width native MLA rope cache")
        cache_for = Glm5NextKPoolIndexerBackend._bound_cache
        indexer = IndexerCache(pool_name, tail_name, cache_for(registry[pool_name]), cache_for(registry[tail_name]))
        return kda, {3: MLACache(mla_name, latent)}, {3: indexer}

    def try_decode(self, runner, output, context, positions, input_ids):
        scheduled = self._schedule(runner, output)
        if scheduled is None:
            return None
        rid, position, prompt, count = scheduled
        if ((rid, position) != (self.request_id, self.next_position) or count != 1
                or position < prompt or not 0 < position < getattr(self.bridge, "max_context_tokens", 64)):
            return None
        if context.cudagraph_runtime_mode.name != "NONE" or getattr(context, "capturing", False):
            raise RuntimeError("GLM53 MK decode entered a non-eager VA ForwardContext")
        # For a text request through the multimodal wrapper the runner embeds this very
        # token during preprocessing and can then pass input_ids=None to the model.
        token = runner.input_ids.gpu[:1] if input_ids is None else input_ids
        if token.shape != (1,) or positions is None or positions.shape not in ((1,), (3, 1)):
            raise RuntimeError("GLM53 MK requires one unpadded token and its scheduler position")
        if not bool((positions == position).all().item()):
            raise RuntimeError("GLM53 MK scheduler and model positions differ")
        executor = getattr(context, "device_metadata_executor", None)
        try:
            if executor is not None and executor.submission_in_flight:
                torch.npu.current_stream().wait_stream(executor.stream)
            prepared = getattr(context, "glm53_cpu_addresses", None)
            addresses = None
            if (prepared is not None and prepared[0] == (rid, position)
                    and not getattr(context, "virtual_engine", 0)
                    and self._early_cache_bindings_match(runner, prepared[1])):
                _, (kda, mla, indexer), addresses = prepared
            else:
                materialize = getattr(context, "glm53_materialize_native_attention_metadata", None)
                if materialize is not None:
                    materialize()
                kda, mla, indexer = self._cache_descriptors(runner)
            cpu_reads = None
            if (addresses is None and os.getenv("MK_GLM53_CPU_METADATA_PREFETCH") == "1"
                    and getattr(self, "tp_size", None) == 8 and self.bridge._request == rid):
                from vllm_ascend.worker.glm53_mk_cpu_metadata import CpuDecodeMetadata, UnsupportedMetadata
                if not hasattr(self, "_cpu_decode_metadata"):
                    try:
                        self._cpu_decode_metadata = CpuDecodeMetadata(runner)
                    except (AttributeError, IndexError, KeyError, UnsupportedMetadata):
                        self._cpu_decode_metadata = None
                if self._cpu_decode_metadata is not None:
                    cpu_reads = self._cpu_decode_metadata.read(runner, context.attn_metadata, position, kda, mla, indexer)
                mode = cpu_reads is not None
                reported = getattr(self, "_cpu_metadata_reported", set())
                if mode not in reported:
                    from vllm.distributed import get_tensor_model_parallel_rank
                    _LOG.info("GLM53 CPU metadata prefetch rank=%d enabled=%s", get_tensor_model_parallel_rank(), mode)
                    self._cpu_metadata_reported = reported | {mode}
            shadow = addresses is not None and context.attn_metadata is not None
            result = self.bridge.step(context, request_id=rid, position=position, input_ids=token,
                                      kda=kda, mla=mla, indexer=indexer,
                                      cpu_addresses=addresses, cpu_reads=cpu_reads)
            if getattr(self, "_report_address_steps", False):
                _LOG.info("GLM53 MK metadata built rank=%d request=%s position=%d native_calls=%d",
                          self.gate.rank, rid, position, context.glm53_native_metadata_builds)
            if getattr(self, "_report_address_steps", False):
                address_source = (
                    "shadow-device" if shadow else "cpu-plan" if addresses is not None
                    else "legacy-cpu" if cpu_reads is not None else "native-device"
                )
                _LOG.info("GLM53 MK metadata address source rank=%d request=%s position=%d source=%s",
                          self.gate.rank, rid, position, address_source)
            phase = position - prompt
            if addresses is not None and (shadow or self._report_address_steps
                                          or phase in (1, 15, 62) or not self._address_plan_reported):
                _LOG.info("GLM53 native builder deferral rank=%d request=%s position=%d shadow=%s",
                          self.gate.rank, rid, position, shadow)
                self._address_plan_reported = True
        finally:
            if executor is not None and executor.submission_in_flight:
                executor.release()
        if result.hidden_states.shape != (1, self.text.config.hidden_size):
            raise RuntimeError("GLM53 MK returned an invalid hidden-state shape")
        if result.logits is not None and result.logits.shape != (1, self.text.config.vocab_size):
            raise RuntimeError("GLM53 MK returned an invalid logits shape")
        self.next_position = position + 1
        _LOG.info("GLM53 MK decode request=%s position=%d graph=%s imported=%s cache_writeback=complete",
                  rid, position, "reuse" if result.graph_reused else "capture", result.imported)
        return _Step(rid, position, result.hidden_states, result.logits)

    def process_logits(self, runner, output, step, logits_indices):
        scheduled = self._schedule(runner, output)
        if (scheduled is None or scheduled[:2] != (step.request_id, step.position)
                or (self.request_id, self.next_position) != (step.request_id, step.position + 1)
                or step.logits_consumed or logits_indices.shape != (1,) or int(logits_indices.item()) != 0):
            raise RuntimeError("GLM53 MK logits must be consumed once for their exact scheduled token")
        step.logits_consumed = True
        if step.raw_logits is None:
            return runner.model.compute_logits(step.hidden_states[logits_indices])
        # Native scaling may be in-place. Give VA its own logits storage for subsequent sampling.
        raw = step.raw_logits.clone()
        step.raw_logits = None
        return self.logits_processor(self.text.lm_head, raw)
