"""Single-request GLM-Next draft replay on the native unpadded, K=1 proposer."""

from copy import copy
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from vllm.logger import logger
from vllm_ascend import envs


def _require(ok, detail):
    if not ok:
        raise RuntimeError(f"GLM53 MK MTP: {detail}")


def _positions(tensor):
    rows = tensor.tolist()
    if tensor.ndim == 2:
        _require(len(rows) == 3 and rows[0] == rows[1] == rows[2], "text position axes disagree")
        rows = rows[0]
    _require(bool(rows) and rows == list(range(rows[0], rows[0] + len(rows))), "non-contiguous positions")
    return rows


def _page(metadata, cache):
    table = metadata.block_table
    _require(table.ndim == 2 and table.shape[0] == 1 and table.shape[1] > 0, "expected single page-table row")
    page = int(table[0, 0].item())
    _require(0 <= page < cache.shape[0], "physical page outside allocated pool")
    return page


@dataclass
class NativeCache:
    latent: object
    pool: object
    tail: object
    pages: tuple

    def signature(self):
        return (self.pages, tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.dtype, t.device)
                                  for t in (self.latent, self.pool, self.tail)))

    @classmethod
    def resolve(cls, native, metas, live):
        latent, pool, tail = native
        sfa, km, tm = metas
        n, start, end = len(live), live[0], live[-1] + 1
        _require(1 <= n <= 2 and 0 < start < end <= 64, "outside short K=1 draft")
        for cache, width in ((latent, 512), (pool, 128)):
            _require(cache.ndim == 4 and tuple(cache.shape[2:]) == (1, width)
                     and cache.dtype == torch.bfloat16 and cache.stride(-1) == 1, "invalid MLA or pooled cache")
        _require(tuple(tail.shape[1:]) == (5, 256) and tail.dtype == torch.float32
                 and tail.stride(1) == 256 and tail.stride(2) == 1 and tail.stride(0) % 256 == 0
                 and tail.stride(0) >= 5 * 256, "expected native speculative FP32 tail ring of five")
        _require(latent.device == pool.device == tail.device, "draft caches are on different devices")
        mp, kp, tp = (_page(metadata, cache) for metadata, cache in zip(metas, native))
        _require(sfa.block_size == latent.shape[1] and end <= latent.shape[1]
                 and sfa.num_actual_tokens == sfa.num_input_tokens == n
                 and sfa.cos is None and sfa.sin is None, "incompatible unpadded SFA metadata")
        _require(_positions(sfa.positions) == live and sfa.query_start_loc.tolist() == [0, n]
                 and sfa.seq_lens.tolist() == [end], "SFA positions/query/length disagree")
        mla_slots = [mp * latent.shape[1] + p for p in live]
        _require(sfa.slot_mapping[:n].tolist() == mla_slots, "SFA slot differs from physical page")
        _require(km.num_actual_tokens == n and km.compress_ratio == 4 and km.query_metadata is None
                 and _positions(km.positions) == live and km.query_start_loc.tolist() == [0, n]
                 and km.seq_lens.tolist() == [end // 4] and km.raw_seq_lens.tolist() == [end]
                 and km.start_pos.tolist() == [start] and km.pool_tail.tolist() == [end % 4]
                 and (end - 1) // 4 < pool.shape[1], "incompatible compressed KPool metadata")
        slots = [kp * pool.shape[1] + p // 4 if p % 4 == 3 else -1 for p in live]
        compact = [slot for slot in slots if slot >= 0] or [-1]
        _require(km.slot_mapping.tolist() == slots and km.pooled_slot_mapping.tolist() == compact,
                 "compressed addresses differ from physical KPool rows")
        ci = tm.cann_inputs
        page_rows = tail.stride(0) // 256
        _require(tm.block_size == 5 and ci.page_rows == page_rows and ci.retained_indices is not None,
                 "incompatible speculative tail addressing")
        base = tp * page_rows
        history = [base + (start - start % 4 + j) % 5 for j in range(4)]
        _require(ci.history_indices.tolist() == [history], "tail history differs from five-row ring")
        writes, retained = ci.write_indices.tolist(), ci.retained_indices.tolist()
        _require(len(writes) == len(retained) == 1 and n <= len(writes[0]) <= 5
                 and len(writes[0]) == len(retained[0]), "invalid retained tail rows")
        width = len(writes[0])
        indices = list(range(n - width, n))
        _require(retained[0] == [max(0, i) for i in indices]
                 and writes[0] == [-1 if i < 0 else base + live[i] % 5 for i in indices],
                 "retained tail writes differ from accepted target rows")
        return cls(latent, pool, tail, (mp, kp, tp))


def _make_batch(device):
    def zeros(shape, dtype):
        return torch.zeros(shape, dtype=dtype, device=device)

    return SimpleNamespace(
        mla_latent_cache={4: zeros((2, 64, 1, 512), torch.bfloat16)},
        index_k_buffer={4: zeros((2, 64, 1, 128), torch.bfloat16)},
        index_k_tail_buffer={4: zeros((1, 4, 128), torch.bfloat16)},
        index_k_tail_score_buffer={4: zeros((1, 4, 128), torch.float32)},
        block_tables=zeros((1, 1), torch.int32).fill_(1),
        kv_seq_len=zeros(1, torch.int32),
        mla_slot_indices_i64=zeros((1, 1), torch.int64),
        index_rope_positions=zeros(1, torch.int64),
        index_req_pool_indices=zeros(1, torch.int32),
        index_valid_mask=zeros(1, torch.int32).fill_(1),
        index_cos_sin_cache=None,
    )


class GLM53MKDraft:
    def __init__(self, graph, batch, names, native_layer, processor):
        self.graph = graph
        self.batch = batch
        self.names = names
        self.native_layer = native_layer
        self.processor = copy(processor)
        self.processor.logits_as_input = True
        self.request = None
        self.next_position = None
        self.cache_signature = None
        self.invalid = False

    @classmethod
    def load(cls, proposer, target):
        from blockrt.models.glm_5_3_flash.model_args import ModelArgsGLM5_3Flash
        from blockrt.models.glm_5_3_flash.modules.glm5_3_mtp import Glm53MTPDecoder
        from blockrt.models.glm_5_3_flash.modules.glm5_3_mtp_graph import Glm53MTPGraphRunner
        from blockrt.models.glm_5_3_flash.weight_loader import WeightLoader

        runner = proposer.runner
        spec, vcfg = proposer.speculative_config, proposer.vllm_config
        parallel = vcfg.parallel_config
        topology = (parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.data_parallel_size,
                    parallel.prefill_context_parallel_size, parallel.decode_context_parallel_size)
        _require(topology == (1, 1, 1, 1, 1) and runner.model_config.enforce_eager
                 and not proposer.use_cuda_graph and runner.scheduler_config.max_num_seqs == 1
                 and not runner.use_async_scheduling and vcfg.kv_transfer_config is None
                 and getattr(vcfg, "ec_transfer_config", None) is None
                 and spec.method == "mtp" and spec.num_speculative_tokens == 1
                 and spec.disable_padded_drafter_batch and not proposer._enable_probabilistic_draft_probs
                 and not proposer.parallel_drafting and not proposer.needs_extra_input_slots
                 and proposer.pass_hidden_states_to_model and not runner.use_aux_hidden_state_outputs
                 and not envs.VLLM_ASCEND_GLM53_MK_DECODE,
                 "requires eager TP1/EP1, a synchronous native target, one request, unpadded greedy MTP K=1")
        cfg = getattr(target, "language_model", target).config
        predictor = proposer.model.model
        native_layer = predictor.mtp_start_layer_idx
        _require(cfg.model_type == "glm5_next_text" and cfg.num_hidden_layers == 4
                 and cfg.qk_rope_head_dim == 0 and cfg.index_topk == 2048 and cfg.index_kpool == 4
                 and cfg.index_kpool_compress and cfg.index_kpool_always_select_tail
                 and native_layer in (4, 45) and predictor.num_mtp_layers == 1,
                 "requires four-layer GLM-Next and one NoPE/KPool MTP layer")
        attn = predictor._mtp_layers[0].mtp_block.self_attn
        owners = (attn.mla_attn.mla_attn, attn.indexer.k_cache, attn.indexer.tail_cache)
        registry = runner.compilation_config.static_forward_context
        keys = {id(owner): name for name, owner in registry.items()}
        names = tuple(keys[id(owner)] for owner in owners)
        _require(len(set(names)) == 3, "aliased draft attention-group names")
        config = ModelArgsGLM5_3Flash()
        config.num_hidden_layers, config.num_nextn_predict_layers = 4, 1
        config.routed_expert_format = "W4A4_MXFP4"
        config.indexer_skip_rope = True
        config.mxfp8_decode = True
        _require((cfg.vocab_size, cfg.hidden_size) == (config.vocab_size, config.hidden_size),
                 "draft vocabulary or hidden size differs from MK")
        path = envs.VLLM_ASCEND_GLM53_MK_CHECKPOINT or runner.model_config.model
        logger.info("GLM53 MK MTP loading checkpoint=%s checkpoint_layer=45 native_layer=%d cache_keys=%s",
                    path, native_layer, names)
        with torch.inference_mode(False), torch.no_grad():
            weights = WeightLoader(config, path).load_nextn(checkpoint_layer=45, vllm_ascend=True)
            decoder = Glm53MTPDecoder(config, device="meta").to_empty(device=runner.device)
            decoder.load_weights(weights)
            del weights
            device_id = runner.device.index if runner.device.index is not None else torch.npu.current_device()
            graph = Glm53MTPGraphRunner(decoder, device_id=device_id, max_graphs=1)
            batch = _make_batch(runner.device)
        logger.info("GLM53 MK MTP loaded native_layer=%d mk_layer=4 native_target_verify=true", native_layer)
        return cls(graph, batch, names, native_layer, predictor.logits_processor)

    def _release(self, reason):
        if self.request is not None:
            logger.info("GLM53 MK MTP release request=%s next_position=%s reason=%s",
                        self.request, self.next_position, reason)
        self.request, self.next_position, self.cache_signature = None, None, None

    def retire(self, output):
        if self.request is None or output is None:
            return
        rid = self.request
        if (rid in output.finished_req_ids or any(r.req_id == rid for r in output.scheduled_new_reqs)
                or rid in output.scheduled_cached_reqs.resumed_req_ids
                or rid in (getattr(output, "preempted_req_ids", None) or ())):
            self._release("new lifetime, retirement or resumption")

    def _native_caches(self, proposer):
        from vllm_ascend.attention.indexer_kpool import Glm5NextKPoolIndexerBackend

        registry = proposer.runner.compilation_config.static_forward_context
        mla, pool, tail = (registry[name] for name in self.names)
        latent, rope = mla.kv_cache
        _require(rope.shape[-1] == 0, "expected a zero-width native draft MLA RoPE cache")
        bound = Glm5NextKPoolIndexerBackend._bound_cache
        return latent, bound(pool), bound(tail)

    def _import(self, cache, start):
        mp, kp, tp = cache.pages
        b = self.batch
        latent, pool = b.mla_latent_cache[4][1], b.index_k_buffer[4][1]
        keys, gates = b.index_k_tail_buffer[4][0], b.index_k_tail_score_buffer[4][0]
        for t in (latent, pool, keys, gates):
            t.zero_()
        latent[:start].copy_(cache.latent[mp, :start])
        pool[:start // 4].copy_(cache.pool[kp, :start // 4])
        for pos in range(start - start % 4, start):
            keys[pos % 4].copy_(cache.tail[tp, pos % 5, :128])
            gates[pos % 4].copy_(cache.tail[tp, pos % 5, 128:])

    def _commit(self, cache, live):
        mp, kp, tp = cache.pages
        b = self.batch
        for pos in live:
            cache.latent[mp, pos].copy_(b.mla_latent_cache[4][1, pos])
            if pos % 4 == 3:
                cache.pool[kp, pos // 4].copy_(b.index_k_buffer[4][1, pos // 4])
            cache.tail[tp, pos % 5, :128].copy_(b.index_k_tail_buffer[4][0, pos % 4])
            cache.tail[tp, pos % 5, 128:].copy_(b.index_k_tail_score_buffer[4][0, pos % 4])

    def propose(self, proposer, output, inputs, native):
        _require(not self.invalid, "previous draft launch or native write failed")
        runner, batch = proposer.runner, proposer.runner.input_batch
        n = inputs["num_tokens"]
        if (inputs["batch_size"] != 1 or batch.num_reqs != 1 or inputs["num_input_tokens"] != n
                or proposer.num_speculative_tokens != 1 or n < 1 or len(output.num_scheduled_tokens) != 1):
            self._release("outside unpadded single-request draft")
            return native()
        rid = batch.req_ids[0]
        req = runner.requests[rid]
        if (rid not in output.num_scheduled_tokens or req.mm_features or req.prompt_embeds is not None
                or req.lora_request is not None or req.sampling_params is None or req.sampling_params.temperature != 0
                or inputs["token_indices_to_sample"].tolist() != [n - 1]):
            self._release("outside terminal greedy text proposal")
            return native()
        live = _positions(proposer._get_positions(n))
        _require(len(live) == n and _positions(inputs["target_positions"]) == live,
                 "draft positions differ from compacted target hidden rows")
        start, end, prompt = live[0], live[-1] + 1, int(batch.num_prompt_tokens[0])
        if start < prompt and end <= prompt and end <= 64:
            ids = native()
            if start == 0 or (rid, start) == (self.request, self.next_position):
                self.request, self.next_position, self.cache_signature = rid, end, None
                logger.info("GLM53 MTP native seed request=%s positions=[%d,%d) draft_token=%d",
                            rid, start, end, int(ids[0, 0].item()))
            else:
                self._release("unobserved native prefix")
            return ids
        scheduled_drafts = output.scheduled_spec_decode_tokens.get(rid, ())
        if ((rid, start) != (self.request, self.next_position) or end > 64 or start < prompt
                or len(scheduled_drafts) != 1 or n not in (1, 2)):
            self._release("outside continuous single-candidate verification")
            return native()
        _require(len(inputs["multi_steps_attn_metadata"]) == 1, "expected one native draft metadata step")
        metas = inputs["multi_steps_attn_metadata"][0]
        cache = NativeCache.resolve(self._native_caches(proposer), tuple(metas[name] for name in self.names), live)
        signature = cache.signature()
        imported = self.cache_signature is None
        _require(imported or signature == self.cache_signature, "native draft pages changed mid-request")
        hidden, tokens = proposer.hidden_states[:n], proposer.input_ids[:n]
        _require(tuple(hidden.shape) == (n, 4096) and hidden.dtype == torch.bfloat16
                 and tuple(tokens.shape) == (n,) and tokens.dtype in (torch.int32, torch.int64)
                 and hidden.device == tokens.device == cache.latent.device, "invalid compacted draft inputs")
        before = next(iter(self.graph._graphs.values()))[0] if self.graph._graphs else None
        try:
            with torch.inference_mode(False), torch.no_grad():
                if imported:
                    self._import(cache, start)
                for i, pos in enumerate(live):
                    b = self.batch
                    b.mla_slot_indices_i64.fill_(64 + pos)
                    b.kv_seq_len.fill_(pos + 1)
                    b.index_rope_positions.fill_(pos)
                    result = self.graph.run(b, tokens[i:i + 1].contiguous(), hidden[i:i + 1].contiguous())
                raw = result.logits.clone()
            logits = self.processor(None, raw)
            ids, probs = proposer._sample_draft_from_logits(logits, inputs["sampling_metadata"])
            _require(tuple(ids.shape) == (1,) and probs is None, "expected one native greedy draft token")
            with torch.inference_mode():
                self._commit(cache, live)
                torch.npu.synchronize(self.graph.device_id)
            after = next(iter(self.graph._graphs.values()))[0]
            _require(before is None or before is after, "draft graph address changed during replay")
        except Exception:
            self.invalid = True
            raise
        proposer._last_draft_probs = None
        self.cache_signature, self.next_position = signature, end
        logger.info("GLM53 MK MTP proposal request=%s positions=[%d,%d) graph=%s imported=%s "
                    "verified_candidate=%d accepted_drafts=%d accepted_length=%d draft_token=%d cache_commit=%d",
                    rid, start, end, "capture" if before is None else "reuse", imported,
                    scheduled_drafts[0], n - 1, n, int(ids[0].item()), n)
        return ids.view(1, 1)
