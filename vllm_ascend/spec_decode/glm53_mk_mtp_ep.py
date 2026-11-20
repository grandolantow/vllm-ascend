"""Single-origin packed layer-45 MTP replay on VA's eager TP1 DP=EP workers."""

import builtins
import json
from copy import copy
from dataclasses import dataclass

import torch
from vllm.logger import logger

from vllm_ascend import envs
from vllm_ascend.spec_decode.glm53_mk_mtp import GLM53MKDraft, NativeCache, _make_batch, _positions, _require


DEPTHS = {2: 8, 6: 45, 8: 45}


class DraftGate:
    def __init__(self, rank, size, exchange):
        self.rank, self.size, self.exchange = rank, size, exchange
        self.sequence = 0

    def gather(self, phase, payload):
        rows = self.exchange((self.rank, self.sequence, phase, payload))
        _require([(r, s, p) for r, s, p, _ in rows] ==
                 [(i, self.sequence, phase) for i in range(self.size)], f"DP vote order mismatch at {phase}")
        return [v for _, _, _, v in rows]

    def check(self, phase, error=None):
        errors = self.gather(phase, error)
        _require(not any(errors), f"round {self.sequence} {phase} failed: {errors}")

    def select(self, role, request, positions, error):
        self.sequence += 1
        rows = self.gather("select", (role, request, positions, error))
        _require(not any(row[3] for row in rows), f"draft eligibility failed: {rows}")
        real = [i for i, row in enumerate(rows) if row[0] == "real"]
        if len(real) == 1 and all(row[0] in ("real", "idle") for row in rows):
            owner = real[0]
            return owner, rows[owner][1], rows[owner][2]
        return None


def _detail(exc):
    return f"{type(exc).__name__}: {exc}"


def _capture_companion(moe, device, device_id):
    from blockrt.models.glm_5_3_flash import mxfp4
    from blockrt.runtime.tracer import DAGCapture

    # This input contributes zero rows. Dispatch returns the remote rows to
    # this rank's real packed weights, and combine sends computed results back.
    expanded = torch.zeros((8, moe.hidden), dtype=torch.bfloat16, device=device)
    counts = torch.zeros(moe.experts, dtype=torch.int64, device=device)
    names = ("moe_metadata_exchange", "moe_dispatch", "moe_combine")
    with DAGCapture() as capture:
        returned = mxfp4.grouped(moe, expanded, counts)
        comm = [node for node in capture.toposort() if node.op_name in names]
        _require(tuple(node.op_name for node in comm) == names, "unexpected companion collective order")
        received = comm[1].outputs[2]
        _require(tuple(received.shape) == (moe.expert_partition.count,) and received.dtype == torch.int64,
                 "invalid local cumulative expert counts")
        capture.retain_output(returned)
        capture.retain_output(received)
    graph = capture.build(device_id=device_id, num_experts=0, skip_steps=0,
                          callback="libdefault_callback.so", enable_static_fusion=False)
    return graph, received, capture, expanded, counts


def _cache_for_rows(native, metas, live, padded):
    n = len(live)
    sfa, pool, tail = metas
    _require(sfa.num_actual_tokens == n and sfa.num_input_tokens == padded and n <= padded <= 2,
             "SFA compact/padded counts disagree")
    if padded > n:
        # The pinned builder can slice SFA positions from an already-compact
        # source, while slots are padded. KPool materializes both at DP width.
        for name, slots in (("SFA", sfa.slot_mapping), ("KPool", pool.slot_mapping), ("tail", tail.slot_mapping)):
            _require(slots.shape[0] == padded and slots[n:].tolist() == [-1] * (padded - n),
                     f"{name} padded slots must be disabled")
        sfa, pool = copy(sfa), copy(pool)
        sfa.positions = sfa.positions[:n]
        sfa.num_input_tokens = n
        pool.positions, pool.slot_mapping = pool.positions[:n], pool.slot_mapping[:n]
    return NativeCache.resolve(native, (sfa, pool, tail), live)


@dataclass
class Proposal:
    role: str
    request: object = None
    positions: tuple = ()
    cache: object = None
    candidate: object = None


class PackedEPDraft(GLM53MKDraft):
    def __init__(self, graph, batch, names, layer, processor, gate, runtime, companion):
        super().__init__(graph, batch, names, layer, processor)
        self.gate, self.runtime, self.companion = gate, runtime, companion

    def _plan(self, proposer, output, inputs, runtime_dummy):
        _require(not self.invalid, "previous packed draft round failed")
        if output is None:
            return Proposal("idle" if runtime_dummy else "native")
        batch = proposer.runner.input_batch
        n, padded = inputs["num_tokens"], inputs["num_input_tokens"]
        if (inputs["batch_size"] != 1 or batch.num_reqs != 1 or not 1 <= n <= padded
                or proposer.num_speculative_tokens != 1 or len(output.num_scheduled_tokens) != 1):
            return Proposal("native")
        rid = batch.req_ids[0]
        req = proposer.runner.requests[rid]
        if (rid not in output.num_scheduled_tokens or req.mm_features or req.prompt_embeds is not None
                or req.lora_request is not None or req.sampling_params is None or req.sampling_params.temperature != 0
                or inputs["token_indices_to_sample"].tolist() != [n - 1]):
            return Proposal("native")
        live = _positions(proposer._get_positions(n))
        _require(len(live) == n and live == _positions(inputs["target_positions"]), "inconsistent compacted rows")
        start, end, prompt = live[0], live[-1] + 1, int(batch.num_prompt_tokens[0])
        if start < prompt and end <= prompt and end <= 64:
            return Proposal("seed", rid, tuple(live))
        candidates = output.scheduled_spec_decode_tokens.get(rid, ())
        if ((rid, start) != (self.request, self.next_position) or start < prompt or end > 64
                or len(candidates) != 1 or n not in (1, 2) or padded > 2):
            return Proposal("native")
        _require(len(inputs["multi_steps_attn_metadata"]) == 1, "expected a single draft metadata step")
        metas = inputs["multi_steps_attn_metadata"][0]
        cache = _cache_for_rows(self._native_caches(proposer), tuple(metas[name] for name in self.names), live, padded)
        _require(self.cache_signature is None or self.cache_signature == cache.signature(), "native draft pages changed")
        hidden, tokens = proposer.hidden_states[:n], proposer.input_ids[:n]
        _require(tuple(hidden.shape) == (n, 4096) and hidden.dtype == torch.bfloat16 and tuple(tokens.shape) == (n,)
                 and tokens.dtype in (torch.int32, torch.int64) and hidden.device == tokens.device == cache.latent.device,
                 "invalid compacted target hidden states or draft token IDs")
        return Proposal("real", rid, tuple(live), cache, int(candidates[0]))

    def _native_round(self, plan, native):
        ids = native()
        if plan.role == "seed" and (plan.positions[0] == 0 or
                                    (plan.request, plan.positions[0]) == (self.request, self.next_position)):
            self.request, self.next_position, self.cache_signature = plan.request, plan.positions[-1] + 1, None
            logger.info("GLM53 MTP EP native seed rank=%d request=%s positions=[%d,%d) draft_token=%d",
                        self.gate.rank, self.request, plan.positions[0], self.next_position, int(ids[0, 0].item()))
        else:
            self._release("collective native draft or idle worker")
        return ids

    def _row(self, proposer, plan, owner, index, pos):
        phase = f"row{index}"
        voted = False
        agreed = False

        def before_launch():
            nonlocal voted, agreed
            _require(getattr(builtins, "_mk_shmem_ptr", None) is self.runtime.stage,
                     "SHMEM symmetric buffer changed after capture")
            torch.npu.synchronize(self.graph.device_id)
            voted = True
            self.gate.check(phase + ".ready")
            agreed = True
            self.runtime.barrier()

        result, route, receive = None, None, None
        try:
            if owner == self.gate.rank:
                if index == 0 and self.cache_signature is None:
                    self._import(plan.cache, pos)
                self.batch.mla_slot_indices_i64.fill_(64 + pos)
                self.batch.kv_seq_len.fill_(pos + 1)
                self.batch.index_rope_positions.fill_(pos)
                result = self.graph.run(self.batch, proposer.input_ids[index:index + 1].contiguous(),
                                        proposer.hidden_states[index:index + 1].contiguous(), before_launch=before_launch)
                route = self.graph.last_debug["moe.expert_ids"][0].tolist()
                _require(len(route) == len(set(route)) == 8 and all(0 <= i < 288 for i in route), "invalid draft top-8")
            else:
                before_launch()
                self.companion[0].run()
                torch.npu.synchronize(self.graph.device_id)
                receive = int(self.companion[1][-1].item())
                _require(0 <= receive <= 8, "invalid companion receive count")
        except Exception as exc:
            if not voted:
                self.gate.check(phase + ".ready", _detail(exc))
            elif agreed:
                self.gate.gather(phase + ".complete", (_detail(exc), None, None))
            raise

        reports = self.gate.gather(phase + ".complete", (None, route, receive))
        _require(not any(report[0] for report in reports), f"draft row {index} failed: {reports}")
        routed = reports[owner][1]
        width = 288 // self.gate.size
        counts = [sum(r * width <= expert < (r + 1) * width for expert in routed) for r in range(self.gate.size)]
        _require(all(r == owner or reports[r][2] == counts[r] for r in range(self.gate.size)),
                 f"remote expert receipt differs from layer-45 routing: {reports}")
        return result, routed, counts

    def respond(self, proposer, output, inputs, native, runtime_dummy=False):
        plan, error = Proposal("native"), None
        try:
            if output is not None:
                self.retire(output)
            plan = self._plan(proposer, output, inputs, runtime_dummy)
        except Exception as exc:
            error = _detail(exc)
        decision = self.gate.select(plan.role, plan.request, plan.positions, error)
        if decision is None:
            logger.info("GLM53 MTP EP native round=%d rank=%d local=%s request=%s num_tokens=%s padded_tokens=%s",
                        self.gate.sequence, self.gate.rank, plan.role, plan.request,
                        inputs.get("num_tokens"), inputs.get("num_input_tokens"))
            return self._native_round(plan, native)
        owner, rid, live = decision
        real = self.gate.rank == owner
        if not real:
            self._release("idle companion")
        receipts, ids, result, commit_error = [], None, None, None
        try:
            with torch.inference_mode(False), torch.no_grad():
                for index, pos in enumerate(live):
                    result, route, counts = self._row(proposer, plan, owner, index, pos)
                    receipts.append((pos, route, counts))
                if real:
                    logits = self.processor(None, result.logits.clone())
                    ids, probs = proposer._sample_draft_from_logits(logits, inputs["sampling_metadata"])
                    _require(tuple(ids.shape) == (1,) and probs is None, "expected one native greedy sample")
                    with torch.inference_mode():
                        self._commit(plan.cache, live)
                        torch.npu.synchronize(self.graph.device_id)
        except Exception as exc:
            self.invalid = True
            if len(receipts) < len(live):
                raise
            commit_error = _detail(exc)

        # Publish a receipt only after all SHMEM rows, the sample, and the
        # native MLA, compressed-K and speculative tail writes completed.
        candidate, proposed = None, None
        if real and commit_error is None:
            try:
                candidate, proposed = plan.candidate, int(ids.item())
            except Exception as exc:
                commit_error = _detail(exc)
        outcomes = self.gate.gather("commit", (commit_error, candidate, proposed))
        if any(o[0] for o in outcomes):
            self.invalid = True
            raise RuntimeError(f"GLM53 packed draft cache/sample commit failed: {outcomes}")
        if real:
            self.cache_signature, self.next_position = plan.cache.signature(), live[-1] + 1
        proposer._last_draft_probs = None
        for pos, route, counts in receipts:
            entry = dict(round=self.gate.sequence, checkpoint_layer=45, request=rid, position=pos, rank=self.gate.rank,
                         owner=owner, ep_size=self.gate.size, role="real" if real else "companion", origin_tokens=int(real),
                         expert_ids=route, routed_rows=counts, received_rows=None if real else counts[self.gate.rank],
                         cache_commit=len(live) if real else 0, accepted_drafts=len(live) - 1,
                         verified_candidate=outcomes[owner][1], draft_token=outcomes[owner][2], native_target_verify=True)
            logger.info("GLM53_MTP_EP_ROW %s", json.dumps(entry, separators=(",", ":")))
        return ids.view(1, 1) if real else None  # native dummy_run discards the runnable result

    def close(self):
        self.runtime.close()


def _prepare(proposer, target, gate, path):
    from blockrt.models.glm_5_3_flash.model_args import ModelArgsGLM5_3Flash
    from blockrt.models.glm_5_3_flash.modules.glm5_3_mtp import Glm53MTPDecoder
    from blockrt.models.glm_5_3_flash.modules.glm5_3_mtp_graph import Glm53MTPGraphRunner
    from blockrt.models.glm_5_3_flash.weight_loader import WeightLoader

    runner, spec, cfg = proposer.runner, proposer.speculative_config, proposer.vllm_config
    parallel = cfg.parallel_config
    _require((parallel.tensor_parallel_size, parallel.pipeline_parallel_size, parallel.prefill_context_parallel_size,
              parallel.decode_context_parallel_size) == (1, 1, 1, 1) and parallel.enable_expert_parallel
             and parallel.expert_placement_strategy == "linear" and not parallel.enable_eplb,
             "requires static contiguous TP1/PP1/PCP1/DCP1 experts with DP=EP")
    fine = runner.ascend_config.finegrained_tp_config
    _require(all(s <= 1 for s in (fine.embedding_tensor_parallel_size, fine.lmhead_tensor_parallel_size,
                                  fine.oproj_tensor_parallel_size, fine.mlp_tensor_parallel_size)),
             "requires local draft embedding, head and dense projections")
    _require(runner.model_config.enforce_eager and not proposer.use_cuda_graph and not runner.use_async_scheduling
             and runner.scheduler_config.max_num_seqs == 1 and cfg.kv_transfer_config is None
             and getattr(cfg, "ec_transfer_config", None) is None and not runner.use_aux_hidden_state_outputs
             and spec.method == "mtp" and spec.num_speculative_tokens == 1 and spec.disable_padded_drafter_batch
             and not proposer._enable_probabilistic_draft_probs and getattr(proposer, "dynamic_spec", None) is None
             and proposer.pass_hidden_states_to_model and not proposer.parallel_drafting and not proposer.needs_extra_input_slots,
             "requires synchronous eager B1 text MTP K=1 with compacted target verify rows")
    native = getattr(target, "language_model", target).config
    predictor = proposer.model.model
    layer = predictor.mtp_start_layer_idx
    depth, size, rank = DEPTHS[gate.size], gate.size, gate.rank
    _require(native.model_type == "glm5_next_text" and native.num_hidden_layers == depth
             and layer in (depth, 45) and predictor.num_mtp_layers == 1 and native.qk_rope_head_dim == 0
             and native.index_topk == 2048 and native.index_kpool == 4 and native.index_kpool_compress
             and native.index_kpool_always_select_tail and native.n_routed_experts == 288,
             f"requires {depth}-layer GLM5-Next and one real NoPE/KPool MTP layer")
    block = predictor._mtp_layers[0].mtp_block
    moe = block.mlp
    width = 288 // size
    _require((moe.ep_rank, moe.ep_size, moe.n_logical_experts, moe.n_redundant_experts,
              moe.n_local_physical_experts, moe.physical_expert_start, moe.physical_expert_end) ==
             (rank, size, 288, 0, width, rank * width, (rank + 1) * width), "VA draft expert ownership mismatch")
    attn = block.self_attn
    registry = {id(owner): name for name, owner in runner.compilation_config.static_forward_context.items()}
    names = tuple(registry[id(owner)] for owner in (attn.mla_attn.mla_attn, attn.indexer.k_cache, attn.indexer.tail_cache))
    _require(len(set(names)) == 3, "overlapping native draft cache owners")
    args = ModelArgsGLM5_3Flash()
    # One standalone draft decoder. Its canonical scratch key is 4 so the
    # verified EP1 import/commit implementation can address these same buffers.
    # load_nextn aliases only keys, and explicitly reads source layer 45.
    args.num_hidden_layers, args.num_nextn_predict_layers = 4, 1
    args.ep_rank, args.ep_size = rank, size
    args.routed_expert_format, args.mxfp8_decode, args.indexer_skip_rope = "W4A4_MXFP4", True, True
    _require((native.vocab_size, native.hidden_size) == (args.vocab_size, args.hidden_size), "incompatible vocabulary")
    weights = WeightLoader(args, path).load_nextn(checkpoint_layer=45, vllm_ascend=True, packed_mxfp4=True)
    _require(weights.source_prefix.endswith(".layers.45") and "model.layers.4.shared_head.head.weight" in weights
             and "rot.weight" in weights, "real layer-45 head and VA input rotation are required")
    decoder = Glm53MTPDecoder(args, packed_mxfp4=True).to_empty(device=runner.device)
    decoder.load_weights(weights)
    del weights
    torch.npu.synchronize()
    device_id = torch.npu.current_device()
    _require(runner.device.index in (None, device_id), "VA and MK draft devices differ")
    graph = Glm53MTPGraphRunner(decoder, device_id=device_id, max_graphs=1, retain_debug=True)
    return graph, _make_batch(runner.device), names, layer, predictor.logits_processor


def load_ep_draft(proposer, target):
    from vllm.distributed.parallel_state import get_dp_group, get_ep_group

    enabled = envs.VLLM_ASCEND_GLM53_MK_MTP_EP
    size = proposer.vllm_config.parallel_config.data_parallel_size
    if size not in DEPTHS or not proposer.vllm_config.parallel_config.enable_expert_parallel:
        _require(not enabled, "packed draft requires DP=EP2/6/8")
        return None
    dp, ep = get_dp_group(), get_ep_group()

    def exchange(item):
        rows = [None] * size
        torch.distributed.all_gather_object(rows, item, group=dp.cpu_group)
        return rows

    gate = DraftGate(dp.rank_in_group, size, exchange)
    endpoint = envs.VLLM_ASCEND_GLM53_MK_EP_ENDPOINT
    path = envs.VLLM_ASCEND_GLM53_MK_CHECKPOINT or proposer.runner.model_config.model
    other = (envs.VLLM_ASCEND_GLM53_MK_EP
             or envs.VLLM_ASCEND_GLM53_MK_DECODE
             or envs.VLLM_ASCEND_GLM53_MK_MTP)
    config = gate.gather("config", (enabled, endpoint if enabled else None, path if enabled else None, other))
    _require(all(item == config[0] for item in config), "DP workers disagree on packed draft configuration")
    if not enabled:
        return None
    error, prepared = None, None
    try:
        from blockrt.dist.ep_runtime import ExpertParallelRuntime
        _require(endpoint and not other and ep.world_size == dp.world_size == size
                 and ep.rank_in_group == gate.rank == proposer.runner.dp_rank and tuple(ep.ranks) == tuple(dp.ranks)
                 and getattr(builtins, "_mk_shmem_ptr", None) is None,
                 "requires matching EP/DP groups, dedicated SHMEM endpoint and exclusive MK draft mode")
        with torch.inference_mode(False), torch.no_grad():
            prepared = _prepare(proposer, target, gate, path)
    except Exception as exc:
        error = _detail(exc)
    gate.check("weights", error)
    # Finish relatively slow weight reads before entering the SHMEM rendezvous.
    runtime, error = None, None
    try:
        runtime = ExpertParallelRuntime(gate.rank, size, endpoint)
    except Exception as exc:
        error = _detail(exc)
    gate.check("bootstrap", error)
    companion, error = None, None
    try:
        graph = prepared[0]
        with torch.inference_mode(False), torch.no_grad():
            companion = _capture_companion(graph.model.moe, proposer.runner.device, graph.device_id)
    except Exception as exc:
        error = _detail(exc)
    try:
        gate.check("capture_companion", error)
    except Exception:
        runtime.close()
        raise
    logger.info("GLM53 MTP EP loaded rank=%d/%d target_layers=%d checkpoint_layer=45 experts=[%d,%d) native_target_verify=true",
                gate.rank, size, DEPTHS[size], gate.rank * (288 // size), (gate.rank + 1) * (288 // size))
    return PackedEPDraft(*prepared, gate, runtime, companion)
