"""Hot KV cache helpers for sparse flash attention offload.

The online path in ``sfa_v1.py`` uses NPU tensors for slot lookup and data
movement. This module keeps CPU-only pieces here so they can be unit tested and
reused by the offline analysis script.
"""

from __future__ import annotations

import atexit
import os
import tempfile
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Iterable

import torch


@dataclass
class _SimReqState:
    resident: dict[int, int] = field(default_factory=dict)
    slot_owner: list[int] = field(default_factory=list)
    recent_queue: deque[tuple[int, ...]] = field(default_factory=deque)
    freq: defaultdict[int, int] = field(default_factory=lambda: defaultdict(int))
    ema: defaultdict[int, float] = field(default_factory=lambda: defaultdict(float))
    last_used: defaultdict[int, int] = field(default_factory=lambda: defaultdict(lambda: -1))
    step: int = 0
    cursor: int = 0


class HotKVCacheSimulator:
    """Offline simulator for the hot KV resident-cache policy.

    Records are dictionaries with ``layer``, ``req_ids`` and ``topk_indices``.
    ``topk_indices`` can be a torch tensor, nested list, or tuple with shape
    ``[num_reqs, topk]``.
    """

    def __init__(
        self,
        buffer_size: int,
        recent_window: int = 32,
        ema_beta: float = 0.9,
        recent_weight: float = 1.0,
        ema_weight: float = 0.5,
        age_weight: float = 0.01,
        candidate_size: int = 256,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        self.buffer_size = buffer_size
        self.recent_window = recent_window
        self.ema_beta = ema_beta
        self.recent_weight = recent_weight
        self.ema_weight = ema_weight
        self.age_weight = age_weight
        self.candidate_size = max(1, candidate_size)

    def run(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        states: dict[tuple[str, int], _SimReqState] = {}
        global_stats = {"requests": 0, "needed": 0, "hits": 0, "misses": 0}
        by_layer: defaultdict[str, dict[str, int]] = defaultdict(
            lambda: {"requests": 0, "needed": 0, "hits": 0, "misses": 0})

        for record in records:
            layer = str(record.get("layer", record.get("layer_name", "")))
            req_ids = [int(req_id) for req_id in record.get("req_ids", [])]
            rows = _to_nested_ints(record["topk_indices"])
            if not req_ids:
                req_ids = list(range(len(rows)))

            for row_index, tokens in enumerate(rows):
                req_id = req_ids[row_index] if row_index < len(req_ids) else row_index
                state = states.setdefault((layer, req_id), _SimReqState(
                    slot_owner=[-1 for _ in range(self.buffer_size)]))
                unique_tokens = [token for token in dict.fromkeys(tokens) if token >= 0]
                hits = sum(1 for token in unique_tokens if token in state.resident)
                misses = len(unique_tokens) - hits

                self._observe(state, unique_tokens)
                for token in unique_tokens:
                    if token not in state.resident:
                        slot = self._choose_slot(state, protected=set(unique_tokens))
                        old = state.slot_owner[slot]
                        if old >= 0:
                            state.resident.pop(old, None)
                        state.slot_owner[slot] = token
                        state.resident[token] = slot

                _accumulate(global_stats, len(unique_tokens), hits, misses)
                _accumulate(by_layer[layer], len(unique_tokens), hits, misses)

        _finish_stats(global_stats)
        by_layer_result = dict(by_layer)
        for stats in by_layer_result.values():
            _finish_stats(stats)
        return {"global": global_stats, "by_layer": by_layer_result}

    def _observe(self, state: _SimReqState, tokens: list[int]) -> None:
        state.step += 1
        token_set = set(tokens)
        for token in token_set:
            state.ema[token] = self.ema_beta * state.ema[token] + (1.0 - self.ema_beta)
            state.freq[token] += 1
            state.last_used[token] = state.step
        state.recent_queue.append(tuple(token_set))
        while len(state.recent_queue) > self.recent_window:
            old_tokens = state.recent_queue.popleft()
            for old_token in old_tokens:
                state.freq[old_token] -= 1

    def _choose_slot(self, state: _SimReqState, protected: set[int]) -> int:
        for slot, token in enumerate(state.slot_owner):
            if token < 0:
                return slot

        cap = len(state.slot_owner)
        best_slot = -1
        best_score = float("inf")
        scans = min(cap, self.candidate_size)
        for offset in range(scans):
            slot = (state.cursor + offset) % cap
            token = state.slot_owner[slot]
            if token in protected:
                continue
            score = self._score(state, token)
            if score < best_score:
                best_slot = slot
                best_score = score

        if best_slot < 0:
            for slot, token in enumerate(state.slot_owner):
                if token in protected:
                    continue
                score = self._score(state, token)
                if score < best_score:
                    best_slot = slot
                    best_score = score

        if best_slot < 0:
            best_slot = state.cursor
        state.cursor = (best_slot + 1) % cap
        return best_slot

    def _score(self, state: _SimReqState, token: int) -> float:
        last_used = state.last_used[token]
        age = 0 if last_used < 0 else state.step - last_used
        return (
            self.recent_weight * state.freq[token]
            + self.ema_weight * state.ema[token]
            - self.age_weight * age
        )


def update_hot_kv_cache_state(
    *,
    topk_indices: torch.Tensor,
    token_to_slot: torch.Tensor,
    slot_to_token: torch.Tensor,
    slot_freq: torch.Tensor,
    slot_ema: torch.Tensor,
    slot_last_used: torch.Tensor,
    evict_cursor: torch.Tensor,
    last_req_ids: torch.Tensor,
    req_ids: torch.Tensor,
    step_value: int,
    capacity: int,
    max_model_len: int,
    recent_window: int,
    recent_tokens: torch.Tensor,
    recent_weight: float,
    ema_weight: float,
    age_weight: float,
    ema_beta: float,
    candidate_size: int,
) -> dict[str, torch.Tensor]:
    """Update hot KV resident-cache state for one decode step.

    The state tensors are mutated in place. Duplicate topk tokens only observe
    the first occurrence; later duplicate positions reuse that slot.
    """
    num_reqs = topk_indices.shape[0]
    topk_width = topk_indices.shape[1]
    device = topk_indices.device

    valid_topk = (topk_indices >= 0) & (topk_indices < max_model_len)
    safe_topk = torch.clamp(topk_indices, min=0, max=max_model_len - 1)

    req_changed_mask = last_req_ids != req_ids
    if bool(req_changed_mask.any().item()):
        token_to_slot[req_changed_mask].fill_(-1)
        slot_to_token[req_changed_mask].fill_(-1)
        slot_freq[req_changed_mask].zero_()
        slot_ema[req_changed_mask].zero_()
        slot_last_used[req_changed_mask].fill_(-1)
        recent_tokens[req_changed_mask].fill_(-1)
        evict_cursor[...] = torch.where(
            req_changed_mask,
            torch.zeros_like(evict_cursor),
            evict_cursor,
        )

    positions = torch.arange(topk_width, dtype=torch.int64, device=device)
    same_token = safe_topk.unsqueeze(2) == safe_topk.unsqueeze(1)
    earlier_position = positions.view(1, 1, -1) < positions.view(1, -1, 1)
    previous_valid_same = same_token & earlier_position & valid_topk.unsqueeze(1)
    first_unique_mask = valid_topk & ~previous_valid_same.any(dim=2)

    first_position_candidates = torch.where(
        same_token & valid_topk.unsqueeze(1),
        positions.view(1, 1, -1),
        torch.full((num_reqs, topk_width, topk_width),
                   topk_width,
                   dtype=torch.int64,
                   device=device),
    )
    first_positions = first_position_candidates.min(dim=2).values
    first_positions = torch.clamp(first_positions, max=max(topk_width - 1, 0))

    ring_index = (step_value - 1) % recent_window
    old_recent_tokens = recent_tokens[:, ring_index, :topk_width]
    old_valid = (old_recent_tokens >= 0) & (old_recent_tokens < max_model_len)
    old_safe_tokens = torch.clamp(old_recent_tokens, min=0, max=max_model_len - 1)
    old_slots = torch.gather(token_to_slot, 1, old_safe_tokens.to(torch.int64))
    old_mapped = old_valid & (old_slots >= 0)
    for req_idx in range(num_reqs):
        req_old_slots = old_slots[req_idx][old_mapped[req_idx]].to(torch.int64)
        if req_old_slots.numel() > 0:
            slot_freq[req_idx].scatter_add_(
                0,
                req_old_slots,
                torch.full_like(req_old_slots, -1, dtype=torch.float32),
            )
    slot_freq.clamp_(min=0.0)

    slot_indices = torch.gather(token_to_slot, 1, safe_topk.to(torch.int64))
    hit_mask = first_unique_mask & (slot_indices >= 0)
    miss_mask = first_unique_mask & ~hit_mask

    protected_slot_count = torch.zeros(
        [num_reqs, capacity],
        dtype=torch.int32,
        device=device,
    )
    protected_slot_count.scatter_add_(
        1,
        torch.clamp(slot_indices, min=0).to(torch.int64),
        hit_mask.to(torch.int32),
    )
    protected_slot_mask = protected_slot_count > 0

    last_used = torch.where(
        slot_last_used >= 0,
        slot_last_used,
        torch.zeros_like(slot_last_used),
    )
    age = (step_value - last_used).to(torch.float32)
    evict_score = (
        age_weight * age
        - recent_weight * slot_freq
        - ema_weight * slot_ema
    )
    free_mask = slot_to_token < 0
    evict_score = torch.where(
        free_mask,
        torch.full_like(evict_score, 1.0e9),
        evict_score,
    )
    evict_score = torch.where(
        protected_slot_mask,
        torch.full_like(evict_score, -1.0e9),
        evict_score,
    )

    effective_candidate_size = min(capacity, max(candidate_size, topk_width * 2))
    candidate_offsets = torch.arange(
        effective_candidate_size,
        dtype=torch.int64,
        device=device,
    ).view(1, -1)
    candidate_slots = (evict_cursor.view(-1, 1) + candidate_offsets) % capacity
    candidate_scores = torch.gather(evict_score, 1, candidate_slots)
    target_rank_count = min(topk_width, effective_candidate_size)
    target_ranks = torch.topk(candidate_scores, k=target_rank_count, dim=1).indices
    target_slots_by_rank = torch.gather(candidate_slots, 1, target_ranks).to(torch.int32)
    evict_cursor[:] = (evict_cursor + effective_candidate_size) % capacity

    miss_rank = torch.clamp(torch.cumsum(miss_mask.to(torch.int32), dim=1) - 1, min=0)
    target_slots = torch.gather(target_slots_by_rank, 1, miss_rank.to(torch.int64))
    unique_current_slots = torch.where(hit_mask, slot_indices, target_slots)
    unique_current_slots = torch.where(
        first_unique_mask,
        unique_current_slots,
        torch.full_like(unique_current_slots, -1),
    )

    current_slots = torch.gather(unique_current_slots, 1, first_positions)
    current_slots = torch.where(
        valid_topk,
        current_slots,
        torch.full_like(current_slots, -1),
    )

    current_unique_tokens = torch.where(
        first_unique_mask,
        topk_indices,
        torch.full_like(topk_indices, -1),
    )
    recent_tokens[:, ring_index, :topk_width] = current_unique_tokens
    if recent_tokens.shape[2] > topk_width:
        recent_tokens[:, ring_index, topk_width:].fill_(-1)

    load_token_indices = torch.full(
        [num_reqs, capacity],
        -1,
        dtype=torch.int64,
        device=device,
    )
    for req_idx in range(num_reqs):
        req_miss = miss_mask[req_idx]
        req_slots = target_slots[req_idx][req_miss].to(torch.int64)
        req_tokens = topk_indices[req_idx][req_miss].to(torch.int64)
        if req_slots.numel() > 0:
            old_tokens = slot_to_token[req_idx][req_slots]
            old_valid = old_tokens >= 0
            token_to_slot[req_idx].scatter_(
                0,
                old_tokens[old_valid],
                torch.full_like(old_tokens[old_valid], -1, dtype=torch.int32),
            )
            token_to_slot[req_idx].scatter_(0, req_tokens, req_slots.to(torch.int32))
            slot_to_token[req_idx].scatter_(0, req_slots, req_tokens)
            load_token_indices[req_idx].scatter_(0, req_slots, req_tokens)

        req_current_slots = unique_current_slots[req_idx][first_unique_mask[req_idx]]
        req_current_slots = req_current_slots.to(torch.int64)
        if req_current_slots.numel() > 0:
            slot_last_used[req_idx].scatter_(
                0,
                req_current_slots,
                torch.full_like(req_current_slots, step_value, dtype=torch.int32),
            )
            slot_freq[req_idx].scatter_add_(
                0,
                req_current_slots,
                torch.ones_like(req_current_slots, dtype=torch.float32),
            )
            old_ema = torch.gather(slot_ema[req_idx], 0, req_current_slots)
            new_ema = ema_beta * old_ema + (1.0 - ema_beta)
            slot_ema[req_idx].scatter_(0, req_current_slots, new_ema)

    last_req_ids[...] = req_ids
    return {
        "current_slots": current_slots,
        "load_token_indices": load_token_indices,
        "hit_mask": hit_mask,
        "miss_mask": miss_mask,
        "valid_topk": valid_topk,
    }


class HotKVTopKDump:
    """Periodic torch.save writer for per-layer decode topk tensors."""

    def __init__(self, path: str, flush_interval: int = 198) -> None:
        self.path = path
        self.flush_interval = max(1, int(flush_interval))
        self.records: list[dict[str, Any]] = []
        self._closed = False
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        atexit.register(self.flush)

    def append(self, record: dict[str, Any]) -> None:
        self.records.append(record)
        if len(self.records) % self.flush_interval == 0:
            self.flush()

    def flush(self) -> None:
        if self._closed or not self.records:
            return
        import torch

        payload = {"format": "hot_kv_topk_v1", "records": self.records}
        directory = os.path.dirname(os.path.abspath(self.path))
        fd, tmp_path = tempfile.mkstemp(prefix=".hot_kv_dump_", suffix=".pt", dir=directory)
        os.close(fd)
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, self.path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def close(self) -> None:
        self.flush()
        self._closed = True


_DUMP_WRITERS: dict[str, HotKVTopKDump] = {}


def get_hot_kv_topk_dump(path: str, flush_interval: int = 198) -> HotKVTopKDump:
    writer = _DUMP_WRITERS.get(path)
    if writer is None:
        writer = HotKVTopKDump(path, flush_interval)
        _DUMP_WRITERS[path] = writer
    return writer


def load_topk_records(path: str) -> list[dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "records" in payload:
        return list(payload["records"])
    if isinstance(payload, list):
        return payload
    raise ValueError(f"unsupported hot KV topk dump format: {type(payload)!r}")


def _to_nested_ints(value: Any) -> list[list[int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    return [[int(item) for item in row] for row in value]


def _accumulate(stats: dict[str, int], needed: int, hits: int, misses: int) -> None:
    stats["requests"] += 1
    stats["needed"] += needed
    stats["hits"] += hits
    stats["misses"] += misses


def _finish_stats(stats: dict[str, Any]) -> None:
    needed = stats["needed"]
    stats["hit_rate"] = 0.0 if needed == 0 else stats["hits"] / needed
