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


@dataclass
class _LRUSimReqState:
    resident: dict[int, int] = field(default_factory=dict)
    slot_owner: list[int] = field(default_factory=list)
    last_used: defaultdict[int, int] = field(default_factory=lambda: defaultdict(lambda: -1))
    step: int = 0


def resolve_hot_kv_candidate_size(
    *,
    candidate_size: int,
    topk_width: int,
    capacity: int,
) -> int:
    """Return the online effective candidate window size."""
    if capacity < 1:
        raise ValueError("capacity must be >= 1")
    safe_candidate_size = max(1, int(candidate_size))
    safe_topk_width = max(0, int(topk_width))
    return min(int(capacity), max(safe_candidate_size, safe_topk_width * 2))


def _normalize_sim_tokens(tokens: list[int], topk_length: int | None) -> list[int]:
    if topk_length is not None:
        tokens = tokens[:topk_length]
    return [token for token in dict.fromkeys(tokens) if token >= 0]


def _sim_topk_width(tokens: list[int], topk_length: int | None) -> int:
    if topk_length is None:
        return len(tokens)
    return min(len(tokens), topk_length)


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
        topk_length: int | None = None,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if recent_window < 1:
            raise ValueError("recent_window must be >= 1")
        if topk_length is not None and topk_length < 1:
            raise ValueError("topk_length must be >= 1")
        if topk_length is not None and topk_length > buffer_size:
            raise ValueError("topk_length must be <= buffer_size")
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        self.buffer_size = buffer_size
        self.recent_window = recent_window
        self.ema_beta = ema_beta
        self.recent_weight = recent_weight
        self.ema_weight = ema_weight
        self.age_weight = age_weight
        self.candidate_size = max(1, candidate_size)
        self.topk_length = topk_length

    def _new_state_for_test(self) -> _SimReqState:
        """Create an initialized simulator state for focused policy tests."""
        return _SimReqState(slot_owner=[-1 for _ in range(self.buffer_size)])

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
                needed, hits, misses = self._process_row(state, tokens)
                _accumulate(global_stats, needed, hits, misses)
                _accumulate(by_layer[layer], needed, hits, misses)

        _finish_stats(global_stats)
        by_layer_result = dict(by_layer)
        for stats in by_layer_result.values():
            _finish_stats(stats)
        return {"global": global_stats, "by_layer": by_layer_result}

    def _process_row(
        self,
        state: _SimReqState,
        tokens: list[int],
    ) -> tuple[int, int, int]:
        state.step += 1
        unique_tokens = _normalize_sim_tokens(tokens, self.topk_length)
        topk_width = _sim_topk_width(tokens, self.topk_length)
        effective_candidate_size = resolve_hot_kv_candidate_size(
            candidate_size=self.candidate_size,
            topk_width=topk_width,
            capacity=self.buffer_size,
        )

        self._expire_recent_window(state)

        hit_tokens = [token for token in unique_tokens if token in state.resident]
        miss_tokens = [token for token in unique_tokens if token not in state.resident]
        target_slots = self._select_sim_target_slots(
            state=state,
            protected_tokens=set(hit_tokens),
            effective_candidate_size=effective_candidate_size,
            required_slot_count=len(miss_tokens),
        )

        for token, slot in zip(miss_tokens, target_slots):
            old_token = state.slot_owner[slot]
            if old_token >= 0:
                state.resident.pop(old_token, None)
            state.slot_owner[slot] = token
            state.resident[token] = slot

        self._observe_current_tokens(state, unique_tokens)
        return len(unique_tokens), len(hit_tokens), len(miss_tokens)

    def _expire_recent_window(self, state: _SimReqState) -> None:
        while len(state.recent_queue) >= self.recent_window:
            old_tokens = state.recent_queue.popleft()
            for old_token in old_tokens:
                if old_token in state.resident:
                    state.freq[old_token] = max(0, state.freq[old_token] - 1)

    def _observe_current_tokens(
        self,
        state: _SimReqState,
        tokens: list[int],
    ) -> None:
        token_set = set(tokens)
        for token in token_set:
            if token not in state.resident:
                continue
            state.ema[token] = self.ema_beta * state.ema[token] + (1.0 - self.ema_beta)
            state.freq[token] += 1
            state.last_used[token] = state.step
        state.recent_queue.append(tuple(token_set))

    def _sim_evict_scores(
        self,
        state: _SimReqState,
        protected_slots: set[int],
    ) -> list[float]:
        scores: list[float] = []
        for slot, token in enumerate(state.slot_owner):
            if slot in protected_slots:
                scores.append(-1.0e9)
                continue
            if token < 0:
                scores.append(1.0e9)
                continue
            last_used = state.last_used[token]
            age = state.step if last_used < 0 else state.step - last_used
            scores.append(
                self.age_weight * age
                - self.recent_weight * state.freq[token]
                - self.ema_weight * state.ema[token]
            )
        return scores

    def _rank_slots_by_score(
        self,
        scores: list[float],
        slots: list[int],
    ) -> list[int]:
        ranked = sorted(
            enumerate(slots),
            key=lambda item: (-scores[item[1]], item[0]),
        )
        return [slot for _, slot in ranked]

    def _select_sim_target_slots(
        self,
        state: _SimReqState,
        protected_tokens: set[int],
        effective_candidate_size: int,
        required_slot_count: int,
    ) -> list[int]:
        if required_slot_count <= 0:
            state.cursor = (state.cursor + effective_candidate_size) % self.buffer_size
            return []

        protected_slots = {
            state.resident[token]
            for token in protected_tokens
            if token in state.resident
        }
        scores = self._sim_evict_scores(state, protected_slots)
        candidate_slots = [
            (state.cursor + offset) % self.buffer_size
            for offset in range(effective_candidate_size)
        ]
        candidate_available = sum(
            1 for slot in candidate_slots if slot not in protected_slots
        )
        search_slots = (
            list(range(self.buffer_size))
            if candidate_available < required_slot_count
            else candidate_slots
        )
        selected_slots = self._rank_slots_by_score(scores, search_slots)
        state.cursor = (state.cursor + effective_candidate_size) % self.buffer_size
        return selected_slots[:required_slot_count]


class LRUKVCacheSimulator:
    """Offline simulator for a simple global LRU resident-cache policy.

    The input and output schema intentionally match HotKVCacheSimulator so the
    existing dump analysis and sweep flow can compare policies directly.
    """

    def __init__(
        self,
        buffer_size: int,
        topk_length: int | None = None,
    ) -> None:
        if buffer_size < 1:
            raise ValueError("buffer_size must be >= 1")
        if topk_length is not None and topk_length < 1:
            raise ValueError("topk_length must be >= 1")
        if topk_length is not None and topk_length > buffer_size:
            raise ValueError("topk_length must be <= buffer_size")
        self.buffer_size = buffer_size
        self.topk_length = topk_length

    def _new_state_for_test(self) -> _LRUSimReqState:
        return _LRUSimReqState(slot_owner=[-1 for _ in range(self.buffer_size)])

    def run(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        states: dict[tuple[str, int], _LRUSimReqState] = {}
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
                state = states.setdefault((layer, req_id), _LRUSimReqState(
                    slot_owner=[-1 for _ in range(self.buffer_size)]))
                needed, hits, misses = self._process_row(state, tokens)
                _accumulate(global_stats, needed, hits, misses)
                _accumulate(by_layer[layer], needed, hits, misses)

        _finish_stats(global_stats)
        by_layer_result = dict(by_layer)
        for stats in by_layer_result.values():
            _finish_stats(stats)
        return {"global": global_stats, "by_layer": by_layer_result}

    def _process_row(
        self,
        state: _LRUSimReqState,
        tokens: list[int],
    ) -> tuple[int, int, int]:
        state.step += 1
        unique_tokens = _normalize_sim_tokens(tokens, self.topk_length)
        hit_tokens = [token for token in unique_tokens if token in state.resident]
        miss_tokens = [token for token in unique_tokens if token not in state.resident]
        target_slots = self._select_lru_target_slots(
            state=state,
            protected_tokens=set(hit_tokens),
            required_slot_count=len(miss_tokens),
        )

        for token, slot in zip(miss_tokens, target_slots):
            old_token = state.slot_owner[slot]
            if old_token >= 0:
                state.resident.pop(old_token, None)
            state.slot_owner[slot] = token
            state.resident[token] = slot

        for token in unique_tokens:
            if token in state.resident:
                state.last_used[token] = state.step
        return len(unique_tokens), len(hit_tokens), len(miss_tokens)

    def _select_lru_target_slots(
        self,
        state: _LRUSimReqState,
        protected_tokens: set[int],
        required_slot_count: int,
    ) -> list[int]:
        if required_slot_count <= 0:
            return []

        protected_slots = {
            state.resident[token]
            for token in protected_tokens
            if token in state.resident
        }
        free_slots = [
            slot
            for slot, token in enumerate(state.slot_owner)
            if token < 0 and slot not in protected_slots
        ]
        resident_slots = [
            slot
            for slot, token in enumerate(state.slot_owner)
            if token >= 0 and slot not in protected_slots
        ]
        resident_slots.sort(
            key=lambda slot: (state.last_used[state.slot_owner[slot]], slot)
        )
        selected_slots = free_slots + resident_slots
        if len(selected_slots) < required_slot_count:
            raise RuntimeError(
                "not enough unprotected slots for LRU miss tokens; "
                "ensure topk_length <= buffer_size"
            )
        return selected_slots[:required_slot_count]


def _select_lru_target_slots(
    *,
    slot_to_token: torch.Tensor,
    slot_last_used: torch.Tensor,
    protected_slot_mask: torch.Tensor,
    target_rank_count: int,
) -> torch.Tensor:
    """Select free slots first, then least-recently-used slots."""
    capacity = slot_to_token.shape[1]
    device = slot_to_token.device
    slot_order = torch.arange(
        capacity,
        dtype=torch.int64,
        device=device,
    ).view(1, -1)
    free_score = slot_order
    resident_score = (
        (torch.clamp(slot_last_used.to(torch.int64), min=0) + 1)
        * (capacity + 1)
        + slot_order
    )
    lru_score = torch.where(slot_to_token < 0, free_score, resident_score)
    lru_score = torch.where(
        protected_slot_mask,
        torch.full_like(lru_score, torch.iinfo(torch.int64).max),
        lru_score,
    )
    return torch.topk(
        lru_score,
        k=target_rank_count,
        dim=1,
        largest=False,
    ).indices.to(torch.int32)


def _select_hot_kv_target_slots(
    *,
    evict_score: torch.Tensor,
    protected_slot_mask: torch.Tensor,
    evict_cursor: torch.Tensor,
    effective_candidate_size: int,
    target_rank_count: int,
    required_slot_count: torch.Tensor,
    capacity: int,
) -> torch.Tensor:
    """Select eviction target slots with full-scan fallback.

    Candidate-window scoring is the normal fast path. If a row does not have
    enough unprotected slots in that window to satisfy its misses, scan the full
    row so a miss token does not overwrite a current-step hit when a safe slot
    exists outside the candidate window.
    """
    device = evict_score.device
    candidate_offsets = torch.arange(
        effective_candidate_size,
        dtype=torch.int64,
        device=device,
    ).view(1, -1)
    candidate_slots = (evict_cursor.view(-1, 1) + candidate_offsets) % capacity
    candidate_scores = torch.gather(evict_score, 1, candidate_slots)
    target_ranks = torch.topk(candidate_scores, k=target_rank_count, dim=1).indices
    target_slots_by_rank = torch.gather(candidate_slots, 1, target_ranks)

    candidate_protected = torch.gather(protected_slot_mask, 1, candidate_slots)
    candidate_available_count = (
        (~candidate_protected).sum(dim=1).to(required_slot_count.dtype)
    )
    needs_full_scan = candidate_available_count < required_slot_count
    if bool(needs_full_scan.any().item()):
        fallback_ranks = torch.topk(evict_score, k=target_rank_count, dim=1).indices
        target_slots_by_rank = torch.where(
            needs_full_scan.view(-1, 1),
            fallback_ranks,
            target_slots_by_rank,
        )

    evict_cursor[:] = (evict_cursor + effective_candidate_size) % capacity
    return target_slots_by_rank.to(torch.int32)


def update_lru_kv_cache_state(
    *,
    topk_indices: torch.Tensor,
    token_to_slot: torch.Tensor,
    slot_to_token: torch.Tensor,
    slot_last_used: torch.Tensor,
    last_req_ids: torch.Tensor,
    req_ids: torch.Tensor,
    step_value: int,
    capacity: int,
    max_model_len: int,
) -> dict[str, torch.Tensor]:
    """Update simple LRU resident-cache state for one decode step.

    The state tensors are mutated in place. The output schema intentionally
    matches ``update_hot_kv_cache_state`` so SFA can share KV loading logic.
    """
    num_reqs = topk_indices.shape[0]
    topk_width = topk_indices.shape[1]
    device = topk_indices.device

    valid_topk = (topk_indices >= 0) & (topk_indices < max_model_len)
    safe_topk = torch.clamp(topk_indices, min=0, max=max_model_len - 1)

    req_changed_mask = last_req_ids != req_ids
    if bool(req_changed_mask.any().item()):
        changed_rows = req_changed_mask.nonzero(as_tuple=True)[0]
        token_to_slot[changed_rows] = -1
        slot_to_token[changed_rows] = -1
        slot_last_used[changed_rows] = -1

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

    miss_count_by_req = miss_mask.sum(dim=1).to(torch.int64)
    target_rank_count = min(topk_width, capacity)
    target_slots_by_rank = _select_lru_target_slots(
        slot_to_token=slot_to_token,
        slot_last_used=slot_last_used,
        protected_slot_mask=protected_slot_mask,
        target_rank_count=target_rank_count,
    )
    miss_rank = torch.clamp(torch.cumsum(miss_mask.to(torch.int32), dim=1) - 1, min=0)
    target_slots = torch.gather(target_slots_by_rank, 1, miss_rank.to(torch.int64))
    target_slots = torch.where(
        miss_rank < miss_count_by_req.view(-1, 1),
        target_slots,
        torch.full_like(target_slots, -1),
    )

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

    last_req_ids[...] = req_ids
    return {
        "current_slots": current_slots,
        "load_token_indices": load_token_indices,
        "hit_mask": hit_mask,
        "miss_mask": miss_mask,
        "valid_topk": valid_topk,
    }


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
        changed_rows = req_changed_mask.nonzero(as_tuple=True)[0]
        token_to_slot[changed_rows] = -1
        slot_to_token[changed_rows] = -1
        slot_freq[changed_rows] = 0
        slot_ema[changed_rows] = 0
        slot_last_used[changed_rows] = -1
        recent_tokens[changed_rows] = -1
        evict_cursor[changed_rows] = 0

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

    effective_candidate_size = resolve_hot_kv_candidate_size(
        candidate_size=candidate_size,
        topk_width=topk_width,
        capacity=capacity,
    )
    target_rank_count = min(topk_width, effective_candidate_size)
    miss_count_by_req = miss_mask.sum(dim=1).to(torch.int64)
    target_slots_by_rank = _select_hot_kv_target_slots(
        evict_score=evict_score,
        protected_slot_mask=protected_slot_mask,
        evict_cursor=evict_cursor,
        effective_candidate_size=effective_candidate_size,
        target_rank_count=target_rank_count,
        required_slot_count=miss_count_by_req,
        capacity=capacity,
    )

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
