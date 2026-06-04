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
