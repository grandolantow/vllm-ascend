import pytest
import torch

from vllm_ascend.attention.hot_kv_cache import (
    HotKVCacheSimulator,
    LRUKVCacheSimulator,
    _select_hot_kv_target_slots,
    resolve_hot_kv_candidate_size,
    update_hot_kv_cache_state,
    update_lru_kv_cache_state,
)


def test_hot_kv_cache_simulator_counts_hits_after_resident_load():
    records = [
        {"layer": "layer.0", "step": 0, "req_ids": [7], "topk_indices": torch.tensor([[1, 2, 3, 4]])},
        {"layer": "layer.0", "step": 1, "req_ids": [7], "topk_indices": torch.tensor([[3, 4, 5, 6]])},
    ]

    stats = HotKVCacheSimulator(buffer_size=6, recent_window=4).run(records)

    assert stats["global"]["needed"] == 8
    assert stats["global"]["hits"] == 2
    assert stats["global"]["misses"] == 6
    assert stats["global"]["hit_rate"] == 0.25
    assert stats["by_layer"]["layer.0"]["hits"] == 2


def test_lru_kv_cache_simulator_counts_hits_after_resident_load():
    records = [
        {"layer": "layer.0", "step": 0, "req_ids": [7], "topk_indices": torch.tensor([[1, 2, 3, 4]])},
        {"layer": "layer.0", "step": 1, "req_ids": [7], "topk_indices": torch.tensor([[3, 4, 5, 6]])},
    ]

    stats = LRUKVCacheSimulator(buffer_size=6).run(records)

    assert stats["global"]["requests"] == 2
    assert stats["global"]["needed"] == 8
    assert stats["global"]["hits"] == 2
    assert stats["global"]["misses"] == 6
    assert stats["global"]["hit_rate"] == 0.25
    assert stats["by_layer"]["layer.0"]["hits"] == 2


def test_lru_kv_cache_simulator_evicts_oldest_non_protected_slot():
    simulator = LRUKVCacheSimulator(buffer_size=4)
    state = simulator._new_state_for_test()
    state.slot_owner[:] = [10, 11, 12, 13]
    state.step = 10
    for slot, token in enumerate(state.slot_owner):
        state.resident[token] = slot
        state.last_used[token] = 10
    state.last_used[12] = 1
    state.last_used[13] = 0

    needed, hits, misses = simulator._process_row(state, [10, 20])

    assert (needed, hits, misses) == (2, 1, 1)
    assert state.slot_owner == [10, 11, 12, 20]
    assert state.resident[20] == 3
    assert 13 not in state.resident
    assert state.last_used[10] == 11
    assert state.last_used[20] == 11


def test_lru_kv_cache_simulator_respects_topk_length():
    records = [
        {"layer": "layer.0", "step": 0, "req_ids": [0], "topk_indices": torch.tensor([[1, 2, 3, 4]])},
        {"layer": "layer.0", "step": 1, "req_ids": [0], "topk_indices": torch.tensor([[1, 2, 5, 6]])},
    ]

    stats = LRUKVCacheSimulator(buffer_size=2, topk_length=2).run(records)

    assert stats["global"]["requests"] == 2
    assert stats["global"]["needed"] == 4
    assert stats["global"]["hits"] == 2
    assert stats["global"]["misses"] == 2
    assert stats["global"]["hit_rate"] == 0.5


def test_lru_kv_cache_simulator_deduplicates_repeated_tokens():
    records = [
        {"layer": "layer.0", "step": 0, "req_ids": [0], "topk_indices": torch.tensor([[5, 5, 6, 5]])},
        {"layer": "layer.0", "step": 1, "req_ids": [0], "topk_indices": torch.tensor([[5, 6, 6, 5]])},
    ]

    stats = LRUKVCacheSimulator(buffer_size=2).run(records)

    assert stats["global"]["needed"] == 4
    assert stats["global"]["hits"] == 2
    assert stats["global"]["misses"] == 2
    assert stats["global"]["hit_rate"] == 0.5


def test_lru_kv_cache_simulator_validates_buffer_and_topk_length():
    with pytest.raises(ValueError, match="buffer_size must be >= 1"):
        LRUKVCacheSimulator(buffer_size=0)

    with pytest.raises(ValueError, match="topk_length must be >= 1"):
        LRUKVCacheSimulator(buffer_size=4, topk_length=0)

    with pytest.raises(ValueError, match="topk_length must be <= buffer_size"):
        LRUKVCacheSimulator(buffer_size=2, topk_length=4)


def test_lru_kv_cache_state_outputs_hot_kv_compatible_schema():
    topk_indices = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
    token_to_slot = torch.full((1, 16), -1, dtype=torch.int32)
    slot_to_token = torch.full((1, 6), -1, dtype=torch.int64)
    slot_last_used = torch.full((1, 6), -1, dtype=torch.int32)
    last_req_ids = torch.tensor([7], dtype=torch.int64)
    req_ids = torch.tensor([7], dtype=torch.int64)

    token_to_slot[0, 1] = 0
    slot_to_token[0, 0] = 1
    slot_last_used[0, 0] = 4

    result = update_lru_kv_cache_state(
        topk_indices=topk_indices,
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_last_used=slot_last_used,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=5,
        capacity=6,
        max_model_len=16,
    )

    assert set(result.keys()) == {
        "current_slots",
        "load_token_indices",
        "hit_mask",
        "miss_mask",
        "valid_topk",
    }
    assert result["current_slots"].shape == topk_indices.shape
    assert result["load_token_indices"].shape == (1, 6)
    assert result["hit_mask"].sum().item() == 1
    assert result["miss_mask"].sum().item() == 3
    assert result["current_slots"][0, 0].item() == 0

    load_plan = result["load_token_indices"][0].tolist()
    loaded_slots = [slot for slot, token in enumerate(load_plan) if token >= 0]
    loaded_tokens = [load_plan[slot] for slot in loaded_slots]
    assert loaded_tokens == [2, 3, 4]
    for slot, token in zip(loaded_slots, loaded_tokens):
        assert token_to_slot[0, token].item() == slot
        assert slot_to_token[0, slot].item() == token


def test_lru_kv_cache_state_deduplicates_repeated_miss_tokens():
    topk_indices = torch.tensor([[5, 5, 6, 5]], dtype=torch.int64)
    token_to_slot = torch.full((1, 16), -1, dtype=torch.int32)
    slot_to_token = torch.full((1, 4), -1, dtype=torch.int64)
    slot_last_used = torch.full((1, 4), -1, dtype=torch.int32)
    last_req_ids = torch.tensor([11], dtype=torch.int64)
    req_ids = torch.tensor([11], dtype=torch.int64)

    result = update_lru_kv_cache_state(
        topk_indices=topk_indices,
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_last_used=slot_last_used,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=1,
        capacity=4,
        max_model_len=16,
    )

    current_slots = result["current_slots"][0].tolist()
    assert current_slots[0] == current_slots[1] == current_slots[3]
    assert current_slots[0] != current_slots[2]
    assert (result["load_token_indices"] == 5).sum().item() == 1
    assert (result["load_token_indices"] == 6).sum().item() == 1
    assert result["hit_mask"].sum().item() == 0
    assert result["miss_mask"].sum().item() == 2


def test_lru_kv_cache_state_never_evicts_current_step_hit_slot():
    topk_indices = torch.tensor([[10, 20]], dtype=torch.int64)
    token_to_slot = torch.full((1, 32), -1, dtype=torch.int32)
    slot_to_token = torch.tensor([[10, 11]], dtype=torch.int64)
    slot_last_used = torch.tensor([[0, 7]], dtype=torch.int32)
    last_req_ids = torch.tensor([3], dtype=torch.int64)
    req_ids = torch.tensor([3], dtype=torch.int64)
    token_to_slot[0, 10] = 0
    token_to_slot[0, 11] = 1

    result = update_lru_kv_cache_state(
        topk_indices=topk_indices,
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_last_used=slot_last_used,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=8,
        capacity=2,
        max_model_len=32,
    )

    assert result["current_slots"].tolist() == [[0, 1]]
    assert result["load_token_indices"].tolist() == [[-1, 20]]
    assert slot_to_token.tolist() == [[10, 20]]
    assert token_to_slot[0, 10].item() == 0
    assert token_to_slot[0, 20].item() == 1
    assert token_to_slot[0, 11].item() == -1


def test_lru_kv_cache_state_clears_row_on_request_switch():
    topk_indices = torch.tensor([[5, 6, -1]], dtype=torch.int64)
    token_to_slot = torch.full((1, 16), -1, dtype=torch.int32)
    slot_to_token = torch.tensor([[1, 2, 3, -1]], dtype=torch.int64)
    slot_last_used = torch.full((1, 4), 9, dtype=torch.int32)
    last_req_ids = torch.tensor([10], dtype=torch.int64)
    req_ids = torch.tensor([11], dtype=torch.int64)
    token_to_slot[0, 1] = 0
    token_to_slot[0, 2] = 1
    token_to_slot[0, 3] = 2

    result = update_lru_kv_cache_state(
        topk_indices=topk_indices,
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_last_used=slot_last_used,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=10,
        capacity=4,
        max_model_len=16,
    )

    assert last_req_ids.tolist() == [11]
    assert slot_to_token.tolist() == [[5, 6, -1, -1]]
    assert token_to_slot[0, 1].item() == -1
    assert token_to_slot[0, 2].item() == -1
    assert token_to_slot[0, 3].item() == -1
    assert result["current_slots"].tolist() == [[0, 1, -1]]
    assert result["load_token_indices"].tolist() == [[5, 6, -1, -1]]


def test_hot_kv_cache_simulator_eviction_keeps_recent_tokens():
    records = [
        {"layer": "layer.0", "step": 0, "req_ids": [0], "topk_indices": torch.tensor([[1, 2, 3, 4]])},
        {"layer": "layer.0", "step": 1, "req_ids": [0], "topk_indices": torch.tensor([[1, 2, 5, 6]])},
        {"layer": "layer.0", "step": 2, "req_ids": [0], "topk_indices": torch.tensor([[1, 2, 7, 8]])},
    ]

    stats = HotKVCacheSimulator(buffer_size=4, recent_window=2).run(records)

    assert stats["global"]["needed"] == 12
    assert stats["global"]["hits"] >= 4
    assert stats["by_layer"]["layer.0"]["requests"] == 3


def test_resolve_hot_kv_candidate_size_matches_online_floor_and_capacity():
    assert resolve_hot_kv_candidate_size(
        candidate_size=256,
        topk_width=2048,
        capacity=8192,
    ) == 4096
    assert resolve_hot_kv_candidate_size(
        candidate_size=8192,
        topk_width=2048,
        capacity=4096,
    ) == 4096
    assert resolve_hot_kv_candidate_size(
        candidate_size=1,
        topk_width=2,
        capacity=8,
    ) == 4
    assert resolve_hot_kv_candidate_size(
        candidate_size=0,
        topk_width=2,
        capacity=8,
    ) == 4


def test_select_hot_kv_target_slots_falls_back_when_candidate_slots_are_protected():
    evict_score = torch.tensor(
        [[-1.0e9, -1.0e9, -1.0e9, 0.5, 0.2, 0.1]],
        dtype=torch.float32,
    )
    protected_slot_mask = torch.tensor(
        [[True, True, True, False, False, False]],
        dtype=torch.bool,
    )
    evict_cursor = torch.tensor([0], dtype=torch.int64)

    target_slots = _select_hot_kv_target_slots(
        evict_score=evict_score,
        protected_slot_mask=protected_slot_mask,
        evict_cursor=evict_cursor,
        effective_candidate_size=3,
        target_rank_count=1,
        required_slot_count=torch.tensor([1], dtype=torch.int64),
        capacity=6,
    )

    assert target_slots.tolist() == [[3]]
    assert evict_cursor.tolist() == [3]


def test_select_hot_kv_target_slots_falls_back_when_legal_slots_are_insufficient():
    evict_score = torch.tensor(
        [[-1.0e9, -1.0e9, 0.1, 0.4, 0.3, 0.2]],
        dtype=torch.float32,
    )
    protected_slot_mask = torch.tensor(
        [[True, True, False, False, False, False]],
        dtype=torch.bool,
    )
    evict_cursor = torch.tensor([0], dtype=torch.int64)

    target_slots = _select_hot_kv_target_slots(
        evict_score=evict_score,
        protected_slot_mask=protected_slot_mask,
        evict_cursor=evict_cursor,
        effective_candidate_size=3,
        target_rank_count=2,
        required_slot_count=torch.tensor([2], dtype=torch.int64),
        capacity=6,
    )

    assert target_slots.tolist() == [[3, 4]]
    assert not protected_slot_mask[0, target_slots[0].to(torch.int64)].any().item()


def test_hot_kv_cache_simulator_uses_candidate_window_for_empty_slots():
    simulator = HotKVCacheSimulator(buffer_size=6, candidate_size=1)
    state = simulator._new_state_for_test()
    state.slot_owner[:] = [-1, -1, 10, 11, -1, 12]
    state.cursor = 4
    for token, slot in [(10, 2), (11, 3), (12, 5)]:
        state.resident[token] = slot
        state.freq[token] = 1
        state.ema[token] = 0.1
        state.last_used[token] = 1

    needed, hits, misses = simulator._process_row(state, [99])

    assert (needed, hits, misses) == (1, 0, 1)
    assert state.slot_owner[0] == -1
    assert state.slot_owner[4] == 99
    assert state.resident[99] == 4
    assert state.cursor == 0


def test_hot_kv_cache_simulator_batches_multi_miss_target_slots():
    simulator = HotKVCacheSimulator(
        buffer_size=6,
        candidate_size=1,
        recent_weight=0.0,
        ema_weight=0.0,
        age_weight=1.0,
    )
    state = simulator._new_state_for_test()
    state.slot_owner[:] = [10, 11, 12, 13, 14, 15]
    state.cursor = 0
    state.step = 10
    for token, slot in zip(state.slot_owner, range(6)):
        state.resident[token] = slot
        state.freq[token] = 0
        state.ema[token] = 0.0
        state.last_used[token] = 10
    state.last_used[12] = 1
    state.last_used[13] = 0
    state.last_used[14] = 0
    state.last_used[15] = 0

    needed, hits, misses = simulator._process_row(state, [20, 21])

    assert (needed, hits, misses) == (2, 0, 2)
    assert state.slot_owner[3] == 20
    assert state.slot_owner[2] == 21
    assert state.resident[20] == 3
    assert state.resident[21] == 2
    assert 12 not in state.resident
    assert 13 not in state.resident
    assert state.cursor == 4


def test_hot_kv_cache_simulator_advances_cursor_on_all_hit_row():
    simulator = HotKVCacheSimulator(buffer_size=6, candidate_size=1)
    state = simulator._new_state_for_test()
    state.slot_owner[:] = [10, 11, -1, -1, -1, -1]
    state.cursor = 1
    for token, slot in [(10, 0), (11, 1)]:
        state.resident[token] = slot
        state.freq[token] = 1
        state.ema[token] = 0.1
        state.last_used[token] = 1

    needed, hits, misses = simulator._process_row(state, [10, 11])

    assert (needed, hits, misses) == (2, 2, 0)
    assert state.cursor == 5
    assert state.slot_owner[0:2] == [10, 11]


def test_hot_kv_cache_state_refreshes_stats_on_all_hit_step():
    topk_indices = torch.tensor([[1, 2, 3]], dtype=torch.int64)
    token_to_slot = torch.full((1, 16), -1, dtype=torch.int32)
    slot_to_token = torch.full((1, 8), -1, dtype=torch.int64)
    slot_freq = torch.zeros((1, 8), dtype=torch.float32)
    slot_ema = torch.zeros((1, 8), dtype=torch.float32)
    slot_last_used = torch.full((1, 8), -1, dtype=torch.int32)
    evict_cursor = torch.zeros((1,), dtype=torch.int64)
    last_req_ids = torch.tensor([7], dtype=torch.int64)
    req_ids = torch.tensor([7], dtype=torch.int64)
    recent_tokens = torch.full((1, 4, 3), -1, dtype=torch.int64)

    token_to_slot[0, 1] = 0
    token_to_slot[0, 2] = 1
    token_to_slot[0, 3] = 2
    slot_to_token[0, 0:3] = torch.tensor([1, 2, 3], dtype=torch.int64)
    slot_freq[0, 0:3] = torch.tensor([5.0, 6.0, 7.0], dtype=torch.float32)
    slot_ema[0, 0:3] = torch.tensor([0.2, 0.3, 0.4], dtype=torch.float32)
    slot_last_used[0, 0:3] = torch.tensor([4, 4, 4], dtype=torch.int32)

    result = update_hot_kv_cache_state(
        topk_indices=topk_indices,
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_freq=slot_freq,
        slot_ema=slot_ema,
        slot_last_used=slot_last_used,
        evict_cursor=evict_cursor,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=5,
        capacity=8,
        max_model_len=16,
        recent_window=4,
        recent_tokens=recent_tokens,
        recent_weight=1.0,
        ema_weight=0.5,
        age_weight=0.01,
        ema_beta=0.9,
        candidate_size=4,
    )

    assert result["current_slots"].tolist() == [[0, 1, 2]]
    assert torch.all(result["load_token_indices"] == -1)
    assert result["hit_mask"].sum().item() == 3
    assert result["miss_mask"].sum().item() == 0
    assert slot_last_used[0, 0:3].tolist() == [5, 5, 5]
    assert slot_freq[0, 0:3].tolist() == [6.0, 7.0, 8.0]
    assert slot_ema[0, 0:3].tolist() == pytest.approx([0.28, 0.37, 0.46])


def test_hot_kv_cache_state_deduplicates_repeated_miss_tokens():
    topk_indices = torch.tensor([[5, 5, 6, 5]], dtype=torch.int64)
    token_to_slot = torch.full((1, 16), -1, dtype=torch.int32)
    slot_to_token = torch.full((1, 8), -1, dtype=torch.int64)
    slot_freq = torch.zeros((1, 8), dtype=torch.float32)
    slot_ema = torch.zeros((1, 8), dtype=torch.float32)
    slot_last_used = torch.full((1, 8), -1, dtype=torch.int32)
    evict_cursor = torch.zeros((1,), dtype=torch.int64)
    last_req_ids = torch.tensor([11], dtype=torch.int64)
    req_ids = torch.tensor([11], dtype=torch.int64)
    recent_tokens = torch.full((1, 4, 4), -1, dtype=torch.int64)

    result = update_hot_kv_cache_state(
        topk_indices=topk_indices,
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_freq=slot_freq,
        slot_ema=slot_ema,
        slot_last_used=slot_last_used,
        evict_cursor=evict_cursor,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=1,
        capacity=8,
        max_model_len=16,
        recent_window=4,
        recent_tokens=recent_tokens,
        recent_weight=1.0,
        ema_weight=0.5,
        age_weight=0.01,
        ema_beta=0.9,
        candidate_size=4,
    )

    current_slots = result["current_slots"][0]
    token5_slot = int(current_slots[0].item())
    token6_slot = int(current_slots[2].item())

    assert token5_slot >= 0
    assert token6_slot >= 0
    assert token5_slot != token6_slot
    assert current_slots.tolist()[0] == current_slots.tolist()[1] == current_slots.tolist()[3]
    assert token_to_slot[0, 5].item() == token5_slot
    assert token_to_slot[0, 6].item() == token6_slot
    assert slot_to_token[0].tolist().count(5) == 1
    assert slot_to_token[0].tolist().count(6) == 1
    assert (result["load_token_indices"] == 5).sum().item() == 1
    assert (result["load_token_indices"] == 6).sum().item() == 1
    assert result["hit_mask"].sum().item() == 0
    assert result["miss_mask"].sum().item() == 2


def test_hot_kv_cache_state_uses_recent_window_for_slot_freq():
    token_to_slot = torch.full((1, 16), -1, dtype=torch.int32)
    slot_to_token = torch.full((1, 8), -1, dtype=torch.int64)
    slot_freq = torch.zeros((1, 8), dtype=torch.float32)
    slot_ema = torch.zeros((1, 8), dtype=torch.float32)
    slot_last_used = torch.full((1, 8), -1, dtype=torch.int32)
    evict_cursor = torch.zeros((1,), dtype=torch.int64)
    last_req_ids = torch.tensor([1], dtype=torch.int64)
    req_ids = torch.tensor([1], dtype=torch.int64)
    recent_tokens = torch.full((1, 2, 4), -1, dtype=torch.int64)

    first = update_hot_kv_cache_state(
        topk_indices=torch.tensor([[1, 2, -1, -1]], dtype=torch.int64),
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_freq=slot_freq,
        slot_ema=slot_ema,
        slot_last_used=slot_last_used,
        evict_cursor=evict_cursor,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=1,
        capacity=8,
        max_model_len=16,
        recent_window=2,
        recent_tokens=recent_tokens,
        recent_weight=1.0,
        ema_weight=0.5,
        age_weight=0.01,
        ema_beta=0.9,
        candidate_size=4,
    )
    token1_slot = int(first["current_slots"][0, 0].item())
    token2_slot = int(first["current_slots"][0, 1].item())
    assert slot_freq[0, token1_slot].item() == 1.0
    assert slot_freq[0, token2_slot].item() == 1.0

    update_hot_kv_cache_state(
        topk_indices=torch.tensor([[2, 3, -1, -1]], dtype=torch.int64),
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_freq=slot_freq,
        slot_ema=slot_ema,
        slot_last_used=slot_last_used,
        evict_cursor=evict_cursor,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=2,
        capacity=8,
        max_model_len=16,
        recent_window=2,
        recent_tokens=recent_tokens,
        recent_weight=1.0,
        ema_weight=0.5,
        age_weight=0.01,
        ema_beta=0.9,
        candidate_size=4,
    )
    assert slot_freq[0, token1_slot].item() == 1.0
    assert slot_freq[0, token2_slot].item() == 2.0

    update_hot_kv_cache_state(
        topk_indices=torch.tensor([[3, 4, -1, -1]], dtype=torch.int64),
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_freq=slot_freq,
        slot_ema=slot_ema,
        slot_last_used=slot_last_used,
        evict_cursor=evict_cursor,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=3,
        capacity=8,
        max_model_len=16,
        recent_window=2,
        recent_tokens=recent_tokens,
        recent_weight=1.0,
        ema_weight=0.5,
        age_weight=0.01,
        ema_beta=0.9,
        candidate_size=4,
    )

    assert slot_freq[0, token1_slot].item() == 0.0
    assert slot_freq[0, token2_slot].item() == 1.0
    assert recent_tokens.tolist() == [[[3, 4, -1, -1], [2, 3, -1, -1]]]


def test_hot_kv_cache_state_clears_recent_tokens_on_request_switch():
    topk_indices = torch.tensor([[5, 6, -1, -1]], dtype=torch.int64)
    token_to_slot = torch.full((1, 16), -1, dtype=torch.int32)
    slot_to_token = torch.full((1, 8), -1, dtype=torch.int64)
    slot_freq = torch.ones((1, 8), dtype=torch.float32)
    slot_ema = torch.ones((1, 8), dtype=torch.float32)
    slot_last_used = torch.full((1, 8), 7, dtype=torch.int32)
    evict_cursor = torch.full((1,), 3, dtype=torch.int64)
    last_req_ids = torch.tensor([10], dtype=torch.int64)
    req_ids = torch.tensor([11], dtype=torch.int64)
    recent_tokens = torch.full((1, 2, 4), 99, dtype=torch.int64)

    update_hot_kv_cache_state(
        topk_indices=topk_indices,
        token_to_slot=token_to_slot,
        slot_to_token=slot_to_token,
        slot_freq=slot_freq,
        slot_ema=slot_ema,
        slot_last_used=slot_last_used,
        evict_cursor=evict_cursor,
        last_req_ids=last_req_ids,
        req_ids=req_ids,
        step_value=8,
        capacity=8,
        max_model_len=16,
        recent_window=2,
        recent_tokens=recent_tokens,
        recent_weight=1.0,
        ema_weight=0.5,
        age_weight=0.01,
        ema_beta=0.9,
        candidate_size=4,
    )

    assert last_req_ids.tolist() == [11]
    assert evict_cursor.item() == 0
    assert recent_tokens[0, 0].tolist() == [-1, -1, -1, -1]
    assert recent_tokens[0, 1].tolist() == [5, 6, -1, -1]
    assert slot_freq.sum().item() == 2.0
    assert sorted(slot_freq[0].tolist()) == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0
    ]
    assert sorted(slot_ema[0].tolist()) == [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5
    ]
    assert sorted(slot_last_used[0].tolist()) == [
        -1, -1, -1, -1, -1, -1, 8, 8
    ]
