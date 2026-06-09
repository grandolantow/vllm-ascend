import pytest
import torch

from vllm_ascend.attention.hot_kv_cache import (
    HotKVCacheSimulator,
    update_hot_kv_cache_state,
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
