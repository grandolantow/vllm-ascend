import torch

from vllm_ascend.attention.hot_kv_cache import HotKVCacheSimulator


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
