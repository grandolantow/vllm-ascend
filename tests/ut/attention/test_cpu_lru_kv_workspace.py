import pytest

from vllm_ascend.attention.cpu_cache_miss_topk import (
    make_cpu_lru_kv_workspace,
)


def test_make_cpu_lru_kv_workspace_allocates_expected_shapes():
    torch = pytest.importorskip("torch")

    workspace = make_cpu_lru_kv_workspace(
        topk=6,
        capacity=12,
        max_token=128,
        workspace_threads=4,
    )

    assert workspace.token_mark_workspace.shape == (4, 128)
    assert workspace.token_slot_workspace.shape == (4, 128)
    assert workspace.miss_token_workspace.shape == (4, 6)
    assert workspace.miss_slot_workspace.shape == (4, 6)
    assert workspace.epochs.shape == (4, )
    assert workspace.token_mark_workspace.dtype == torch.int32
    assert workspace.token_slot_workspace.dtype == torch.int32
    assert workspace.miss_token_workspace.dtype == torch.int32
    assert workspace.miss_slot_workspace.dtype == torch.int32
    assert workspace.epochs.dtype == torch.int32
    assert workspace.token_mark_workspace.is_pinned()
    assert workspace.token_slot_workspace.is_pinned()
    assert workspace.miss_token_workspace.is_pinned()
    assert workspace.miss_slot_workspace.is_pinned()
    assert workspace.epochs.is_pinned()
    assert workspace.topk == 6
    assert workspace.capacity == 12
    assert workspace.max_token == 128
    assert workspace.workspace_threads == 4


def test_make_cpu_lru_kv_workspace_rejects_invalid_sizes():
    pytest.importorskip("torch")

    with pytest.raises(ValueError, match="topk must be positive"):
        make_cpu_lru_kv_workspace(
            topk=0,
            capacity=12,
            max_token=128,
            workspace_threads=4,
        )

    with pytest.raises(ValueError, match="capacity must be positive"):
        make_cpu_lru_kv_workspace(
            topk=6,
            capacity=0,
            max_token=128,
            workspace_threads=4,
        )

    with pytest.raises(ValueError, match="max_token must be positive"):
        make_cpu_lru_kv_workspace(
            topk=6,
            capacity=12,
            max_token=0,
            workspace_threads=4,
        )
