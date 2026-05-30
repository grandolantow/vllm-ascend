import numpy as np
import pytest

from vllm_ascend.attention.cpu_cache_miss_topk import (
    NUMBA_AVAILABLE,
    make_cpu_cache_miss_topk_workspace,
    update_topk_indices_cpu,
)


def test_cpu_topk_matches_spec_example_with_raw_tokens():
    if not NUMBA_AVAILABLE:
        pytest.skip("numba is required for CPU cache-miss topk")

    req_ids = np.array([3], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=6, max_token=128)

    out = update_topk_indices_cpu(req_ids, last_req_ids, old, new, workspace)

    np.testing.assert_array_equal(
        out, np.array([[5, -1, 18, -1, 30, 33]], dtype=np.int32))
    np.testing.assert_array_equal(
        old, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int64))
    np.testing.assert_array_equal(last_req_ids, req_ids)
    assert out is new


def test_cpu_topk_workspace_uses_cpu_tensors():
    torch = pytest.importorskip("torch")
    workspace = make_cpu_cache_miss_topk_workspace(topk=6, max_token=128)

    assert isinstance(workspace.mark_workspace, torch.Tensor)
    assert isinstance(workspace.miss_workspace, torch.Tensor)
    assert isinstance(workspace.epochs, torch.Tensor)
    assert workspace.mark_workspace.device.type == "cpu"
    assert workspace.miss_workspace.device.type == "cpu"
    assert workspace.epochs.device.type == "cpu"
    assert workspace.mark_workspace.dtype == torch.int32
    assert workspace.miss_workspace.dtype == torch.int32
    assert workspace.epochs.dtype == torch.int32
    assert workspace.mark_workspace.ndim == 2
    assert workspace.miss_workspace.ndim == 2
    assert workspace.epochs.ndim == 1
    assert workspace.mark_workspace.shape[1] == 128
    assert workspace.miss_workspace.shape[1] == 6
    assert workspace.epochs.shape[0] == workspace.mark_workspace.shape[0]
    assert workspace.miss_workspace.shape[0] == workspace.mark_workspace.shape[0]
    assert workspace.workspace_threads == workspace.mark_workspace.shape[0]


def test_cpu_topk_treats_changed_req_id_as_all_miss():
    if not NUMBA_AVAILABLE:
        pytest.skip("numba is required for CPU cache-miss topk")

    req_ids = np.array([7], dtype=np.int64)
    last_req_ids = np.array([3], dtype=np.int64)
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=6, max_token=128)

    out = update_topk_indices_cpu(req_ids, last_req_ids, old, new, workspace)

    np.testing.assert_array_equal(
        out, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32))
    np.testing.assert_array_equal(
        old, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int64))
    np.testing.assert_array_equal(last_req_ids, req_ids)


def test_cpu_topk_keeps_same_raw_token_per_row():
    if not NUMBA_AVAILABLE:
        pytest.skip("numba is required for CPU cache-miss topk")

    req_ids = np.array([3, 7], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([
        [5, 10, -1, -1],
        [5, 20, -1, -1],
    ],
                   dtype=np.int64)
    new = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ],
                   dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=4, max_token=128)

    out = update_topk_indices_cpu(req_ids, last_req_ids, old, new, workspace)

    np.testing.assert_array_equal(
        out,
        np.array([
            [-1, 30, 31, -1],
            [-1, 40, 41, -1],
        ],
                 dtype=np.int32))
    expected_old = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ],
                            dtype=np.int64)
    np.testing.assert_array_equal(old, expected_old)
    np.testing.assert_array_equal(last_req_ids, req_ids)


def test_cpu_topk_rejects_wrong_old_dtype():
    if not NUMBA_AVAILABLE:
        pytest.skip("numba is required for CPU cache-miss topk")

    req_ids = np.array([0], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[1, 2]], dtype=np.int32)
    new = np.array([[2, 3]], dtype=np.int32)
    workspace = make_cpu_cache_miss_topk_workspace(topk=2, max_token=16)

    with pytest.raises(TypeError, match="topk_indices_old must be int64"):
        update_topk_indices_cpu(req_ids, last_req_ids, old, new, workspace)
