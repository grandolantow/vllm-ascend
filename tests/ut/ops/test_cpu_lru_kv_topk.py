import os
from pathlib import Path

import numpy as np
import pytest


_CPP_BACKEND = None
_CPP_BACKEND_LOAD_ATTEMPTED = False


def _load_cpp_backend():
    global _CPP_BACKEND, _CPP_BACKEND_LOAD_ATTEMPTED
    if _CPP_BACKEND_LOAD_ATTEMPTED:
        if _CPP_BACKEND is None:
            pytest.skip("cpu_sparse_attn extension is unavailable")
        return _CPP_BACKEND

    _CPP_BACKEND_LOAD_ATTEMPTED = True
    pytest.importorskip("torch")
    torch_npu = pytest.importorskip("torch_npu")
    from torch.utils.cpp_extension import load

    src_path = (Path(__file__).resolve().parents[3] / "vllm_ascend" /
                "distributed" / "kv_transfer" / "kv_pool" /
                "ascend_store" / "cpu_sparse_attn.cpp")
    ascend_home = os.environ.get("ASCEND_HOME_PATH",
                                 "/usr/local/Ascend/ascend-toolkit/latest")
    npu_include_path = os.path.join(ascend_home, "include")
    npu_lib_path = os.path.join(ascend_home, "lib64")
    if not os.path.exists(npu_lib_path):
        npu_lib_path = os.path.join(ascend_home, "lib")
    torch_npu_path = os.path.dirname(torch_npu.__file__)
    torch_npu_include = os.path.join(torch_npu_path, "include")
    torch_npu_lib_path = os.path.join(torch_npu_path, "lib")
    os.environ.setdefault("CXX", "clang++")
    os.environ.setdefault("CC", "clang")

    try:
        _CPP_BACKEND = load(
            name="cpu_sparse_attn_lru_kv_topk_test",
            sources=[str(src_path)],
            extra_cflags=[
                "-O3",
                "-std=c++20",
                "-funroll-loops",
                "-fomit-frame-pointer",
                "-fopenmp",
                "-march=armv8.2-a+sve+fp16+bf16",
                "-fPIC",
                f"-I{npu_include_path}",
                f"-I{torch_npu_include}",
            ],
            extra_ldflags=[
                "-fopenmp",
                f"-L{npu_lib_path}",
                "-lascendcl",
                f"-L{torch_npu_lib_path}",
                "-ltorch_npu",
            ],
            verbose=False,
        )
    except Exception as exc:
        pytest.skip(f"cpu_sparse_attn extension is unavailable: {exc}")

    if not hasattr(_CPP_BACKEND, "lru_kv_topk"):
        pytest.skip("cpu_sparse_attn.lru_kv_topk is unavailable")
    return _CPP_BACKEND


def _reference_lru_slots(req_ids, last_req_ids, topk, slot_to_token,
                         lru_slots, capacity, max_token):
    num_reqs, topk_width = topk.shape
    current_slots = np.full((num_reqs, topk_width), -1, dtype=np.int32)
    load_token_indices = np.full((num_reqs, capacity), -1, dtype=np.int32)

    for row in range(num_reqs):
        if last_req_ids[row] != req_ids[row]:
            slot_to_token[row, :] = -1
            lru_slots[row, :] = np.arange(capacity, dtype=np.int32)
            last_req_ids[row] = req_ids[row]

        topk_pos = {}
        for pos in range(topk_width):
            token = int(topk[row, pos])
            if 0 <= token < max_token and token not in topk_pos:
                topk_pos[token] = pos

        hit_slots = []
        evictable_slots = []
        for order in range(capacity):
            slot = int(lru_slots[row, order])
            token = int(slot_to_token[row, slot])
            pos = topk_pos.get(token)
            if pos is None:
                evictable_slots.append(slot)
            else:
                current_slots[row, pos] = slot
                hit_slots.append(slot)

        miss_items = []
        for pos in range(topk_width):
            token = int(topk[row, pos])
            if 0 <= token < max_token and current_slots[row, pos] < 0:
                miss_items.append((pos, token))

        used_miss_slots = []
        for miss_rank, (pos, token) in enumerate(miss_items):
            if miss_rank >= len(evictable_slots):
                break
            slot = evictable_slots[miss_rank]
            slot_to_token[row, slot] = token
            current_slots[row, pos] = slot
            load_token_indices[row, slot] = token
            used_miss_slots.append(slot)

        stale_evictable_slots = evictable_slots[len(used_miss_slots):]
        new_order = stale_evictable_slots + used_miss_slots + hit_slots
        lru_slots[row, :] = np.array(new_order, dtype=np.int32)

    return (
        last_req_ids,
        slot_to_token,
        lru_slots,
        current_slots,
        load_token_indices,
    )


def _cpp_lru(req_ids, last_req_ids, topk, slot_to_token, lru_slots,
             capacity, max_token, requested_threads):
    torch = pytest.importorskip("torch")
    backend = _load_cpp_backend()
    workspace_threads = 4
    req_ids_tensor = torch.from_numpy(req_ids.copy())
    last_req_ids_tensor = torch.from_numpy(last_req_ids.copy())
    topk_tensor = torch.from_numpy(topk.copy())
    slot_to_token_tensor = torch.from_numpy(slot_to_token.copy())
    lru_slots_tensor = torch.from_numpy(lru_slots.copy())
    current_slots_tensor = torch.full(topk.shape, -1, dtype=torch.int32)
    load_token_indices_tensor = torch.full((topk.shape[0], capacity),
                                           -1,
                                           dtype=torch.int32)
    token_mark_workspace = torch.zeros((workspace_threads, max_token),
                                       dtype=torch.int32)
    token_pos_workspace = torch.full((workspace_threads, max_token),
                                     -1,
                                     dtype=torch.int32)
    hit_slot_workspace = torch.empty((workspace_threads, capacity),
                                     dtype=torch.int32)
    evictable_slot_workspace = torch.empty((workspace_threads, capacity),
                                           dtype=torch.int32)
    miss_token_workspace = torch.empty((workspace_threads, topk.shape[1]),
                                       dtype=torch.int32)
    miss_position_workspace = torch.empty((workspace_threads, topk.shape[1]),
                                          dtype=torch.int32)
    miss_slot_workspace = torch.empty((workspace_threads, topk.shape[1]),
                                      dtype=torch.int32)
    epochs = torch.zeros((workspace_threads, ), dtype=torch.int32)

    backend.lru_kv_topk(
        req_ids_tensor.data_ptr(),
        last_req_ids_tensor.data_ptr(),
        topk_tensor.data_ptr(),
        slot_to_token_tensor.data_ptr(),
        lru_slots_tensor.data_ptr(),
        current_slots_tensor.data_ptr(),
        load_token_indices_tensor.data_ptr(),
        token_mark_workspace.data_ptr(),
        token_pos_workspace.data_ptr(),
        hit_slot_workspace.data_ptr(),
        evictable_slot_workspace.data_ptr(),
        miss_token_workspace.data_ptr(),
        miss_position_workspace.data_ptr(),
        miss_slot_workspace.data_ptr(),
        epochs.data_ptr(),
        topk.shape[0],
        topk.shape[1],
        capacity,
        max_token,
        workspace_threads,
        requested_threads,
    )
    return (
        last_req_ids_tensor.numpy(),
        slot_to_token_tensor.numpy(),
        lru_slots_tensor.numpy(),
        current_slots_tensor.numpy(),
        load_token_indices_tensor.numpy(),
    )


@pytest.mark.parametrize("requested_threads", [1, 4])
def test_lru_kv_topk_matches_lru_slots_reference_with_hits_and_misses(
        requested_threads):
    req_ids = np.array([11, 12], dtype=np.int64)
    last_req_ids = np.array([11, 12], dtype=np.int64)
    topk = np.array([
        [7, 5, 22, 9, -1, -1],
        [30, 31, 32, 99, 33, -1],
    ],
                    dtype=np.int32)
    slot_to_token = np.array([
        [5, 8, 9, -1],
        [31, 40, -1, 41],
    ],
                             dtype=np.int32)
    lru_slots = np.array([
        [3, 1, 0, 2],
        [2, 0, 3, 1],
    ],
                         dtype=np.int32)
    capacity = 4
    max_token = 128

    expected = _reference_lru_slots(
        req_ids.copy(),
        last_req_ids.copy(),
        topk.copy(),
        slot_to_token.copy(),
        lru_slots.copy(),
        capacity,
        max_token,
    )
    actual = _cpp_lru(
        req_ids,
        last_req_ids,
        topk,
        slot_to_token,
        lru_slots,
        capacity,
        max_token,
        requested_threads,
    )

    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_lru_kv_topk_resets_lru_slots_when_req_id_changes():
    req_ids = np.array([77], dtype=np.int64)
    last_req_ids = np.array([55], dtype=np.int64)
    topk = np.array([[2, 4, 6, 8]], dtype=np.int32)
    slot_to_token = np.array([[10, 11, 12, 13]], dtype=np.int32)
    lru_slots = np.array([[3, 2, 1, 0]], dtype=np.int32)
    capacity = 4
    max_token = 32

    expected = _reference_lru_slots(
        req_ids.copy(),
        last_req_ids.copy(),
        topk.copy(),
        slot_to_token.copy(),
        lru_slots.copy(),
        capacity,
        max_token,
    )
    actual = _cpp_lru(
        req_ids,
        last_req_ids,
        topk,
        slot_to_token,
        lru_slots,
        capacity,
        max_token,
        requested_threads=1,
    )

    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_array_equal(actual_array, expected_array)
    np.testing.assert_array_equal(actual[2],
                                  np.array([[0, 1, 2, 3]], dtype=np.int32))


def test_lru_kv_topk_does_not_evict_later_topk_hit():
    req_ids = np.array([1], dtype=np.int64)
    last_req_ids = np.array([1], dtype=np.int64)
    topk = np.array([[99, 20]], dtype=np.int32)
    slot_to_token = np.array([[10, 20, 30, 40]], dtype=np.int32)
    lru_slots = np.array([[1, 2, 3, 0]], dtype=np.int32)
    capacity = 4
    max_token = 128

    expected = _reference_lru_slots(
        req_ids.copy(),
        last_req_ids.copy(),
        topk.copy(),
        slot_to_token.copy(),
        lru_slots.copy(),
        capacity,
        max_token,
    )
    actual = _cpp_lru(
        req_ids,
        last_req_ids,
        topk,
        slot_to_token,
        lru_slots,
        capacity,
        max_token,
        requested_threads=1,
    )

    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_array_equal(actual_array, expected_array)
    assert actual[3][0, 1] == 1
    assert actual[4][0, 1] == -1
