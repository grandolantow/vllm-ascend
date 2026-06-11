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


def _reference_lru(req_ids, last_req_ids, topk, slot_to_token, slot_last_used,
                   capacity, max_token, step_value):
    num_reqs, topk_width = topk.shape
    current_slots = np.full((num_reqs, topk_width), -1, dtype=np.int32)
    load_token_indices = np.full((num_reqs, capacity), -1, dtype=np.int32)

    for row in range(num_reqs):
        if last_req_ids[row] != req_ids[row]:
            slot_to_token[row, :] = -1
            slot_last_used[row, :] = -1
            last_req_ids[row] = req_ids[row]

        token_to_slot = {}
        for slot in range(capacity):
            token = int(slot_to_token[row, slot])
            if 0 <= token < max_token:
                token_to_slot[token] = slot

        protected_slots = set()
        miss_tokens = []
        for pos in range(topk_width):
            token = int(topk[row, pos])
            if token < 0 or token >= max_token:
                continue
            hit_slot = token_to_slot.get(token, -1)
            if hit_slot >= 0:
                current_slots[row, pos] = hit_slot
                protected_slots.add(hit_slot)
            else:
                miss_tokens.append((pos, token))

        candidate_slots = [
            slot for slot in range(capacity) if slot not in protected_slots
        ]
        candidate_slots.sort(
            key=lambda slot: (
                0 if slot_to_token[row, slot] < 0 else 1,
                int(slot_last_used[row, slot])
                if slot_last_used[row, slot] >= 0 else -1,
                slot,
            ))

        for miss_rank, (pos, token) in enumerate(miss_tokens):
            if miss_rank >= len(candidate_slots):
                break
            slot = candidate_slots[miss_rank]
            old_token = int(slot_to_token[row, slot])
            if 0 <= old_token < max_token:
                token_to_slot.pop(old_token, None)
            slot_to_token[row, slot] = token
            slot_last_used[row, slot] = step_value
            token_to_slot[token] = slot
            current_slots[row, pos] = slot
            load_token_indices[row, slot] = token

        for pos in range(topk_width):
            token = int(topk[row, pos])
            slot = int(current_slots[row, pos])
            if 0 <= token < max_token and slot >= 0:
                slot_last_used[row, slot] = step_value

    return (last_req_ids, slot_to_token, slot_last_used, current_slots,
            load_token_indices)


def _cpp_lru(req_ids, last_req_ids, topk, slot_to_token, slot_last_used,
             capacity, max_token, step_value, requested_threads):
    torch = pytest.importorskip("torch")
    backend = _load_cpp_backend()
    workspace_threads = 4
    req_ids_tensor = torch.from_numpy(req_ids.copy())
    last_req_ids_tensor = torch.from_numpy(last_req_ids.copy())
    topk_tensor = torch.from_numpy(topk.copy())
    slot_to_token_tensor = torch.from_numpy(slot_to_token.copy())
    slot_last_used_tensor = torch.from_numpy(slot_last_used.copy())
    current_slots_tensor = torch.full(topk.shape, -1, dtype=torch.int32)
    load_token_indices_tensor = torch.full((topk.shape[0], capacity),
                                           -1,
                                           dtype=torch.int32)
    token_mark_workspace = torch.zeros((workspace_threads, max_token),
                                       dtype=torch.int32)
    token_slot_workspace = torch.full((workspace_threads, max_token),
                                      -1,
                                      dtype=torch.int32)
    miss_token_workspace = torch.empty((workspace_threads, topk.shape[1]),
                                       dtype=torch.int32)
    miss_slot_workspace = torch.empty((workspace_threads, topk.shape[1]),
                                      dtype=torch.int32)
    epochs = torch.zeros((workspace_threads, ), dtype=torch.int32)

    backend.lru_kv_topk(
        req_ids_tensor.data_ptr(),
        last_req_ids_tensor.data_ptr(),
        topk_tensor.data_ptr(),
        slot_to_token_tensor.data_ptr(),
        slot_last_used_tensor.data_ptr(),
        current_slots_tensor.data_ptr(),
        load_token_indices_tensor.data_ptr(),
        token_mark_workspace.data_ptr(),
        token_slot_workspace.data_ptr(),
        miss_token_workspace.data_ptr(),
        miss_slot_workspace.data_ptr(),
        epochs.data_ptr(),
        topk.shape[0],
        topk.shape[1],
        capacity,
        max_token,
        workspace_threads,
        requested_threads,
        step_value,
    )
    return (
        last_req_ids_tensor.numpy(),
        slot_to_token_tensor.numpy(),
        slot_last_used_tensor.numpy(),
        current_slots_tensor.numpy(),
        load_token_indices_tensor.numpy(),
    )


@pytest.mark.parametrize("requested_threads", [1, 4])
def test_lru_kv_topk_matches_reference_with_hits_and_misses(
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
    slot_last_used = np.array([
        [3, 1, 9, -1],
        [2, 7, -1, 5],
    ],
                              dtype=np.int32)
    capacity = 4
    max_token = 128
    step_value = 10

    expected = _reference_lru(
        req_ids.copy(),
        last_req_ids.copy(),
        topk.copy(),
        slot_to_token.copy(),
        slot_last_used.copy(),
        capacity,
        max_token,
        step_value,
    )
    actual = _cpp_lru(
        req_ids,
        last_req_ids,
        topk,
        slot_to_token,
        slot_last_used,
        capacity,
        max_token,
        step_value,
        requested_threads,
    )

    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_array_equal(actual_array, expected_array)


def test_lru_kv_topk_resets_state_when_req_id_changes():
    req_ids = np.array([77], dtype=np.int64)
    last_req_ids = np.array([55], dtype=np.int64)
    topk = np.array([[2, 4, 6, 8]], dtype=np.int32)
    slot_to_token = np.array([[10, 11, 12, 13]], dtype=np.int32)
    slot_last_used = np.array([[1, 2, 3, 4]], dtype=np.int32)
    capacity = 4
    max_token = 32
    step_value = 5

    expected = _reference_lru(
        req_ids.copy(),
        last_req_ids.copy(),
        topk.copy(),
        slot_to_token.copy(),
        slot_last_used.copy(),
        capacity,
        max_token,
        step_value,
    )
    actual = _cpp_lru(
        req_ids,
        last_req_ids,
        topk,
        slot_to_token,
        slot_last_used,
        capacity,
        max_token,
        step_value,
        requested_threads=1,
    )

    for expected_array, actual_array in zip(expected, actual):
        np.testing.assert_array_equal(actual_array, expected_array)
