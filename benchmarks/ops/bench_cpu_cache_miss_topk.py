import argparse
import csv
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from numba import get_num_threads, get_thread_id, njit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    get_num_threads = None
    get_thread_id = None
    njit = None
    prange = range


OLD_FLAG = 1
NEW_FLAG = 2
BOTH_FLAG = 3
DEFAULT_NUM_REQS = (1, 2, 4, 16, 32, 128)
DEFAULT_SEQ_LENS = (32768, 131072)
DEFAULT_HIT_RATES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
DEFAULT_TOPK = 2048
DEFAULT_REPEAT = 20
_CPP_BACKEND = None
_CPP_BACKEND_LOAD_ATTEMPTED = False
_CPP_BACKEND_ERROR = None


@dataclass(frozen=True)
class BenchmarkCase:
    num_reqs: int
    seq_len: int
    topk: int
    hit_rate: float


def _parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",")
                 if item.strip())


def _parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",")
                 if item.strip())


def _parse_cpp_thread_list(value: str) -> tuple[int, ...]:
    parsed: list[int] = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parsed.append(0 if item == "auto" else int(item))
    return tuple(parsed)


def _next_power_of_two(value: int) -> int:
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def generate_case(
    num_reqs: int,
    seq_len: int,
    topk: int,
    hit_rate: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if topk > seq_len:
        raise ValueError(f"topk={topk} must be <= seq_len={seq_len}")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"hit_rate={hit_rate} must be in [0, 1]")

    rng = np.random.default_rng(seed)
    overlap_count = int(round(topk * hit_rate))
    miss_count = topk - overlap_count
    req_ids = np.arange(num_reqs, dtype=np.int64)
    old = np.empty((num_reqs, topk), dtype=np.int64)
    new = np.empty((num_reqs, topk), dtype=np.int32)

    for row in range(num_reqs):
        old_tokens = rng.choice(seq_len, size=topk, replace=False)
        overlap_tokens = rng.choice(old_tokens,
                                    size=overlap_count,
                                    replace=False)
        old_mask = np.ones(seq_len, dtype=np.bool_)
        old_mask[old_tokens] = False
        candidates = np.flatnonzero(old_mask)
        miss_tokens = rng.choice(candidates, size=miss_count, replace=False)
        new_tokens = np.concatenate((overlap_tokens, miss_tokens))
        rng.shuffle(new_tokens)

        old[row] = old_tokens.astype(np.int64, copy=False)
        new[row] = new_tokens.astype(np.int32, copy=False)

    return (
        np.ascontiguousarray(req_ids),
        np.ascontiguousarray(req_ids.copy()),
        np.ascontiguousarray(old),
        np.ascontiguousarray(new),
    )


def reference_update_topk_indices(
    req_ids_tensor: np.ndarray,
    last_req_ids_tensor: np.ndarray,
    topk_indices_old: np.ndarray,
    topk_indices_new: np.ndarray,
) -> np.ndarray:
    bs, topk = topk_indices_old.shape

    for row in range(bs):
        req_id = int(req_ids_tensor[row])
        last_req_id = int(last_req_ids_tensor[row])
        old_row = topk_indices_old[row]
        new_row = topk_indices_new[row]

        if last_req_id != req_id:
            seen: set[int] = set()
            write_pos = 0
            out = np.full(topk, -1, dtype=np.int32)
            old_row.fill(-1)
            for slot in range(topk):
                value = int(new_row[slot])
                if value >= 0 and value not in seen:
                    seen.add(value)
                    old_row[write_pos] = value
                    out[write_pos] = value
                    write_pos += 1
                    if write_pos >= topk:
                        break
            new_row[:] = out
            last_req_ids_tensor[row] = req_id
            continue

        old_values = {int(value) for value in old_row if value >= 0}
        new_values = {int(value) for value in new_row if value >= 0}
        seen_miss: set[int] = set()
        miss_values: list[int] = []

        for value in new_row:
            value_int = int(value)
            if value_int >= 0 and value_int not in old_values:
                if value_int not in seen_miss:
                    seen_miss.add(value_int)
                    miss_values.append(value_int)

        miss_pos = 0
        new_row.fill(-1)

        for slot in range(topk):
            old_value = int(old_row[slot])
            if old_value >= 0 and old_value not in new_values:
                if miss_pos < len(miss_values):
                    replacement = miss_values[miss_pos]
                    miss_pos += 1
                    old_row[slot] = replacement
                    new_row[slot] = replacement
                else:
                    old_row[slot] = -1

        if miss_pos < len(miss_values):
            for slot in range(topk):
                if old_row[slot] == -1:
                    replacement = miss_values[miss_pos]
                    miss_pos += 1
                    old_row[slot] = replacement
                    new_row[slot] = replacement
                    if miss_pos >= len(miss_values):
                        break

        last_req_ids_tensor[row] = req_id

    return topk_indices_new


def numpy_isin_update_topk_indices(
    req_ids_tensor: np.ndarray,
    last_req_ids_tensor: np.ndarray,
    topk_indices_old: np.ndarray,
    topk_indices_new: np.ndarray,
) -> np.ndarray:
    bs, topk = topk_indices_old.shape

    for row in range(bs):
        req_id = int(req_ids_tensor[row])
        old_row = topk_indices_old[row]
        new_row = topk_indices_new[row]

        if int(last_req_ids_tensor[row]) != req_id:
            seen: set[int] = set()
            write_pos = 0
            output = np.full(topk, -1, dtype=np.int32)
            old_row.fill(-1)
            for slot in range(topk):
                value_int = int(new_row[slot])
                if value_int >= 0 and value_int not in seen:
                    seen.add(value_int)
                    old_row[write_pos] = value_int
                    output[write_pos] = value_int
                    write_pos += 1
                    if write_pos >= topk:
                        break
            new_row[:] = output
            last_req_ids_tensor[row] = req_id
            continue

        old_valid_raw = old_row[old_row >= 0]
        new_valid = new_row[new_row >= 0]
        miss_mask = np.logical_not(np.isin(new_valid, old_valid_raw))
        raw_miss_values = new_valid[miss_mask]
        seen_miss: set[int] = set()
        miss_values = []

        for value in raw_miss_values:
            value_int = int(value)
            if value_int not in seen_miss:
                seen_miss.add(value_int)
                miss_values.append(value_int)

        miss_values_array = np.asarray(miss_values, dtype=np.int32)
        miss_pos = 0
        output = np.full(topk, -1, dtype=np.int32)
        keep_mask = np.isin(old_row, new_valid)

        for slot in range(topk):
            if old_row[slot] >= 0 and not keep_mask[slot]:
                if miss_pos < miss_values_array.size:
                    replacement = miss_values_array[miss_pos]
                    miss_pos += 1
                    old_row[slot] = replacement
                    output[slot] = replacement
                else:
                    old_row[slot] = -1

        if miss_pos < miss_values_array.size:
            for slot in range(topk):
                if old_row[slot] == -1:
                    replacement = miss_values_array[miss_pos]
                    miss_pos += 1
                    old_row[slot] = replacement
                    output[slot] = replacement
                    if miss_pos >= miss_values_array.size:
                        break

        new_row[:] = output
        last_req_ids_tensor[row] = req_id

    return topk_indices_new


if NUMBA_AVAILABLE:

    @njit(inline="always")
    def _has_old(mark, token, base):
        marker = mark[token]
        return marker == base + OLD_FLAG or marker == base + BOTH_FLAG

    @njit(inline="always")
    def _has_new(mark, token, base):
        marker = mark[token]
        return marker == base + NEW_FLAG or marker == base + BOTH_FLAG

    @njit(inline="always")
    def _set_old(mark, token, base):
        marker = mark[token]
        if marker == base + NEW_FLAG:
            mark[token] = base + BOTH_FLAG
        elif marker != base + OLD_FLAG and marker != base + BOTH_FLAG:
            mark[token] = base + OLD_FLAG

    @njit(inline="always")
    def _set_new(mark, token, base):
        marker = mark[token]
        if marker == base + OLD_FLAG:
            mark[token] = base + BOTH_FLAG
        elif marker != base + NEW_FLAG and marker != base + BOTH_FLAG:
            mark[token] = base + NEW_FLAG

    @njit(nogil=True, parallel=False, boundscheck=False)
    def update_topk_cache_miss_inplace_numba(
        req_ids_tensor,
        last_req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        mark_workspace,
        miss_workspace,
        epochs,
    ):
        bs, topk = topk_indices_old.shape
        max_token = mark_workspace.shape[1]

        for row in prange(bs):
            tid = 0
            base = epochs[tid] + 4

            if base >= 2_147_483_000:
                for token in range(max_token):
                    mark_workspace[tid, token] = 0
                base = 4

            epochs[tid] = base
            mark = mark_workspace[tid]
            miss_buffer = miss_workspace[tid]

            req_id = req_ids_tensor[row]
            if last_req_ids_tensor[row] != req_id:
                write_pos = 0
                for slot in range(topk):
                    topk_indices_old[row, slot] = -1

                for slot in range(topk):
                    value = topk_indices_new[row, slot]
                    topk_indices_new[row, slot] = -1
                    if (value >= 0 and value < max_token
                            and not _has_old(mark, value, base)):
                        _set_old(mark, value, base)
                        topk_indices_old[row, write_pos] = value
                        topk_indices_new[row, write_pos] = value
                        write_pos += 1
                        if write_pos >= topk:
                            break

                last_req_ids_tensor[row] = req_id
                continue

            for slot in range(topk):
                value = topk_indices_old[row, slot]
                if value >= 0 and value < max_token:
                    _set_old(mark, value, base)

            miss_count = 0
            for slot in range(topk):
                value = topk_indices_new[row, slot]
                if value >= 0 and value < max_token:
                    _set_new(mark, value, base)
                    if not _has_old(mark, value, base):
                        miss_buffer[miss_count] = value
                        miss_count += 1
                        _set_old(mark, value, base)

            miss_pos = 0
            for slot in range(topk):
                topk_indices_new[row, slot] = -1
                old_value = topk_indices_old[row, slot]
                old_raw = old_value
                if (old_raw >= 0 and old_raw < max_token
                        and not _has_new(mark, old_raw, base)):
                    if miss_pos < miss_count:
                        replacement = miss_buffer[miss_pos]
                        miss_pos += 1
                        topk_indices_old[row, slot] = replacement
                        topk_indices_new[row, slot] = replacement
                    else:
                        topk_indices_old[row, slot] = -1

            if miss_pos < miss_count:
                for slot in range(topk):
                    if topk_indices_old[row, slot] == -1:
                        replacement = miss_buffer[miss_pos]
                        miss_pos += 1
                        topk_indices_old[row, slot] = replacement
                        topk_indices_new[row, slot] = replacement
                        if miss_pos >= miss_count:
                            break

            last_req_ids_tensor[row] = req_id

        return topk_indices_new


def make_topk_workspace(
    topk: int = DEFAULT_TOPK,
    max_token: int = 131072,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if NUMBA_AVAILABLE is False:
        raise RuntimeError("numba is required for make_topk_workspace")

    num_threads = get_num_threads()
    mark_workspace = np.zeros((num_threads, max_token), dtype=np.int32)
    miss_workspace = np.empty((num_threads, topk), dtype=np.int32)
    epochs = np.zeros((num_threads,), dtype=np.int32)
    return mark_workspace, miss_workspace, epochs


def update_topk_indices_cpu(
    req_ids_tensor: np.ndarray,
    last_req_ids_tensor: np.ndarray,
    topk_indices_old: np.ndarray,
    topk_indices_new: np.ndarray,
    workspace: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    if NUMBA_AVAILABLE is False:
        raise RuntimeError("numba is required for update_topk_indices_cpu")
    if req_ids_tensor.dtype != np.int64:
        raise TypeError("req_ids_tensor must be int64")
    if last_req_ids_tensor.dtype != np.int64:
        raise TypeError("last_req_ids_tensor must be int64")
    if topk_indices_old.dtype != np.int64:
        raise TypeError("topk_indices_old must be int64")
    if topk_indices_new.dtype != np.int32:
        raise TypeError("topk_indices_new must be int32")
    if topk_indices_old.shape != topk_indices_new.shape:
        raise ValueError("old/new shape mismatch")
    if req_ids_tensor.shape[0] != topk_indices_old.shape[0]:
        raise ValueError("req_ids_tensor length must match batch size")
    if last_req_ids_tensor.shape[0] != topk_indices_old.shape[0]:
        raise ValueError("last_req_ids_tensor length must match batch size")
    if not req_ids_tensor.flags.c_contiguous:
        raise ValueError("req_ids_tensor must be C-contiguous")
    if not last_req_ids_tensor.flags.c_contiguous:
        raise ValueError("last_req_ids_tensor must be C-contiguous")
    if not topk_indices_old.flags.c_contiguous:
        raise ValueError("topk_indices_old must be C-contiguous")
    if not topk_indices_new.flags.c_contiguous:
        raise ValueError("topk_indices_new must be C-contiguous")

    mark_workspace, miss_workspace, epochs = workspace
    return update_topk_cache_miss_inplace_numba(
        req_ids_tensor,
        last_req_ids_tensor,
        topk_indices_old,
        topk_indices_new,
        mark_workspace,
        miss_workspace,
        epochs,
    )


def _load_cpp_backend():
    global _CPP_BACKEND, _CPP_BACKEND_LOAD_ATTEMPTED, _CPP_BACKEND_ERROR
    if _CPP_BACKEND_LOAD_ATTEMPTED:
        return _CPP_BACKEND

    _CPP_BACKEND_LOAD_ATTEMPTED = True
    try:
        import torch_npu
        from torch.utils.cpp_extension import load
    except Exception as exc:
        _CPP_BACKEND_ERROR = exc
        return None

    repo_root = Path(__file__).resolve().parents[2]
    src_path = (repo_root / "vllm_ascend" / "distributed" /
                "kv_transfer" / "kv_pool" / "ascend_store" /
                "cpu_sparse_attn.cpp")
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
            name="cpu_sparse_attn_cache_miss_topk_bench",
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
        _CPP_BACKEND_ERROR = exc
        _CPP_BACKEND = None
        return None

    if not hasattr(_CPP_BACKEND, "cache_miss_topk"):
        _CPP_BACKEND_ERROR = RuntimeError(
            "cpu_sparse_attn.cache_miss_topk is unavailable")
        _CPP_BACKEND = None
    return _CPP_BACKEND


def make_cpp_topk_workspace(topk: int,
                            max_token: int,
                            workspace_threads: int = 4):
    import torch

    backend = _load_cpp_backend()
    if backend is None:
        raise RuntimeError(f"C++ backend unavailable: {_CPP_BACKEND_ERROR}")
    mark_workspace = torch.zeros([workspace_threads, max_token],
                                 dtype=torch.int32)
    miss_workspace = torch.empty([workspace_threads, topk], dtype=torch.int32)
    epochs = torch.zeros([workspace_threads], dtype=torch.int32)
    return (backend, mark_workspace, miss_workspace, epochs, topk, max_token,
            workspace_threads)


def make_cpp_hash_topk_workspace(topk: int,
                                 max_token: int,
                                 workspace_threads: int = 4):
    import torch

    backend = _load_cpp_backend()
    if backend is None:
        raise RuntimeError(f"C++ backend unavailable: {_CPP_BACKEND_ERROR}")
    if not hasattr(backend, "cache_miss_topk_hash"):
        raise RuntimeError("cpu_sparse_attn.cache_miss_topk_hash is unavailable")
    hash_capacity = _next_power_of_two(max(8192, topk * 4))
    hash_keys = torch.empty([workspace_threads, hash_capacity],
                            dtype=torch.int32)
    hash_marks = torch.zeros([workspace_threads, hash_capacity],
                             dtype=torch.int32)
    miss_workspace = torch.empty([workspace_threads, topk], dtype=torch.int32)
    epochs = torch.zeros([workspace_threads], dtype=torch.int32)
    return (backend, hash_keys, hash_marks, miss_workspace, epochs, topk,
            max_token, hash_capacity, workspace_threads)


def update_topk_indices_cpp(
    req_ids_tensor: np.ndarray,
    last_req_ids_tensor: np.ndarray,
    topk_indices_old: np.ndarray,
    topk_indices_new: np.ndarray,
    workspace,
    requested_threads: int = 0,
) -> np.ndarray:
    import torch

    (
        backend,
        mark_workspace,
        miss_workspace,
        epochs,
        topk,
        max_token,
        workspace_threads,
    ) = workspace
    if topk_indices_old.shape[1] != topk:
        raise ValueError("workspace topk mismatch")

    req_ids_t = torch.from_numpy(req_ids_tensor)
    last_req_ids_t = torch.from_numpy(last_req_ids_tensor)
    old_t = torch.from_numpy(topk_indices_old)
    new_t = torch.from_numpy(topk_indices_new)
    backend.cache_miss_topk(
        req_ids_t.data_ptr(),
        last_req_ids_t.data_ptr(),
        old_t.data_ptr(),
        new_t.data_ptr(),
        mark_workspace.data_ptr(),
        miss_workspace.data_ptr(),
        epochs.data_ptr(),
        req_ids_tensor.shape[0],
        topk,
        max_token,
        workspace_threads,
        requested_threads,
    )
    return topk_indices_new


def update_topk_indices_cpp_hash(
    req_ids_tensor: np.ndarray,
    last_req_ids_tensor: np.ndarray,
    topk_indices_old: np.ndarray,
    topk_indices_new: np.ndarray,
    workspace,
    requested_threads: int = 0,
) -> np.ndarray:
    import torch

    (
        backend,
        hash_keys,
        hash_marks,
        miss_workspace,
        epochs,
        topk,
        max_token,
        hash_capacity,
        workspace_threads,
    ) = workspace
    if topk_indices_old.shape[1] != topk:
        raise ValueError("workspace topk mismatch")

    req_ids_t = torch.from_numpy(req_ids_tensor)
    last_req_ids_t = torch.from_numpy(last_req_ids_tensor)
    old_t = torch.from_numpy(topk_indices_old)
    new_t = torch.from_numpy(topk_indices_new)
    backend.cache_miss_topk_hash(
        req_ids_t.data_ptr(),
        last_req_ids_t.data_ptr(),
        old_t.data_ptr(),
        new_t.data_ptr(),
        hash_keys.data_ptr(),
        hash_marks.data_ptr(),
        miss_workspace.data_ptr(),
        epochs.data_ptr(),
        req_ids_tensor.shape[0],
        topk,
        max_token,
        hash_capacity,
        workspace_threads,
        requested_threads,
    )
    return topk_indices_new


def _time_impl(
    name: str,
    func: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    last_req_ids_base: np.ndarray,
    old_base: np.ndarray,
    new_base: np.ndarray,
    repeat: int,
    expected_last_req_ids: np.ndarray,
    expected_old: np.ndarray,
    expected_out: np.ndarray,
) -> dict[str, float | str]:
    best_ms = float("inf")
    total_ms = 0.0

    for _ in range(repeat):
        last_req_ids = last_req_ids_base.copy()
        old = old_base.copy()
        new = new_base.copy()
        start = time.perf_counter()
        out = func(last_req_ids, old, new)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        best_ms = min(best_ms, elapsed_ms)
        total_ms += elapsed_ms

    if not np.array_equal(out, expected_out):
        raise AssertionError(f"{name} output mismatch")
    if not np.array_equal(old, expected_old):
        raise AssertionError(f"{name} updated old mismatch")
    if not np.array_equal(last_req_ids, expected_last_req_ids):
        raise AssertionError(f"{name} updated last_req_ids mismatch")

    return {
        "impl": name,
        "avg_ms": total_ms / repeat,
        "best_ms": best_ms,
    }


def _iter_cases(
    num_reqs_values: tuple[int, ...],
    seq_lens: tuple[int, ...],
    topk: int,
    hit_rates: tuple[float, ...],
) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(num_reqs=num_reqs,
                      seq_len=seq_len,
                      topk=topk,
                      hit_rate=hit_rate)
        for num_reqs in num_reqs_values
        for seq_len in seq_lens
        for hit_rate in hit_rates
    ]


def run_benchmark(
    num_reqs_values: tuple[int, ...] = DEFAULT_NUM_REQS,
    seq_lens: tuple[int, ...] = DEFAULT_SEQ_LENS,
    topk: int = DEFAULT_TOPK,
    hit_rates: tuple[float, ...] = DEFAULT_HIT_RATES,
    repeat: int = DEFAULT_REPEAT,
    seed: int = 0,
    csv_path: Path | None = None,
    include_numpy: bool = True,
    backends: tuple[str, ...] = ("numba", "numpy"),
    cpp_threads: tuple[int, ...] = (0,),
) -> list[dict[str, float | int | str]]:
    results: list[dict[str, float | int | str]] = []
    writer = None
    csv_file = None

    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = csv_path.open("w", newline="")
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "impl",
                "num_reqs",
                "seq_len",
                "topk",
                "hit_rate",
                "avg_ms",
                "best_ms",
            ],
        )
        writer.writeheader()

    try:
        for case_id, case in enumerate(
                _iter_cases(num_reqs_values, seq_lens, topk, hit_rates)):
            req_ids, last_req_ids_base, old_base, new_base = generate_case(
                num_reqs=case.num_reqs,
                seq_len=case.seq_len,
                topk=case.topk,
                hit_rate=case.hit_rate,
                seed=seed + case_id,
            )
            expected_last_req_ids = last_req_ids_base.copy()
            expected_old = old_base.copy()
            expected_new = new_base.copy()
            expected_out = reference_update_topk_indices(
                req_ids,
                expected_last_req_ids,
                expected_old,
                expected_new,
            )

            impls: list[tuple[str, Callable[[np.ndarray, np.ndarray, np.ndarray],
                                            np.ndarray]]] = []
            requested_backends = set(backends)

            if "cpp" in requested_backends:
                try:
                    max_cpp_threads = max((thread for thread in cpp_threads
                                           if thread > 0),
                                          default=4)
                    workspace_threads = max(64, max_cpp_threads)
                    cpp_workspace = make_cpp_topk_workspace(
                        topk=case.topk,
                        max_token=case.seq_len,
                        workspace_threads=workspace_threads,
                    )
                    for requested_threads in cpp_threads:
                        warm_last_req_ids = last_req_ids_base.copy()
                        warm_old = old_base.copy()
                        warm_new = new_base.copy()
                        update_topk_indices_cpp(
                            req_ids,
                            warm_last_req_ids,
                            warm_old,
                            warm_new,
                            cpp_workspace,
                            requested_threads=requested_threads,
                        )

                        def cpp_impl(last_req_ids,
                                     old,
                                     new,
                                     req_ids=req_ids,
                                     workspace=cpp_workspace,
                                     requested_threads=requested_threads):
                            return update_topk_indices_cpp(
                                req_ids,
                                last_req_ids,
                                old,
                                new,
                                workspace,
                                requested_threads=requested_threads,
                            )

                        thread_name = (
                            "auto" if requested_threads <= 0 else
                            str(requested_threads))
                        impls.append((f"cpp_omp_t{thread_name}", cpp_impl))
                except Exception as exc:
                    print(
                        f"SKIP impl=cpp reason={exc}",
                        flush=True,
                    )

            if "cpp_hash" in requested_backends:
                try:
                    max_cpp_threads = max((thread for thread in cpp_threads
                                           if thread > 0),
                                          default=8)
                    workspace_threads = max(64, max_cpp_threads)
                    cpp_hash_workspace = make_cpp_hash_topk_workspace(
                        topk=case.topk,
                        max_token=case.seq_len,
                        workspace_threads=workspace_threads,
                    )
                    for requested_threads in cpp_threads:
                        warm_last_req_ids = last_req_ids_base.copy()
                        warm_old = old_base.copy()
                        warm_new = new_base.copy()
                        update_topk_indices_cpp_hash(
                            req_ids,
                            warm_last_req_ids,
                            warm_old,
                            warm_new,
                            cpp_hash_workspace,
                            requested_threads=requested_threads,
                        )

                        def cpp_hash_impl(
                                last_req_ids,
                                old,
                                new,
                                req_ids=req_ids,
                                workspace=cpp_hash_workspace,
                                requested_threads=requested_threads):
                            return update_topk_indices_cpp_hash(
                                req_ids,
                                last_req_ids,
                                old,
                                new,
                                workspace,
                                requested_threads=requested_threads,
                            )

                        thread_name = (
                            "auto" if requested_threads <= 0 else
                            str(requested_threads))
                        impls.append((f"cpp_hash_omp_t{thread_name}",
                                      cpp_hash_impl))
                except Exception as exc:
                    print(
                        f"SKIP impl=cpp_hash reason={exc}",
                        flush=True,
                    )

            if "numba" in requested_backends and NUMBA_AVAILABLE:
                workspace = make_topk_workspace(topk=case.topk,
                                                max_token=case.seq_len)
                warm_last_req_ids = last_req_ids_base.copy()
                warm_old = old_base.copy()
                warm_new = new_base.copy()
                update_topk_indices_cpu(req_ids, warm_last_req_ids,
                                        warm_old, warm_new, workspace)

                def numba_impl(last_req_ids, old, new, req_ids=req_ids,
                               workspace=workspace):
                    return update_topk_indices_cpu(req_ids, last_req_ids,
                                                   old, new, workspace)

                impls.append(("numba_stamp", numba_impl))

            if include_numpy and "numpy" in requested_backends:
                def numpy_impl(last_req_ids, old, new, req_ids=req_ids):
                    return numpy_isin_update_topk_indices(
                        req_ids, last_req_ids, old, new)

                impls.append(("numpy_isin", numpy_impl))

            for name, func in impls:
                measured = _time_impl(
                    name=name,
                    func=func,
                    last_req_ids_base=last_req_ids_base,
                    old_base=old_base,
                    new_base=new_base,
                    repeat=repeat,
                    expected_last_req_ids=expected_last_req_ids,
                    expected_old=expected_old,
                    expected_out=expected_out,
                )
                row = {
                    "impl": measured["impl"],
                    "num_reqs": case.num_reqs,
                    "seq_len": case.seq_len,
                    "topk": case.topk,
                    "hit_rate": case.hit_rate,
                    "avg_ms": measured["avg_ms"],
                    "best_ms": measured["best_ms"],
                }
                results.append(row)
                print(
                    f"RESULT impl={row['impl']} num_reqs={row['num_reqs']} "
                    f"seq_len={row['seq_len']} topk={row['topk']} "
                    f"hit_rate={row['hit_rate']:.1f} "
                    f"avg_ms={row['avg_ms']:.4f} "
                    f"best_ms={row['best_ms']:.4f}",
                    flush=True,
                )
                if writer is not None:
                    writer.writerow(row)
                    csv_file.flush()
    finally:
        if csv_file is not None:
            csv_file.close()

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPU cache-miss topk validation benchmark")
    parser.add_argument("--num-reqs",
                        default=",".join(str(v) for v in DEFAULT_NUM_REQS))
    parser.add_argument("--seq-lens",
                        default=",".join(str(v) for v in DEFAULT_SEQ_LENS))
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--hit-rates",
                        default=",".join(str(v) for v in DEFAULT_HIT_RATES))
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv",
                        type=Path,
                        default=Path("benchmarks/ops/"
                                     "cpu_cache_miss_topk_results.csv"))
    parser.add_argument("--skip-numpy",
                        action="store_true",
                        help="skip the slower NumPy baseline in full sweeps")
    parser.add_argument("--backends",
                        default="numba,numpy",
                        help=("comma-separated backends: "
                              "cpp,cpp_hash,numba,numpy"))
    parser.add_argument("--cpp-threads",
                        default="auto",
                        help=("comma-separated C++ threads: "
                              "auto,1,2,4,8,16,32,64"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        num_reqs_values=_parse_int_list(args.num_reqs),
        seq_lens=_parse_int_list(args.seq_lens),
        topk=args.topk,
        hit_rates=_parse_float_list(args.hit_rates),
        repeat=args.repeat,
        seed=args.seed,
        csv_path=args.csv,
        include_numpy=not args.skip_numpy,
        backends=tuple(item.strip() for item in args.backends.split(",")
                       if item.strip()),
        cpp_threads=_parse_cpp_thread_list(args.cpp_threads),
    )


if __name__ == "__main__":
    main()
