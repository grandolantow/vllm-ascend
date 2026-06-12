import argparse
import csv
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEFAULT_NUM_REQS = (1, 2, 4, 16, 32, 128)
DEFAULT_SEQ_LENS = (32768, 131072)
DEFAULT_HIT_RATES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0)
DEFAULT_TOPK = 2048
DEFAULT_CAPACITY = 2048
DEFAULT_REPEAT = 20
DEFAULT_WARMUP = 3
DEFAULT_CPP_THREADS = (0, 1, 2, 4, 8, 16, 32, 64)

_CPP_BACKEND = None
_CPP_BACKEND_LOAD_ATTEMPTED = False
_CPP_BACKEND_ERROR = None


@dataclass(frozen=True)
class BenchmarkCase:
    num_reqs: int
    seq_len: int
    topk: int
    capacity: int
    hit_rate: float


@dataclass
class GeneratedCase:
    req_ids: np.ndarray
    last_req_ids: np.ndarray
    topk_indices: np.ndarray
    slot_to_token: np.ndarray
    lru_slots: np.ndarray
    hit_count: int
    miss_count: int


@dataclass
class ReferenceResult:
    last_req_ids: np.ndarray
    slot_to_token: np.ndarray
    lru_slots: np.ndarray
    current_slots: np.ndarray
    load_token_indices: np.ndarray


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


def _thread_name(requested_threads: int) -> str:
    return "auto" if requested_threads <= 0 else str(requested_threads)


def generate_case(
    num_reqs: int,
    seq_len: int,
    topk: int,
    capacity: int,
    hit_rate: float,
    seed: int,
) -> GeneratedCase:
    if num_reqs <= 0:
        raise ValueError("num_reqs must be positive")
    if topk <= 0:
        raise ValueError("topk must be positive")
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if not 0.0 <= hit_rate <= 1.0:
        raise ValueError(f"hit_rate={hit_rate} must be in [0, 1]")

    hit_count = int(round(topk * hit_rate))
    miss_count = topk - hit_count
    if hit_count > capacity:
        raise ValueError(
            f"hit_count={hit_count} must be <= capacity={capacity}")
    if seq_len < capacity + miss_count:
        raise ValueError(
            f"seq_len={seq_len} must be >= capacity + miss_count "
            f"({capacity + miss_count})")

    rng = np.random.default_rng(seed)
    req_ids = np.arange(num_reqs, dtype=np.int64)
    last_req_ids = req_ids.copy()
    topk_indices = np.empty((num_reqs, topk), dtype=np.int32)
    slot_to_token = np.empty((num_reqs, capacity), dtype=np.int32)
    lru_slots = np.empty((num_reqs, capacity), dtype=np.int32)

    for row in range(num_reqs):
        resident_tokens = rng.choice(seq_len, size=capacity, replace=False)
        slot_to_token[row] = resident_tokens.astype(np.int32, copy=False)
        lru_slots[row] = rng.permutation(capacity).astype(np.int32,
                                                          copy=False)

        hit_tokens = rng.choice(resident_tokens,
                                size=hit_count,
                                replace=False)
        resident_mask = np.ones(seq_len, dtype=np.bool_)
        resident_mask[resident_tokens] = False
        miss_candidates = np.flatnonzero(resident_mask)
        miss_tokens = rng.choice(miss_candidates,
                                 size=miss_count,
                                 replace=False)
        topk_tokens = np.concatenate((hit_tokens, miss_tokens))
        rng.shuffle(topk_tokens)
        topk_indices[row] = topk_tokens.astype(np.int32, copy=False)

    return GeneratedCase(
        req_ids=np.ascontiguousarray(req_ids),
        last_req_ids=np.ascontiguousarray(last_req_ids),
        topk_indices=np.ascontiguousarray(topk_indices),
        slot_to_token=np.ascontiguousarray(slot_to_token),
        lru_slots=np.ascontiguousarray(lru_slots),
        hit_count=hit_count,
        miss_count=miss_count,
    )


def reference_lru_slots(
    req_ids: np.ndarray,
    last_req_ids: np.ndarray,
    topk_indices: np.ndarray,
    slot_to_token: np.ndarray,
    lru_slots: np.ndarray,
    capacity: int,
    max_token: int,
) -> ReferenceResult:
    num_reqs, topk = topk_indices.shape
    current_slots = np.full((num_reqs, topk), -1, dtype=np.int32)
    load_token_indices = np.full((num_reqs, capacity), -1, dtype=np.int32)

    for row in range(num_reqs):
        if last_req_ids[row] != req_ids[row]:
            slot_to_token[row, :] = -1
            lru_slots[row, :] = np.arange(capacity, dtype=np.int32)
            last_req_ids[row] = req_ids[row]

        topk_pos: dict[int, int] = {}
        for pos in range(topk):
            token = int(topk_indices[row, pos])
            if 0 <= token < max_token and token not in topk_pos:
                topk_pos[token] = pos

        hit_slots: list[int] = []
        evictable_slots: list[int] = []
        for order in range(capacity):
            slot = int(lru_slots[row, order])
            token = int(slot_to_token[row, slot])
            pos = topk_pos.get(token)
            if pos is None:
                evictable_slots.append(slot)
            else:
                current_slots[row, pos] = slot
                hit_slots.append(slot)

        miss_items: list[tuple[int, int]] = []
        for pos in range(topk):
            token = int(topk_indices[row, pos])
            if 0 <= token < max_token and current_slots[row, pos] < 0:
                miss_items.append((pos, token))

        used_miss_slots: list[int] = []
        for miss_rank, (pos, token) in enumerate(miss_items):
            if miss_rank >= len(evictable_slots):
                break
            slot = evictable_slots[miss_rank]
            slot_to_token[row, slot] = token
            current_slots[row, pos] = slot
            load_token_indices[row, slot] = token
            used_miss_slots.append(slot)

        stale_evictable_slots = evictable_slots[len(used_miss_slots):]
        lru_slots[row, :] = np.array(
            stale_evictable_slots + used_miss_slots + hit_slots,
            dtype=np.int32,
        )

    return ReferenceResult(
        last_req_ids=last_req_ids,
        slot_to_token=slot_to_token,
        lru_slots=lru_slots,
        current_slots=current_slots,
        load_token_indices=load_token_indices,
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
            name="cpu_sparse_attn_lru_kv_topk_bench",
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

    if not hasattr(_CPP_BACKEND, "lru_kv_topk"):
        _CPP_BACKEND_ERROR = RuntimeError(
            "cpu_sparse_attn.lru_kv_topk is unavailable")
        _CPP_BACKEND = None
    return _CPP_BACKEND


def make_cpp_lru_workspace(
    topk: int,
    capacity: int,
    max_token: int,
    workspace_threads: int,
):
    import torch

    backend = _load_cpp_backend()
    if backend is None:
        raise RuntimeError(f"C++ backend unavailable: {_CPP_BACKEND_ERROR}")

    token_mark_workspace = torch.zeros((workspace_threads, max_token),
                                       dtype=torch.int32)
    token_pos_workspace = torch.full((workspace_threads, max_token),
                                     -1,
                                     dtype=torch.int32)
    hit_slot_workspace = torch.empty((workspace_threads, capacity),
                                     dtype=torch.int32)
    evictable_slot_workspace = torch.empty((workspace_threads, capacity),
                                           dtype=torch.int32)
    miss_token_workspace = torch.empty((workspace_threads, topk),
                                       dtype=torch.int32)
    miss_position_workspace = torch.empty((workspace_threads, topk),
                                          dtype=torch.int32)
    miss_slot_workspace = torch.empty((workspace_threads, topk),
                                      dtype=torch.int32)
    epochs = torch.zeros((workspace_threads, ), dtype=torch.int32)
    return (
        backend,
        token_mark_workspace,
        token_pos_workspace,
        hit_slot_workspace,
        evictable_slot_workspace,
        miss_token_workspace,
        miss_position_workspace,
        miss_slot_workspace,
        epochs,
        topk,
        capacity,
        max_token,
        workspace_threads,
    )


def run_cpp_lru_once(
    generated: GeneratedCase,
    workspace,
    requested_threads: int,
):
    import torch

    (
        backend,
        token_mark_workspace,
        token_pos_workspace,
        hit_slot_workspace,
        evictable_slot_workspace,
        miss_token_workspace,
        miss_position_workspace,
        miss_slot_workspace,
        epochs,
        topk,
        capacity,
        max_token,
        workspace_threads,
    ) = workspace

    last_req_ids = generated.last_req_ids.copy()
    slot_to_token = generated.slot_to_token.copy()
    lru_slots = generated.lru_slots.copy()
    current_slots = np.full((generated.req_ids.shape[0], topk),
                            -1,
                            dtype=np.int32)
    load_token_indices = np.full((generated.req_ids.shape[0], capacity),
                                 -1,
                                 dtype=np.int32)

    req_ids_t = torch.from_numpy(generated.req_ids)
    last_req_ids_t = torch.from_numpy(last_req_ids)
    topk_t = torch.from_numpy(generated.topk_indices)
    slot_to_token_t = torch.from_numpy(slot_to_token)
    lru_slots_t = torch.from_numpy(lru_slots)
    current_slots_t = torch.from_numpy(current_slots)
    load_token_indices_t = torch.from_numpy(load_token_indices)

    start = time.perf_counter()
    backend.lru_kv_topk(
        req_ids_t.data_ptr(),
        last_req_ids_t.data_ptr(),
        topk_t.data_ptr(),
        slot_to_token_t.data_ptr(),
        lru_slots_t.data_ptr(),
        current_slots_t.data_ptr(),
        load_token_indices_t.data_ptr(),
        token_mark_workspace.data_ptr(),
        token_pos_workspace.data_ptr(),
        hit_slot_workspace.data_ptr(),
        evictable_slot_workspace.data_ptr(),
        miss_token_workspace.data_ptr(),
        miss_position_workspace.data_ptr(),
        miss_slot_workspace.data_ptr(),
        epochs.data_ptr(),
        generated.req_ids.shape[0],
        topk,
        capacity,
        max_token,
        workspace_threads,
        requested_threads,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return (
        elapsed_ms,
        ReferenceResult(
            last_req_ids=last_req_ids,
            slot_to_token=slot_to_token,
            lru_slots=lru_slots,
            current_slots=current_slots,
            load_token_indices=load_token_indices,
        ),
    )


def _assert_matches_reference(name: str, actual: ReferenceResult,
                              expected: ReferenceResult) -> None:
    if not np.array_equal(actual.last_req_ids, expected.last_req_ids):
        raise AssertionError(f"{name} last_req_ids mismatch")
    if not np.array_equal(actual.slot_to_token, expected.slot_to_token):
        raise AssertionError(f"{name} slot_to_token mismatch")
    if not np.array_equal(actual.lru_slots, expected.lru_slots):
        raise AssertionError(f"{name} lru_slots mismatch")
    if not np.array_equal(actual.current_slots, expected.current_slots):
        raise AssertionError(f"{name} current_slots mismatch")
    if not np.array_equal(actual.load_token_indices,
                          expected.load_token_indices):
        raise AssertionError(f"{name} load_token_indices mismatch")


def _iter_cases(
    num_reqs_values: tuple[int, ...],
    seq_lens: tuple[int, ...],
    topk: int,
    capacity: int,
    hit_rates: tuple[float, ...],
) -> list[BenchmarkCase]:
    return [
        BenchmarkCase(
            num_reqs=num_reqs,
            seq_len=seq_len,
            topk=topk,
            capacity=capacity,
            hit_rate=hit_rate,
        )
        for num_reqs in num_reqs_values
        for seq_len in seq_lens
        for hit_rate in hit_rates
    ]


def run_benchmark(
    num_reqs_values: tuple[int, ...] = DEFAULT_NUM_REQS,
    seq_lens: tuple[int, ...] = DEFAULT_SEQ_LENS,
    topk: int = DEFAULT_TOPK,
    capacity: int = DEFAULT_CAPACITY,
    hit_rates: tuple[float, ...] = DEFAULT_HIT_RATES,
    repeat: int = DEFAULT_REPEAT,
    warmup: int = DEFAULT_WARMUP,
    seed: int = 0,
    csv_path: Path | None = None,
    cpp_threads: tuple[int, ...] = DEFAULT_CPP_THREADS,
) -> list[dict[str, float | int | str]]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if warmup < 0:
        raise ValueError("warmup must be non-negative")

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
                "capacity",
                "hit_rate",
                "actual_hit_rate",
                "miss_count",
                "cpp_threads",
                "avg_ms",
                "best_ms",
                "us_per_req",
                "ns_per_slot",
            ],
        )
        writer.writeheader()

    try:
        for case_id, case in enumerate(
                _iter_cases(num_reqs_values, seq_lens, topk, capacity,
                            hit_rates)):
            generated = generate_case(
                num_reqs=case.num_reqs,
                seq_len=case.seq_len,
                topk=case.topk,
                capacity=case.capacity,
                hit_rate=case.hit_rate,
                seed=seed + case_id,
            )
            expected = reference_lru_slots(
                req_ids=generated.req_ids.copy(),
                last_req_ids=generated.last_req_ids.copy(),
                topk_indices=generated.topk_indices.copy(),
                slot_to_token=generated.slot_to_token.copy(),
                lru_slots=generated.lru_slots.copy(),
                capacity=case.capacity,
                max_token=case.seq_len,
            )

            max_cpp_threads = max((thread for thread in cpp_threads
                                   if thread > 0),
                                  default=4)
            workspace_threads = max(64, max_cpp_threads)
            try:
                workspace = make_cpp_lru_workspace(
                    topk=case.topk,
                    capacity=case.capacity,
                    max_token=case.seq_len,
                    workspace_threads=workspace_threads,
                )
            except Exception as exc:
                print(f"SKIP impl=cpp reason={exc}", flush=True)
                continue

            for requested_threads in cpp_threads:
                impl_name = f"cpp_omp_t{_thread_name(requested_threads)}"
                for _ in range(warmup):
                    run_cpp_lru_once(
                        generated=generated,
                        workspace=workspace,
                        requested_threads=requested_threads,
                    )

                best_ms = float("inf")
                total_ms = 0.0
                actual = None
                for _ in range(repeat):
                    elapsed_ms, actual = run_cpp_lru_once(
                        generated=generated,
                        workspace=workspace,
                        requested_threads=requested_threads,
                    )
                    best_ms = min(best_ms, elapsed_ms)
                    total_ms += elapsed_ms

                assert actual is not None
                _assert_matches_reference(impl_name, actual, expected)
                avg_ms = total_ms / repeat
                actual_hit_rate = generated.hit_count / case.topk
                us_per_req = best_ms * 1000.0 / case.num_reqs
                ns_per_slot = (
                    best_ms * 1_000_000.0 /
                    (case.num_reqs * case.capacity))
                row: dict[str, float | int | str] = {
                    "impl": impl_name,
                    "num_reqs": case.num_reqs,
                    "seq_len": case.seq_len,
                    "topk": case.topk,
                    "capacity": case.capacity,
                    "hit_rate": case.hit_rate,
                    "actual_hit_rate": actual_hit_rate,
                    "miss_count": generated.miss_count,
                    "cpp_threads": _thread_name(requested_threads),
                    "avg_ms": avg_ms,
                    "best_ms": best_ms,
                    "us_per_req": us_per_req,
                    "ns_per_slot": ns_per_slot,
                }
                results.append(row)
                print(
                    "LRU_KV_TOPK_RESULT "
                    f"impl={row['impl']} "
                    f"num_reqs={row['num_reqs']} "
                    f"seq_len={row['seq_len']} "
                    f"topk={row['topk']} "
                    f"capacity={row['capacity']} "
                    f"hit_rate={row['hit_rate']:.3f} "
                    f"actual_hit_rate={row['actual_hit_rate']:.3f} "
                    f"miss_count={row['miss_count']} "
                    f"cpp_threads={row['cpp_threads']} "
                    f"avg_ms={row['avg_ms']:.4f} "
                    f"best_ms={row['best_ms']:.4f} "
                    f"us_per_req={row['us_per_req']:.4f} "
                    f"ns_per_slot={row['ns_per_slot']:.4f}",
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
        description="CPU LRU KV topk C++ bookkeeping benchmark")
    parser.add_argument("--num-reqs",
                        default=",".join(str(v) for v in DEFAULT_NUM_REQS))
    parser.add_argument("--seq-lens",
                        default=",".join(str(v) for v in DEFAULT_SEQ_LENS))
    parser.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    parser.add_argument("--capacity", type=int, default=DEFAULT_CAPACITY)
    parser.add_argument("--hit-rates",
                        default=",".join(str(v) for v in DEFAULT_HIT_RATES))
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument("--warmup", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv",
                        type=Path,
                        default=Path("benchmarks/ops/"
                                     "cpu_lru_kv_topk_results.csv"))
    parser.add_argument("--cpp-threads",
                        default="auto,1,2,4,8,16,32,64",
                        help=("comma-separated C++ threads: "
                              "auto,1,2,4,8,16,32,64"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        num_reqs_values=_parse_int_list(args.num_reqs),
        seq_lens=_parse_int_list(args.seq_lens),
        topk=args.topk,
        capacity=args.capacity,
        hit_rates=_parse_float_list(args.hit_rates),
        repeat=args.repeat,
        warmup=args.warmup,
        seed=args.seed,
        csv_path=args.csv,
        cpp_threads=_parse_cpp_thread_list(args.cpp_threads),
    )


if __name__ == "__main__":
    main()
