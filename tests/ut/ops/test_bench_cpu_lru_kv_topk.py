import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "ops"
    / "bench_cpu_lru_kv_topk.py"
)
SPEC = importlib.util.spec_from_file_location("bench_cpu_lru_kv_topk",
                                              MODULE_PATH)
bench_lru = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_lru
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_lru)


def test_generate_case_produces_requested_hit_rate_and_unique_topk():
    case = bench_lru.generate_case(
        num_reqs=3,
        seq_len=128,
        topk=16,
        capacity=32,
        hit_rate=0.5,
        seed=7,
    )

    assert case.req_ids.dtype == np.int64
    assert case.last_req_ids.dtype == np.int64
    assert case.topk_indices.dtype == np.int32
    assert case.slot_to_token.dtype == np.int32
    assert case.lru_slots.dtype == np.int32
    assert case.req_ids.shape == (3, )
    assert case.topk_indices.shape == (3, 16)
    assert case.slot_to_token.shape == (3, 32)
    assert case.lru_slots.shape == (3, 32)

    for row in range(3):
        assert len(set(case.topk_indices[row].tolist())) == 16
        resident = set(case.slot_to_token[row].tolist())
        hits = sum(int(token) in resident for token in case.topk_indices[row])
        assert hits == 8
        np.testing.assert_array_equal(np.sort(case.lru_slots[row]),
                                      np.arange(32, dtype=np.int32))


def test_reference_lru_slots_matches_slot_aligned_output():
    req_ids = np.array([1], dtype=np.int64)
    last_req_ids = np.array([1], dtype=np.int64)
    topk = np.array([[99, 20, 30, 101]], dtype=np.int32)
    slot_to_token = np.array([[10, 20, 30, 40]], dtype=np.int32)
    lru_slots = np.array([[1, 2, 3, 0]], dtype=np.int32)

    result = bench_lru.reference_lru_slots(
        req_ids=req_ids.copy(),
        last_req_ids=last_req_ids.copy(),
        topk_indices=topk.copy(),
        slot_to_token=slot_to_token.copy(),
        lru_slots=lru_slots.copy(),
        capacity=4,
        max_token=128,
    )

    np.testing.assert_array_equal(result.current_slots,
                                  np.array([[3, 1, 2, 0]], dtype=np.int32))
    np.testing.assert_array_equal(result.load_token_indices,
                                  np.array([[101, -1, -1, 99]],
                                           dtype=np.int32))
    np.testing.assert_array_equal(result.lru_slots,
                                  np.array([[3, 0, 1, 2]], dtype=np.int32))
