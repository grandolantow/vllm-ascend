import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "benchmarks"
    / "ops"
    / "bench_cpu_cache_miss_topk.py"
)
SPEC = importlib.util.spec_from_file_location("bench_cpu_cache_miss_topk",
                                              MODULE_PATH)
bench_cpu = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench_cpu
assert SPEC.loader is not None
SPEC.loader.exec_module(bench_cpu)

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
            name="cpu_sparse_attn_cache_miss_topk_test",
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

    if not hasattr(_CPP_BACKEND, "cache_miss_topk"):
        pytest.skip("cpu_sparse_attn.cache_miss_topk is unavailable")
    return _CPP_BACKEND


def _cpp_update_topk_indices(req_ids,
                             last_req_ids,
                             old,
                             new,
                             max_token,
                             requested_threads=1,
                             workspace_threads=4,
                             use_hash=False):
    torch = pytest.importorskip("torch")
    backend = _load_cpp_backend()

    req_ids_tensor = torch.from_numpy(req_ids.copy())
    last_req_ids_tensor = torch.from_numpy(last_req_ids.copy())
    old_tensor = torch.from_numpy(old.copy())
    new_tensor = torch.from_numpy(new.copy())
    if use_hash:
        if not hasattr(backend, "cache_miss_topk_hash"):
            pytest.skip("cpu_sparse_attn.cache_miss_topk_hash is unavailable")
        hash_capacity = bench_cpu._next_power_of_two(
            max(8192, new.shape[1] * 4))
        hash_keys = torch.empty([workspace_threads, hash_capacity],
                                dtype=torch.int32)
        hash_marks = torch.zeros([workspace_threads, hash_capacity],
                                 dtype=torch.int32)
        miss_workspace = torch.empty([workspace_threads, new.shape[1]],
                                     dtype=torch.int32)
        epochs = torch.zeros([workspace_threads], dtype=torch.int32)
        backend.cache_miss_topk_hash(
            req_ids_tensor.data_ptr(),
            last_req_ids_tensor.data_ptr(),
            old_tensor.data_ptr(),
            new_tensor.data_ptr(),
            hash_keys.data_ptr(),
            hash_marks.data_ptr(),
            miss_workspace.data_ptr(),
            epochs.data_ptr(),
            req_ids.shape[0],
            new.shape[1],
            max_token,
            hash_capacity,
            workspace_threads,
            requested_threads,
        )
    else:
        mark_workspace = torch.zeros([workspace_threads, max_token],
                                     dtype=torch.int32)
        miss_workspace = torch.empty([workspace_threads, new.shape[1]],
                                     dtype=torch.int32)
        epochs = torch.zeros([workspace_threads], dtype=torch.int32)

        backend.cache_miss_topk(
            req_ids_tensor.data_ptr(),
            last_req_ids_tensor.data_ptr(),
            old_tensor.data_ptr(),
            new_tensor.data_ptr(),
            mark_workspace.data_ptr(),
            miss_workspace.data_ptr(),
            epochs.data_ptr(),
            req_ids.shape[0],
            new.shape[1],
            max_token,
            workspace_threads,
            requested_threads,
        )
    return last_req_ids_tensor.numpy(), old_tensor.numpy(), new_tensor.numpy()


def test_reference_matches_spec_example_with_raw_tokens():
    req_ids = np.array([3], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    np.testing.assert_array_equal(out,
                                  np.array([[5, -1, 18, -1, 30, 33]],
                                           dtype=np.int32))
    np.testing.assert_array_equal(
        old, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int64))
    np.testing.assert_array_equal(last_req_ids, req_ids)
    assert out is new


def test_reference_treats_changed_req_id_as_all_miss():
    req_ids = np.array([7], dtype=np.int64)
    last_req_ids = np.array([3], dtype=np.int64)
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    new = np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    np.testing.assert_array_equal(
        out, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32))
    np.testing.assert_array_equal(
        old, np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int64))
    np.testing.assert_array_equal(last_req_ids, req_ids)


def test_reference_keeps_same_raw_token_per_row():
    req_ids = np.array([3, 7], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([
        [5, 10, -1, -1],
        [5, 20, -1, -1],
    ], dtype=np.int64)
    new = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    np.testing.assert_array_equal(out,
                                  np.array([
                                      [-1, 30, 31, -1],
                                      [-1, 40, 41, -1],
                                  ], dtype=np.int32))
    expected_old = np.array([
        [5, 30, 31, -1],
        [5, 40, 41, -1],
    ], dtype=np.int64)
    np.testing.assert_array_equal(old, expected_old)
    np.testing.assert_array_equal(last_req_ids, req_ids)


def test_reference_deduplicates_repeated_new_tokens():
    req_ids = np.array([11], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[1, 2, 3, -1, -1, -1]], dtype=np.int64)
    new = np.array([[2, 4, 4, 5, 5, 6]], dtype=np.int32)

    out = bench_cpu.reference_update_topk_indices(req_ids, last_req_ids, old,
                                                  new)

    assert sorted(out[out >= 0].tolist()) == [4, 5, 6]
    assert sorted(old[old >= 0].tolist()) == [2, 4, 5, 6]
    assert old[0, 0] == 4
    np.testing.assert_array_equal(last_req_ids, req_ids)


@pytest.mark.parametrize("num_reqs", [1, 2, 4])
@pytest.mark.parametrize("seq_len", [32768, 131072])
@pytest.mark.parametrize("hit_rate", [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0])
def test_generate_case_exact_overlap(num_reqs, seq_len, hit_rate):
    topk = 2048

    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=num_reqs,
        seq_len=seq_len,
        topk=topk,
        hit_rate=hit_rate,
        seed=123,
    )

    assert req_ids.shape == (num_reqs,)
    assert last_req_ids.shape == (num_reqs,)
    assert old.shape == (num_reqs, topk)
    assert new.shape == (num_reqs, topk)
    assert req_ids.dtype == np.int64
    assert last_req_ids.dtype == np.int64
    assert old.dtype == np.int64
    assert new.dtype == np.int32
    assert old.flags.c_contiguous
    assert new.flags.c_contiguous
    assert int(new.min()) >= 0
    assert int(new.max()) < seq_len

    expected_overlap = int(round(topk * hit_rate))
    for row in range(num_reqs):
        old_raw = old[row]
        assert int(old_raw.min()) >= 0
        assert int(old_raw.max()) < seq_len
        overlap = len(set(old_raw.tolist()) & set(new[row].tolist()))
        assert overlap == expected_overlap


def test_numpy_baseline_matches_reference_with_req_ids():
    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=256,
        hit_rate=0.3,
        seed=7,
    )
    ref_old = old.copy()
    ref_last_req_ids = last_req_ids.copy()
    ref_new = new.copy()
    np_old = old.copy()
    np_last_req_ids = last_req_ids.copy()
    np_new = new.copy()

    expected = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)
    actual = bench_cpu.numpy_isin_update_topk_indices(
        req_ids, np_last_req_ids, np_old, np_new)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np_old, ref_old)
    np.testing.assert_array_equal(np_last_req_ids, ref_last_req_ids)
    assert actual is np_new


def test_numba_stamp_matches_reference_when_numba_available():
    if bench_cpu.NUMBA_AVAILABLE is False:
        pytest.skip("numba is not installed in this environment")

    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.7,
        seed=11,
    )
    ref_old = old.copy()
    ref_last_req_ids = last_req_ids.copy()
    ref_new = new.copy()
    nb_old = old.copy()
    nb_last_req_ids = last_req_ids.copy()
    nb_new = new.copy()

    expected = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)
    workspace = bench_cpu.make_topk_workspace(topk=512, max_token=32768)
    actual = bench_cpu.update_topk_indices_cpu(req_ids, nb_last_req_ids,
                                               nb_old, nb_new, workspace)

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(nb_old, ref_old)
    np.testing.assert_array_equal(nb_last_req_ids, ref_last_req_ids)
    assert actual is nb_new


def test_cpp_backend_matches_reference_when_available():
    cases = []

    req_ids = np.array([3], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32),
        128,
    ))

    req_ids = np.array([3, 7], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([
        [5, 10, -1, -1],
        [5, 20, -1, -1],
    ], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([
            [5, 30, 31, -1],
            [5, 40, 41, -1],
        ], dtype=np.int32),
        128,
    ))

    req_ids = np.array([11], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[1, 2, 3, -1, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[2, 4, 4, 5, 5, 6]], dtype=np.int32),
        128,
    ))

    req_ids = np.array([5], dtype=np.int64)
    last_req_ids = req_ids.copy()
    old = np.array([[-1, -1, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[8, 9, 10, 11]], dtype=np.int32),
        128,
    ))

    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.7,
        seed=13,
    )
    cases.append((req_ids, last_req_ids, old, new, 32768))

    req_ids = np.array([7], dtype=np.int64)
    last_req_ids = np.array([3], dtype=np.int64)
    old = np.array([[3, 7, 15, 22, -1, -1]], dtype=np.int64)
    cases.append((
        req_ids,
        last_req_ids,
        old,
        np.array([[5, 7, 18, 22, 30, 33]], dtype=np.int32),
        128,
    ))

    for req_ids, last_req_ids, old, new, max_token in cases:
        ref_old = old.copy()
        ref_last_req_ids = last_req_ids.copy()
        ref_new = new.copy()
        expected = bench_cpu.reference_update_topk_indices(
            req_ids, ref_last_req_ids, ref_old, ref_new)

        cpp_last_req_ids, cpp_old, actual = _cpp_update_topk_indices(
            req_ids,
            last_req_ids,
            old,
            new,
            max_token=max_token,
        )

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(cpp_old, ref_old)
        np.testing.assert_array_equal(cpp_last_req_ids, ref_last_req_ids)


@pytest.mark.parametrize("requested_threads", [-1, 1, 2, 4, 8, 16, 32, 64])
def test_cpp_backend_thread_counts_match_reference_when_available(
        requested_threads):
    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.7,
        seed=17,
    )
    ref_old = old.copy()
    ref_last_req_ids = last_req_ids.copy()
    ref_new = new.copy()
    expected = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)

    cpp_last_req_ids, cpp_old, actual = _cpp_update_topk_indices(
        req_ids,
        last_req_ids,
        old,
        new,
        max_token=32768,
        requested_threads=requested_threads,
        workspace_threads=64,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cpp_old, ref_old)
    np.testing.assert_array_equal(cpp_last_req_ids, ref_last_req_ids)


@pytest.mark.parametrize("requested_threads", [-1, 1, 2, 4, 8, 16, 32, 64])
def test_cpp_hash_backend_thread_counts_match_reference_when_available(
        requested_threads):
    req_ids, last_req_ids, old, new = bench_cpu.generate_case(
        num_reqs=4,
        seq_len=32768,
        topk=512,
        hit_rate=0.5,
        seed=23,
    )
    ref_old = old.copy()
    ref_last_req_ids = last_req_ids.copy()
    ref_new = new.copy()
    expected = bench_cpu.reference_update_topk_indices(
        req_ids, ref_last_req_ids, ref_old, ref_new)

    cpp_last_req_ids, cpp_old, actual = _cpp_update_topk_indices(
        req_ids,
        last_req_ids,
        old,
        new,
        max_token=32768,
        requested_threads=requested_threads,
        workspace_threads=64,
        use_hash=True,
    )

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(cpp_old, ref_old)
    np.testing.assert_array_equal(cpp_last_req_ids, ref_last_req_ids)
