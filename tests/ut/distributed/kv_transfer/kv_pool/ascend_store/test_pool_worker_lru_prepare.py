import sys
import types

import torch


def _install_pool_worker_import_stubs(monkeypatch):
    fake_torch_npu = types.ModuleType("torch_npu")
    fake_torch_npu.__file__ = __file__

    class FakeNPU:

        def __init__(self):
            self.capture_state = False
            self.launched = []

        def is_current_stream_capturing(self):
            return self.capture_state

        def current_stream(self):
            return "fake_stream"

        def _subscribe_report(self, stream):
            return None

        def _launch_host_func(self, stream, fn, args):
            self.launched.append((stream, fn, args))
            fn(args)

    fake_torch_npu.npu = FakeNPU()
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)

    fake_backend = types.SimpleNamespace(
        cache_miss_topk=lambda *args: None,
        lru_kv_topk=lambda *args: None,
    )
    monkeypatch.setattr(
        "torch.utils.cpp_extension.load",
        lambda *args, **kwargs: fake_backend,
    )


class FakeWorkspace:

    def __init__(self):
        self.token_mark_workspace = torch.empty((1, 16), dtype=torch.int32)
        self.token_pos_workspace = torch.empty((1, 16), dtype=torch.int32)
        self.hit_slot_workspace = torch.empty((1, 4), dtype=torch.int32)
        self.evictable_slot_workspace = torch.empty((1, 4), dtype=torch.int32)
        self.miss_token_workspace = torch.empty((1, 4), dtype=torch.int32)
        self.miss_position_workspace = torch.empty((1, 4), dtype=torch.int32)
        self.miss_slot_workspace = torch.empty((1, 4), dtype=torch.int32)
        self.epochs = torch.empty((1, 16), dtype=torch.int32)
        self.workspace_threads = 1


def test_prepare_lru_kv_topk_prints_cache_miss_style_logs(monkeypatch, capsys):
    _install_pool_worker_import_stubs(monkeypatch)

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (  # noqa: E501
        KVPoolWorker,
    )

    worker = object.__new__(KVPoolWorker)
    worker.cpu_lru_kv_workspace = FakeWorkspace()
    worker.topk = 4
    worker.topk_indices_buffer_cpu = torch.empty((1, 4), dtype=torch.int32)
    worker.lru_slot_to_token_buffer_cpu = torch.empty((1, 4),
                                                      dtype=torch.int32)
    worker.lru_slots_buffer_cpu = torch.empty((1, 4), dtype=torch.int32)
    worker.lru_current_slots_buffer_cpu = torch.empty((1, 4),
                                                      dtype=torch.int32)
    worker.lru_load_token_indices_buffer_cpu = torch.empty((1, 4),
                                                           dtype=torch.int32)
    worker.req_ids_tensor_buffer_cpu = torch.empty((1,), dtype=torch.int64)
    worker.last_req_ids_tensor_buffer_cpu = torch.empty((1,),
                                                        dtype=torch.int64)

    def fake_lru_cpu(args):
        current_slots_ptr = args[5]
        load_token_indices_ptr = args[6]
        current_slots = worker.lru_current_slots_buffer_cpu
        load_token_indices = worker.lru_load_token_indices_buffer_cpu
        assert current_slots.data_ptr() == current_slots_ptr
        assert load_token_indices.data_ptr() == load_token_indices_ptr
        current_slots.fill_(0)
        load_token_indices.fill_(-1)

    worker.lru_kv_topk_cpu = fake_lru_cpu

    prepared = worker.prepare_lru_kv_topk(
        "model.layers.0.self_attn.attn",
        1,
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        torch.full((1, 4), -1, dtype=torch.int32),
        torch.arange(4, dtype=torch.int32).view(1, 4),
        torch.full((1, 4), -1, dtype=torch.int32),
        torch.full((1, 4), -1, dtype=torch.int32),
        torch.tensor([11], dtype=torch.int64),
        torch.tensor([-1], dtype=torch.int64),
        16,
        False,
    )

    output = capsys.readouterr().out
    assert prepared is True
    assert "[SFA][lru_prepare][worker]" in output
    assert "capturing=False prepared=True action=copy_h2d" in output


def test_prepare_lru_kv_topk_uses_runtime_capture_state(monkeypatch, capsys):
    _install_pool_worker_import_stubs(monkeypatch)

    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store import pool_worker
    from vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.pool_worker import (  # noqa: E501
        KVPoolWorker,
    )

    monkeypatch.setattr(pool_worker.torch_npu.npu, "capture_state", True)

    worker = object.__new__(KVPoolWorker)
    worker.cpu_lru_kv_workspace = FakeWorkspace()
    worker.topk = 4
    worker.topk_indices_buffer_cpu = torch.empty((1, 4), dtype=torch.int32)
    worker.lru_slot_to_token_buffer_cpu = torch.empty((1, 4),
                                                      dtype=torch.int32)
    worker.lru_slots_buffer_cpu = torch.empty((1, 4), dtype=torch.int32)
    worker.lru_current_slots_buffer_cpu = torch.empty((1, 4),
                                                      dtype=torch.int32)
    worker.lru_load_token_indices_buffer_cpu = torch.empty((1, 4),
                                                           dtype=torch.int32)
    worker.req_ids_tensor_buffer_cpu = torch.empty((1,), dtype=torch.int64)
    worker.last_req_ids_tensor_buffer_cpu = torch.empty((1,),
                                                        dtype=torch.int64)

    def fake_lru_cpu(args):
        worker.lru_current_slots_buffer_cpu.fill_(0)
        worker.lru_load_token_indices_buffer_cpu.fill_(-1)

    worker.lru_kv_topk_cpu = fake_lru_cpu

    prepared = worker.prepare_lru_kv_topk(
        "model.layers.0.self_attn.attn",
        1,
        torch.tensor([[1, 2, 3, 4]], dtype=torch.int32),
        torch.full((1, 4), -1, dtype=torch.int32),
        torch.arange(4, dtype=torch.int32).view(1, 4),
        torch.full((1, 4), -1, dtype=torch.int32),
        torch.full((1, 4), -1, dtype=torch.int32),
        torch.tensor([11], dtype=torch.int64),
        torch.tensor([-1], dtype=torch.int64),
        16,
        False,
    )

    output = capsys.readouterr().out
    assert prepared is True
    assert "capturing=True action=launch_host_func" in output
    assert "capturing=True prepared=True action=copy_h2d" in output
    assert len(pool_worker.torch_npu.npu.launched) == 1
