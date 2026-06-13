import pytest


def test_maybe_prepare_lru_kv_topk_requires_connector(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_ascend.attention.utils as attention_utils

    monkeypatch.setattr(attention_utils, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(attention_utils, "is_v1_kv_transfer_group", lambda: False)

    tensor = torch.empty((1, 2), dtype=torch.int32)
    ids = torch.empty((1, ), dtype=torch.int64)
    with pytest.raises(RuntimeError, match="requires v1 KV transfer group"):
        attention_utils.maybe_prepare_lru_kv_topk_graph(
            "layer.0",
            1,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            ids,
            ids,
            128,
            False,
        )


def test_maybe_prepare_lru_kv_topk_forwards_exact_contract(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_ascend.attention.utils as attention_utils

    calls = []

    class FakeConnector:

        def prepare_lru_kv_topk(self, *args):
            calls.append(args)
            return True

    monkeypatch.setattr(attention_utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(attention_utils, "is_v1_kv_transfer_group",
                        lambda: True)
    monkeypatch.setattr(attention_utils, "get_kv_transfer_group",
                        lambda: FakeConnector())

    topk = torch.tensor([[3, 4]], dtype=torch.int32)
    slot_to_token = torch.full((1, 4), -1, dtype=torch.int32)
    lru_slots = torch.arange(4, dtype=torch.int32).view(1, 4)
    current_slots = torch.full((1, 2), -1, dtype=torch.int32)
    load_token_indices = torch.full((1, 4), -1, dtype=torch.int32)
    req_ids = torch.tensor([9], dtype=torch.int64)
    last_req_ids = torch.tensor([9], dtype=torch.int64)

    prepared = attention_utils.maybe_prepare_lru_kv_topk_graph(
        "layer.0",
        1,
        topk,
        slot_to_token,
        lru_slots,
        current_slots,
        load_token_indices,
        req_ids,
        last_req_ids,
        128,
        True,
    )

    assert prepared is True
    assert len(calls) == 1
    assert calls[0][0] == "layer.0"
    assert calls[0][1] == 1
    assert calls[0][2] is topk
    assert calls[0][3] is slot_to_token
    assert calls[0][4] is lru_slots
    assert calls[0][5] is current_slots
    assert calls[0][6] is load_token_indices
    assert calls[0][7] is req_ids
    assert calls[0][8] is last_req_ids
    assert calls[0][9] == 128
    assert calls[0][10] is True


def test_maybe_prepare_lru_kv_topk_uses_runtime_capture_state(monkeypatch):
    torch = pytest.importorskip("torch")
    import vllm_ascend.attention.utils as attention_utils

    calls = []

    class FakeConnector:

        def prepare_lru_kv_topk(self, *args):
            calls.append(args)
            return True

    monkeypatch.setattr(attention_utils, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(attention_utils, "is_v1_kv_transfer_group",
                        lambda: True)
    monkeypatch.setattr(attention_utils, "get_kv_transfer_group",
                        lambda: FakeConnector())
    monkeypatch.setattr(attention_utils, "_is_current_stream_capturing",
                        lambda: True)

    topk = torch.tensor([[3, 4]], dtype=torch.int32)
    slot_to_token = torch.full((1, 4), -1, dtype=torch.int32)
    lru_slots = torch.arange(4, dtype=torch.int32).view(1, 4)
    current_slots = torch.full((1, 2), -1, dtype=torch.int32)
    load_token_indices = torch.full((1, 4), -1, dtype=torch.int32)
    req_ids = torch.tensor([9], dtype=torch.int64)
    last_req_ids = torch.tensor([9], dtype=torch.int64)

    prepared = attention_utils.maybe_prepare_lru_kv_topk_graph(
        "layer.0",
        1,
        topk,
        slot_to_token,
        lru_slots,
        current_slots,
        load_token_indices,
        req_ids,
        last_req_ids,
        128,
        False,
    )

    assert prepared is True
    assert len(calls) == 1
    assert calls[0][10] is True
