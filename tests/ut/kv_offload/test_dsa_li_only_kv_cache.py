from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from vllm_ascend.ascend_config import DSASparseAttentionConfig
from vllm_ascend.core.kv_cache_interface import AscendMLAAttentionSpec


def _vllm_config(index_topk: int = 2048):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(index_topk=index_topk),
        ),
        scheduler_config=SimpleNamespace(max_num_seqs=64),
    )


def test_dsa_config_defaults_keep_legacy_layout():
    cfg = DSASparseAttentionConfig({}, _vllm_config())

    assert cfg.mode == "baseline"
    assert cfg.enable_cpu_kv_store is False
    assert cfg.hbm_kv_cache_layout == "legacy"
    assert cfg.selection_cache_max_tokens is None
    assert cfg.selection_cache_max_topk == 2048


def test_dsa_config_accepts_li_only_layout_and_selection_sizing():
    cfg = DSASparseAttentionConfig(
        {
            "mode": "fused_overlap",
            "enable_cpu_kv_store": True,
            "hbm_kv_cache_layout": "li_only",
            "selection_cache_max_tokens": 128,
            "selection_cache_max_topk": 4096,
        },
        _vllm_config(),
    )

    assert cfg.hbm_kv_cache_layout == "li_only"
    assert cfg.selection_cache_max_tokens == 128
    assert cfg.selection_cache_max_topk == 4096


@pytest.mark.parametrize("layout", ["", "device_only", "cpu_only", "li"])
def test_dsa_config_rejects_unknown_hbm_layout(layout):
    with pytest.raises(ValueError, match="hbm_kv_cache_layout"):
        DSASparseAttentionConfig(
            {
                "mode": "fused_overlap",
                "enable_cpu_kv_store": True,
                "hbm_kv_cache_layout": layout,
            },
            _vllm_config(),
        )


@pytest.mark.parametrize("value", [0, -1])
def test_dsa_config_rejects_invalid_selection_cache_max_tokens(value):
    with pytest.raises(ValueError, match="selection_cache_max_tokens"):
        DSASparseAttentionConfig(
            {"selection_cache_max_tokens": value},
            _vllm_config(),
        )


def test_li_only_mla_spec_keeps_logical_dims_and_overrides_page_size():
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=704,
        sparse_head_dim=(512, 64, 128),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
        hbm_kv_cache_layout="li_only",
    )

    assert spec.head_size == 704
    assert spec.sparse_head_dim == (512, 64, 128)
    assert spec.sparse_kv_cache_ratio == (704 / 512, 704 / 64, 704 / 128, None)
    assert spec.li_page_size_bytes == 128 * 128 * 2
    assert spec.kv_lora_page_size_bytes == 128 * 512 * 2
    assert spec.k_rope_page_size_bytes == 128 * 64 * 2
    assert spec.full_kv_page_size_bytes == 128 * (512 + 64) * 2
    assert spec.page_size_bytes == 128 * 128 * 2


def test_legacy_mla_spec_page_size_stays_full_sparse_page():
    spec = AscendMLAAttentionSpec(
        block_size=128,
        num_kv_heads=1,
        head_size=704,
        sparse_head_dim=(512, 64, 128),
        dtype=torch.bfloat16,
        cache_dtype_str="auto",
    )

    assert spec.page_size_bytes == 128 * 704 * 2


def test_v2_kv_cache_spec_rejects_li_only_layout():
    from vllm_ascend.worker.v2 import attn_utils

    ascend_config = SimpleNamespace(
        dsa_sparse_attention_config=SimpleNamespace(
            hbm_kv_cache_layout="li_only",
        )
    )

    with patch("vllm_ascend.worker.v2.attn_utils.get_ascend_config", return_value=ascend_config):
        with pytest.raises(NotImplementedError, match="V2 model runner"):
            attn_utils.get_kv_cache_spec(_vllm_config())
