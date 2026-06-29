from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from vllm_ascend.kv_offload.dsa_pd_mooncake import (
    kv_transfer_uses_mooncake,
    should_use_dsa_pd_mooncake_cpu_kv,
)


class TestDSAPDMooncakeCPUKV(TestCase):
    def test_detects_nested_mooncake_connector(self):
        kv_transfer_config = SimpleNamespace(
            kv_connector="",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "P2pNcclConnector"},
                    {"kv_connector": "MooncakeLayerwiseConnector"},
                ],
            },
        )

        self.assertTrue(kv_transfer_uses_mooncake(kv_transfer_config))

    @patch("vllm_ascend.kv_offload.dsa_pd_mooncake.envs.VLLM_ASCEND_DSA_PD_MOONCAKE_CPU_KV", True, create=True)
    def test_cpu_kv_requires_decode_consumer_sparse_fused_mooncake(self):
        kv_transfer_config = SimpleNamespace(
            is_kv_consumer=True,
            kv_connector="MooncakeConnector",
            kv_connector_extra_config={},
        )
        vllm_config = SimpleNamespace(kv_transfer_config=kv_transfer_config)

        self.assertTrue(
            should_use_dsa_pd_mooncake_cpu_kv(
                vllm_config,
                use_sparse=True,
                is_kv_consumer=True,
                enable_cpu_kv_store=True,
                dsa_sparse_attention_mode="fused_overlap",
            )
        )
        self.assertFalse(
            should_use_dsa_pd_mooncake_cpu_kv(
                vllm_config,
                use_sparse=True,
                is_kv_consumer=True,
                enable_cpu_kv_store=True,
                dsa_sparse_attention_mode="baseline",
            )
        )
