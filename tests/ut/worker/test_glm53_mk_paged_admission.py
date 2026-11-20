"""CPU regression checks for the GLM-5.3 MK paged decode configuration."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from vllm_ascend import envs
from vllm_ascend.worker import va_mk_bridge


class TestGLM53MKPagedAdmission(TestCase):
    def test_ep6_accepts_16k_and_128k(self):
        model = self.make_model(tp_size=1, ep_size=6, depth=45)
        tensor_factory = SimpleNamespace(
            inference_mode=lambda _: nullcontext(),
            no_grad=nullcontext,
            empty=lambda *args, **kwargs: object(),
            zeros=lambda *args, **kwargs: object(),
            int64="int64",
            bfloat16="bfloat16",
            float32="float32",
        )
        with patch.object(va_mk_bridge, "torch", tensor_factory):
            for capacity in (16 * 1024, 128 * 1024):
                with self.subTest(capacity=capacity):
                    bridge = va_mk_bridge.VAMKBridge(model, ep_size=6, max_context_tokens=capacity)
                    self.assertEqual(bridge.max_context_tokens, capacity)

    def test_paged_decode_rejects_tp1_ep8(self):
        model = self.make_model(tp_size=1, ep_size=8, depth=45)
        with self.assertRaisesRegex(ValueError, "paged VA bridge requires"):
            va_mk_bridge.VAMKBridge(model, ep_size=8, max_context_tokens=16 * 1024)

    def test_environment_defaults_and_lazy_overrides(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(envs.VLLM_ASCEND_GLM53_MK_EP)
            self.assertFalse(envs.VLLM_ASCEND_GLM53_MK_MTP_EP)
            self.assertEqual(envs.VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS, "64")
            self.assertIsNone(envs.VLLM_ASCEND_GLM53_MK_CHECKPOINT)
            with patch.dict("os.environ", {
                "VLLM_ASCEND_GLM53_MK_EP": "1",
                "VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS": "131072",
            }):
                self.assertTrue(envs.VLLM_ASCEND_GLM53_MK_EP)
                self.assertEqual(envs.VLLM_ASCEND_GLM53_MK_MAX_CONTEXT_TOKENS, "131072")

    @staticmethod
    def make_model(tp_size, ep_size, depth):
        return SimpleNamespace(
            tp_size=tp_size,
            tp_rank=0,
            model_args=SimpleNamespace(
                num_hidden_layers=depth,
                ep_size=ep_size,
                ep_rank=0,
                attention_impl="mk_decode",
                linear_layer_ids=tuple(i for i in range(depth) if i % 4 != 3),
                full_attention_layer_ids=tuple(range(3, depth, 4)),
                indexer_skip_rope=True,
                index_kpool=4,
            ),
            _mk_decode_state=SimpleNamespace(
                _shared_pools=None,
                batch=SimpleNamespace(index_cos_sin_cache=SimpleNamespace(shape=(64,))),
                positions=SimpleNamespace(device="cpu"),
            ),
        )
