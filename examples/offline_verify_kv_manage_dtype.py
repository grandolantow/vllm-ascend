import os
import random
import string

# =========================
# Runtime env
# =========================
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

os.environ["VLLM_USE_V1"] = "1"

# 打开 kv manage / DSA LI-only debug log
os.environ.setdefault("VLLM_ASCEND_DSA_LI_ONLY_DEBUG", "1")
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_ASCEND_DSA_PD_MOONCAKE_CPU_KV", "1")

TP_SIZE = int(os.environ.get("TP_SIZE", "16"))
os.environ.setdefault(
    "ASCEND_RT_VISIBLE_DEVICES",
    ",".join(str(i) for i in range(TP_SIZE)),
)

import torch
import torch_npu  # noqa: F401
from vllm import LLM, SamplingParams


# =========================
# Config knobs
# =========================
MODEL_PATH = os.environ.get("MODEL_PATH", "/mnt/share/s00886374/GLM-5.2-w8a8")

NUM_REQ = int(os.environ.get("NUM_REQ", "1"))

# 默认 4096：足够触发 topk/selection/lru 相关路径，比 32k 快很多。
# 如果要更接近你原脚本，运行时设 INPUT_LEN=32768。
INPUT_LEN = int(os.environ.get("INPUT_LEN", "4096"))

MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "33792"))
MAX_NUM_BATCHED_TOKENS = int(os.environ.get("MAX_NUM_BATCHED_TOKENS", str(MAX_MODEL_LEN)))
MAX_NUM_SEQS = int(os.environ.get("MAX_NUM_SEQS", "4"))

MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "2"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.862"))

TOPK = int(os.environ.get("TOPK", "2048"))
LRU_BUFFER_SIZE = int(os.environ.get("LRU_BUFFER_SIZE", "4096"))
SELECTION_TOPK_BLOCK_SIZE = int(os.environ.get("SELECTION_TOPK_BLOCK_SIZE", "64"))
SELECTION_CACHE_MAX_TOKENS = int(
    os.environ.get("SELECTION_CACHE_MAX_TOKENS", str(MAX_MODEL_LEN))
)
SELECTION_CACHE_MAX_TOPK = int(os.environ.get("SELECTION_CACHE_MAX_TOPK", str(TOPK)))

STOP_AFTER_INIT = os.environ.get("STOP_AFTER_INIT", "0") == "1"


def generate_prompts_auto(input_len: int, batchsize: int):
    random.seed(0)
    return [
        " ".join(random.choice(string.ascii_letters) for _ in range(input_len))
        for _ in range(batchsize)
    ]


prompts = generate_prompts_auto(INPUT_LEN, NUM_REQ)
sampling_params = SamplingParams(max_tokens=MAX_TOKENS, temperature=0)

print("========== offline kv manage dtype verify ==========")
print(f"MODEL_PATH={MODEL_PATH}")
print(f"ASCEND_RT_VISIBLE_DEVICES={os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}")
print(f"TP_SIZE={TP_SIZE}")
print(f"NUM_REQ={NUM_REQ}")
print(f"INPUT_LEN={INPUT_LEN}")
print(f"MAX_MODEL_LEN={MAX_MODEL_LEN}")
print(f"MAX_NUM_BATCHED_TOKENS={MAX_NUM_BATCHED_TOKENS}")
print(f"MAX_TOKENS={MAX_TOKENS}")
print(f"TOPK={TOPK}")
print(f"LRU_BUFFER_SIZE={LRU_BUFFER_SIZE}")
print(f"SELECTION_CACHE_MAX_TOKENS={SELECTION_CACHE_MAX_TOKENS}")
print(f"SELECTION_CACHE_MAX_TOPK={SELECTION_CACHE_MAX_TOPK}")
print("====================================================")


llm = LLM(
    model=MODEL_PATH,
    compilation_config={
        "cudagraph_mode": "FULL_DECODE_ONLY",
        "cudagraph_capture_sizes": [1, 2, 4],
    },
    tensor_parallel_size=TP_SIZE,
    enable_expert_parallel=True,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    quantization="ascend",
    max_num_seqs=MAX_NUM_SEQS,
    max_model_len=MAX_MODEL_LEN,
    max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,

    # =========================
    # 关键：KV manage / DSA LI-only config
    # =========================
    additional_config={

        "dsa_sparse_attention_config": {
            # 不要用 baseline；baseline 下 CPU KV store / li_only 会被忽略
            "mode": "fused_overlap",

            # 让 DSA + CPU KV store gate 生效
            "enable_cpu_kv_store": True,

            # 本次重点：HBM 只放 LI / selection cache 侧 KV
            "hbm_kv_cache_layout": "li_only",

            # selection cache 静态容量
            "selection_cache_max_tokens": SELECTION_CACHE_MAX_TOKENS,
            "selection_cache_max_topk": SELECTION_CACHE_MAX_TOPK,
            "selection_topk_block_size": SELECTION_TOPK_BLOCK_SIZE,

            # 明确 sparse_count，避免依赖模型 config 自动推导
            "sparse_count": TOPK,
        },
    },

    # 保留你脚本里的离线 kv_both 路径，单进程里同时具备 producer/consumer 角色
    kv_transfer_config={
        "kv_connector": "MooncakeConnectorV1",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "use_ascend_direct": True,
        },
    },

    block_size=128,
    async_scheduling=False,
    disable_hybrid_kv_cache_manager=False,
)

print("[offline_verify] LLM init done.")

if STOP_AFTER_INIT:
    print("[offline_verify] STOP_AFTER_INIT=1, exit after KV cache allocation/init.")
    raise SystemExit(0)

print("[offline_verify] start first generate, expect prefill + short decode.")
outputs = llm.generate(prompts, sampling_params)
print("[offline_verify] generate done.")

for i, output in enumerate(outputs):
    generated_text = output.outputs[0].text
    print(f"[offline_verify] output[{i}] generated={generated_text!r}")
