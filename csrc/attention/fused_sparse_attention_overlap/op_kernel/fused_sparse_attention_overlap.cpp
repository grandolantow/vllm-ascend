/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Licensed under the Apache License, Version 2.0.
 */

#include "kernel_operator.h"
#include "fused_sparse_attention_overlap_kernel.h"

using namespace AscendC;
using namespace FusedSparseAttentionOverlapNs;

extern "C" __global__ __aicore__ void fused_sparse_attention_overlap(
    GM_ADDR query,
    GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
    GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status,
    GM_ADDR selection_topk_indices,
    GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
    GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq,
    GM_ADDR full_q_actual_seq,
    GM_ADDR hit_mask_out, GM_ADDR miss_indices_out,
    GM_ADDR attention_output, GM_ADDR selection_kv_actual_seq,
    GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    TPipe pipe;
    GET_TILING_DATA(tilingData, tiling);

    if (TILING_KEY_IS(0)) {
        FusedSparseAttentionOverlapOp<DTYPE_QUERY> op;
        op.Init(&pipe, &tilingData,
                query, selection_k_rope, selection_kv_cache,
                selection_kv_block_table, selection_kv_block_status,
                selection_topk_indices,
                full_k_rope, full_kv_cache,
                full_kv_block_table, full_kv_actual_seq, full_q_actual_seq,
                hit_mask_out, miss_indices_out,
                attention_output, selection_kv_actual_seq);
        op.Process();
    }
}
