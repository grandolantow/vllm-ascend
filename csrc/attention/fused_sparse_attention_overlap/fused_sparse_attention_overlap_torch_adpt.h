/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_TORCH_ADPT_H
#define FUSED_SPARSE_ATTENTION_OVERLAP_TORCH_ADPT_H
namespace vllm_ascend {

at::Tensor npu_fused_sparse_attention_overlap(
    const at::Tensor &query,
    const at::Tensor &selection_k_rope,
    const at::Tensor &selection_kv_cache,
    const at::Tensor &selection_kv_block_table,
    const at::Tensor &selection_kv_block_status,
    const at::Tensor &selection_topk_indices,
    const at::Tensor &full_k_rope,
    const at::Tensor &full_kv_cache,
    const at::Tensor &full_kv_block_table,
    const at::Tensor &full_kv_actual_seq,
    const at::Tensor &full_q_actual_seq,
    double scale_value,
    int64_t sparse_block_size,
    int64_t selection_topk_block_size,
    c10::string_view layout_query,
    c10::string_view layout_kv,
    int64_t sparse_mode)
{
    std::string layout_query_str = std::string(layout_query);
    std::string layout_kv_str = std::string(layout_kv);

    for (size_t i = 0; i < query.sizes().size(); i++) {
        TORCH_CHECK(query.size(i) > 0, "All values within query's shape should be greater "
                                       "than 0, but shape[", i, "] is ", query.size(i));
    }

    // Construct output tensors
    at::Tensor attention_output = at::empty(query.sizes(), query.options().dtype(query.dtype()));

    const int SIZE = 8;
    c10::SmallVector<int64_t, SIZE> actual_seq_shape = {selection_kv_block_table.size(0)};
    at::Tensor selection_kv_actual_seq = at::empty(actual_seq_shape, selection_kv_block_table.options());

    at::Tensor hit_mask = at::empty(selection_topk_indices.sizes(),
                                    selection_topk_indices.options().dtype(torch::kInt32));
    at::Tensor miss_indices = at::empty(selection_topk_indices.sizes(),
                                        selection_topk_indices.options());

    char *layout_query_ptr = const_cast<char *>(layout_query_str.c_str());
    char *layout_kv_ptr = const_cast<char *>(layout_kv_str.c_str());

    EXEC_NPU_CMD(
        aclnnFusedSparseAttentionOverlap,
        query,
        selection_k_rope, selection_kv_cache, selection_kv_block_table,
        selection_kv_block_status, selection_topk_indices,
        full_k_rope, full_kv_cache, full_kv_block_table,
        full_kv_actual_seq, full_q_actual_seq,
        scale_value, sparse_block_size, selection_topk_block_size,
        layout_query_ptr, layout_kv_ptr, sparse_mode,
        hit_mask, miss_indices,
        attention_output, selection_kv_actual_seq);

    return attention_output;
}

}
#endif
