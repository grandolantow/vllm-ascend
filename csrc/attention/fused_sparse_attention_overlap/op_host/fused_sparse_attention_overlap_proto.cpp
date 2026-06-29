/**
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

#include <graph/utils/type_utils.h>
#include <register/op_impl_registry.h>
#include "error/ops_error.h"

using namespace ge;
namespace ops {

constexpr size_t INPUT_IDX_QUERY = 0;
constexpr size_t INPUT_IDX_SELECTION_K_ROPE = 1;
constexpr size_t INPUT_IDX_SELECTION_KV_CACHE = 2;
constexpr size_t INPUT_IDX_SELECTION_KV_BLOCK_TABLE = 3;
constexpr size_t INPUT_IDX_SELECTION_KV_BLOCK_STATUS = 4;
constexpr size_t INPUT_IDX_SELECTION_TOPK_INDICES = 5;

// Output indices - must match def.cpp order
constexpr size_t OUTPUT_IDX_HIT_MASK = 0;
constexpr size_t OUTPUT_IDX_MISS_INDICES = 1;
constexpr size_t OUTPUT_IDX_ATTENTION_OUTPUT = 2;
constexpr size_t OUTPUT_IDX_SELECTION_KV_ACTUAL_SEQ = 3;

static ge::graphStatus InferShape4FusedSparseAttentionOverlap(gert::InferShapeContext* context)
{
    const gert::Shape* queryShape = context->GetInputShape(INPUT_IDX_QUERY);
    OPS_LOG_E_IF_NULL(context, queryShape, return ge::GRAPH_FAILED);
    auto attentionOutputShape = context->GetOutputShape(OUTPUT_IDX_ATTENTION_OUTPUT);
    *attentionOutputShape = *queryShape;

    const gert::Shape* selKvBlockTableShape = context->GetInputShape(INPUT_IDX_SELECTION_KV_BLOCK_TABLE);
    OPS_LOG_E_IF_NULL(context, selKvBlockTableShape, return ge::GRAPH_FAILED);
    gert::Shape* selKvActualSeqShape = context->GetOutputShape(OUTPUT_IDX_SELECTION_KV_ACTUAL_SEQ);
    OPS_LOG_E_IF_NULL(context, selKvActualSeqShape, return ge::GRAPH_FAILED);
    *selKvActualSeqShape = *selKvBlockTableShape;
    int64_t dim = static_cast<int64_t>(selKvBlockTableShape->GetDimNum());
    selKvActualSeqShape->SetDimNum(dim - 1);

    const gert::Shape* selTopkIndicesShape = context->GetInputShape(INPUT_IDX_SELECTION_TOPK_INDICES);
    OPS_LOG_E_IF_NULL(context, selTopkIndicesShape, return ge::GRAPH_FAILED);
    auto hitMaskShape = context->GetOutputShape(OUTPUT_IDX_HIT_MASK);
    *hitMaskShape = *selTopkIndicesShape;
    auto missIndicesShape = context->GetOutputShape(OUTPUT_IDX_MISS_INDICES);
    *missIndicesShape = *selTopkIndicesShape;

    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus InferDtype4FusedSparseAttentionOverlap(gert::InferDataTypeContext* context)
{
    const auto query_dtype = context->GetInputDataType(INPUT_IDX_QUERY);
    const auto topk_indices_dtype = context->GetInputDataType(INPUT_IDX_SELECTION_TOPK_INDICES);

    context->SetOutputDataType(OUTPUT_IDX_HIT_MASK, topk_indices_dtype);
    context->SetOutputDataType(OUTPUT_IDX_MISS_INDICES, topk_indices_dtype);
    context->SetOutputDataType(OUTPUT_IDX_ATTENTION_OUTPUT, query_dtype);
    context->SetOutputDataType(OUTPUT_IDX_SELECTION_KV_ACTUAL_SEQ, ge::DT_INT32);

    return GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(FusedSparseAttentionOverlap)
    .InferShape(InferShape4FusedSparseAttentionOverlap)
    .InferDataType(InferDtype4FusedSparseAttentionOverlap);
}  // namespace ops
