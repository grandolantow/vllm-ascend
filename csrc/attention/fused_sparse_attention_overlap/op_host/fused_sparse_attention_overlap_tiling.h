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

#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_TILING_H_
#define FUSED_SPARSE_ATTENTION_OVERLAP_TILING_H_

#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"
#include <graph/utils/type_utils.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(FusedSparseAttentionOverlapTilingData)
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, mainCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, tailCoreBsLoopNum);
TILING_DATA_FIELD_DEF(int64_t, selTopKBlockSize);
TILING_DATA_FIELD_DEF(int64_t, sparseBlockSize);
TILING_DATA_FIELD_DEF(int64_t, fullKvBlockNum);
TILING_DATA_FIELD_DEF(int64_t, fullKvBlockSize);
TILING_DATA_FIELD_DEF(int64_t, kRopeDim);
TILING_DATA_FIELD_DEF(int64_t, kvCacheDim);
TILING_DATA_FIELD_DEF(int64_t, selKvBlockNum);
TILING_DATA_FIELD_DEF(int64_t, selKvBlockSize);
TILING_DATA_FIELD_DEF(int64_t, fullMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, selMaxBlockNum);
TILING_DATA_FIELD_DEF(int64_t, batchsize);
TILING_DATA_FIELD_DEF(int64_t, seq);
TILING_DATA_FIELD_DEF(int64_t, rawSeq);
TILING_DATA_FIELD_DEF(int64_t, headnum);
TILING_DATA_FIELD_DEF(int64_t, topk);
TILING_DATA_FIELD_DEF(int64_t, headDim);
TILING_DATA_FIELD_DEF(int64_t, kRopeUbSize);
TILING_DATA_FIELD_DEF(int64_t, kvCacheUbSize);
TILING_DATA_FIELD_DEF(int64_t, gatherQueueUbSize);
TILING_DATA_FIELD_DEF(int64_t, outMissFloatUbSize);
TILING_DATA_FIELD_DEF(int64_t, buffNum);
TILING_DATA_FIELD_DEF(int64_t, layOut);
TILING_DATA_FIELD_DEF(int64_t, sparseMode);
TILING_DATA_FIELD_DEF(int64_t, enableOverlap);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(int64_t, commBufGmOffset);   // GM comm buffer byte offset per core
TILING_DATA_FIELD_DEF(int64_t, mm2OutputGmOffset);  // Mm2 output area byte offset per core
TILING_DATA_FIELD_DEF(int64_t, perCoreWorkspaceSize); // workspace size per core (bytes)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedSparseAttentionOverlap, FusedSparseAttentionOverlapTilingData)

enum class DataLayout : uint32_t {
    BSND = 0,
    TND = 1
};

struct FusedSparseAttentionOverlapCompileInfo {
};

class FusedSparseAttentionOverlapTiling {
public:
    explicit FusedSparseAttentionOverlapTiling(gert::TilingContext* context) : context_(context)
    {}
    ~FusedSparseAttentionOverlapTiling()
    {}
    ge::graphStatus RunTiling();

protected:
    ge::graphStatus DoOpTiling();
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus GetShapeAttrsInfo();
    ge::graphStatus PostTiling();

private:
    ge::graphStatus GetInputAttrs();
    ge::graphStatus GetQueryShape();
    ge::graphStatus GetSelKvCacheShape();
    ge::graphStatus GetSelBlockTable();
    ge::graphStatus GetTopkIndices();
    ge::graphStatus GetFullKvCacheShape();
    ge::graphStatus GetFullKvBlkTable();
    ge::graphStatus GetSeqLenIn();

private:
    FusedSparseAttentionOverlapTilingData tilingData_;
    int64_t selTopKBlockSize_ = 0;
    int64_t sparseBlockSize_ = 0;
    int64_t sparseMode_ = 3;
    float scaleValue_ = 0.0f;

    int64_t coreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t ubBlockSize_ = 0;

    int64_t t_ = 0;
    int64_t batchSize_ = 0;
    int64_t seq_ = 0;
    int64_t headnum_ = 0;
    int64_t topk_ = 0;
    int64_t headDim_ = 0;
    int64_t selKvBlockTableRow_ = 0;
    DataLayout topKLayout_ = DataLayout::BSND;
    ge::DataType queryDtype_ = ge::DT_FLOAT16;
    int64_t dtypeSize_ = 2;

    gert::TilingContext *context_ = nullptr;
};

} // namespace optiling

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_TILING_H_
