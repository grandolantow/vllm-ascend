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

#include "fused_sparse_attention_overlap_tiling.h"

namespace optiling {

constexpr int32_t INPUT_IDX_QUERY = 0;
constexpr int32_t SEL_K_ROPE_IDX = 1;
constexpr int32_t SEL_KV_CACHE_IDX = 2;
constexpr int32_t SEL_KV_BLOCK_TABLE_IDX = 3;
constexpr int32_t SEL_KV_BLOCK_STAT_IDX = 4;
constexpr int32_t SEL_TOPK_INDICES_IDX = 5;
constexpr int32_t FULL_K_ROPE_IDX = 6;
constexpr int32_t FULL_KV_CACHE_IDX = 7;
constexpr int32_t FULL_KV_BLOCK_TABLE_IDX = 8;
constexpr int32_t FULL_KV_ACTSEQ_IDX = 9;
constexpr int32_t FULL_Q_ACTSEQ_IDX = 10;

constexpr size_t CONST1 = 1;
constexpr size_t CONST2 = 2;
constexpr size_t CONST3 = 3;
constexpr size_t CONST4 = 4;

constexpr int64_t DEFAULT_TOPK_BLOCK_SIZE = 1;
constexpr int64_t DEFAULT_WORKSPACE_SIZE = 32;

template <typename T>
static inline T CeilDiv(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)));
}

template <typename T>
static inline T CeilAlign(T num, T rnd)
{
    return (((rnd) == 0) ? 0 : (((num) + (rnd) - 1) / (rnd)) * (rnd));
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetPlatformInfo()
{
    auto platformInfo = context_->GetPlatformInfo();
    OPS_ERR_IF(platformInfo == nullptr, OPS_LOG_E(context_->GetNodeName(), "get platformInfo nullptr."),
        return ge::GRAPH_FAILED);
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    coreNum_ = ascendcPlatform.GetCoreNumAiv();
    OPS_ERR_IF(coreNum_ <= 0, OPS_LOG_E(context_->GetNodeName(), "coreNum must be greater than 0."),
        return ge::GRAPH_FAILED);

    uint64_t ubSizePlatForm;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSizePlatForm);
    ubSize_ = static_cast<int64_t>(ubSizePlatForm);
    OPS_ERR_IF(ubSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "ubSize must be greater than 0."),
        return ge::GRAPH_FAILED);

    ubBlockSize_ = 32;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetInputAttrs()
{
    auto attrs = context_->GetAttrs();
    OPS_ERR_IF(attrs == nullptr, OPS_LOG_E(context_->GetNodeName(), "get attrs nullptr."),
        return ge::GRAPH_FAILED);

    // attr 0: scale_value (float)
    const float* scalePtr = attrs->GetAttrPointer<float>(0);
    scaleValue_ = (scalePtr != nullptr) ? *scalePtr : 0.041666666666666664f;

    // attr 1: sparse_block_size (int)
    const int64_t* sparseBlockPtr = attrs->GetAttrPointer<int64_t>(1);
    sparseBlockSize_ = (sparseBlockPtr != nullptr && *sparseBlockPtr > 0) ? *sparseBlockPtr : 1;

    // attr 2: selection_topk_block_size (int)
    const int64_t* topkBlockPtr = attrs->GetAttrPointer<int64_t>(2);
    selTopKBlockSize_ = (topkBlockPtr != nullptr && *topkBlockPtr > 0) ? *topkBlockPtr : DEFAULT_TOPK_BLOCK_SIZE;

    // attr 5: sparse_mode (int)
    const int64_t* sparseModePtr = attrs->GetAttrPointer<int64_t>(5);
    sparseMode_ = (sparseModePtr != nullptr) ? *sparseModePtr : 3;

    tilingData_.set_selTopKBlockSize(selTopKBlockSize_);
    tilingData_.set_sparseBlockSize(sparseBlockSize_);
    tilingData_.set_sparseMode(sparseMode_);
    tilingData_.set_enableOverlap(1);
    tilingData_.set_scaleValue(scaleValue_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetQueryShape()
{
    auto queryIn = context_->GetInputShape(INPUT_IDX_QUERY);
    OPS_ERR_IF(queryIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get queryIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape queryShape = queryIn->GetStorageShape();
    size_t dimsN = queryShape.GetDimNum();
    // query: TND [T, N, D] or BSND [B, S, N, D]
    if (dimsN == CONST3) {
        headDim_ = queryShape.GetDim(CONST2);
    } else if (dimsN == CONST4) {
        headDim_ = queryShape.GetDim(CONST3);
    } else {
        OPS_LOG_E(context_->GetNodeName(), "query dim:%lu should be 3 or 4.", dimsN);
        return ge::GRAPH_FAILED;
    }
    tilingData_.set_headDim(headDim_);

    // Get query dtype to derive element size for UB allocation
    auto queryDesc = context_->GetInputDesc(INPUT_IDX_QUERY);
    OPS_ERR_IF(queryDesc == nullptr, OPS_LOG_E(context_->GetNodeName(), "get queryDesc nullptr."),
        return ge::GRAPH_FAILED);
    queryDtype_ = queryDesc->GetDataType();
    dtypeSize_ = static_cast<int64_t>(ge::GetSizeByDataType(queryDtype_));
    OPS_ERR_IF(dtypeSize_ <= 0, OPS_LOG_E(context_->GetNodeName(), "unsupported query dtype, size=%ld.", dtypeSize_),
        return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetSelKvCacheShape()
{
    auto selKRopeIn = context_->GetInputShape(SEL_K_ROPE_IDX);
    OPS_ERR_IF(selKRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKRopeIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKRopeShape = selKRopeIn->GetStorageShape();

    auto selKvCacheIn = context_->GetInputShape(SEL_KV_CACHE_IDX);
    OPS_ERR_IF(selKvCacheIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvCacheIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvCacheShape = selKvCacheIn->GetStorageShape();
    size_t dimsN = selKvCacheShape.GetDimNum();
    OPS_ERR_IF(dimsN != CONST3,
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_cache dim:%lu should be 3.", dimsN),
        return ge::GRAPH_FAILED);
    tilingData_.set_selKvBlockNum(selKvCacheShape.GetDim(0));
    tilingData_.set_selKvBlockSize(selKvCacheShape.GetDim(1));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetSelBlockTable()
{
    auto selKvBlkTIn = context_->GetInputShape(SEL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(selKvBlkTIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBlkTIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvBlkTInShape = selKvBlkTIn->GetStorageShape();
    size_t dimsN = selKvBlkTInShape.GetDimNum();
    OPS_ERR_IF(dimsN != CONST2,
        OPS_LOG_E(context_->GetNodeName(), "selection_kv_block_table dim:%lu should be 2.", dimsN),
        return ge::GRAPH_FAILED);
    tilingData_.set_selMaxBlockNum(selKvBlkTInShape.GetDim(1));
    selKvBlockTableRow_ = selKvBlkTInShape.GetDim(0);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetTopkIndices()
{
    auto selKvBlkStIn = context_->GetInputShape(SEL_KV_BLOCK_STAT_IDX);
    OPS_ERR_IF(selKvBlkStIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selKvBlkStIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selKvBlkStShape = selKvBlkStIn->GetStorageShape();

    auto selTopKIn = context_->GetInputShape(SEL_TOPK_INDICES_IDX);
    OPS_ERR_IF(selTopKIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get selTopKIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape selTopKInShape = selTopKIn->GetStorageShape();
    size_t dimsN = selTopKInShape.GetDimNum();
    OPS_ERR_IF(dimsN != CONST4 && dimsN != CONST3,
        OPS_LOG_E(context_->GetNodeName(), "selection_topk_indices dim:%lu should be 3 or 4.", dimsN),
        return ge::GRAPH_FAILED);

    if (dimsN == CONST4) {
        topKLayout_ = DataLayout::BSND;
        batchSize_ = selTopKInShape.GetDim(0);
        seq_ = selTopKInShape.GetDim(CONST1);
        headnum_ = selTopKInShape.GetDim(CONST2);
        topk_ = selTopKInShape.GetDim(CONST3);
    } else {
        topKLayout_ = DataLayout::TND;
        t_ = selTopKInShape.GetDim(0);
        headnum_ = selTopKInShape.GetDim(CONST1);
        topk_ = selTopKInShape.GetDim(CONST2);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetFullKvCacheShape()
{
    auto fulKRopeIn = context_->GetInputShape(FULL_K_ROPE_IDX);
    OPS_ERR_IF(fulKRopeIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKRopeIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKRopeShape = fulKRopeIn->GetStorageShape();
    size_t dimsNFullKRope = fulKRopeShape.GetDimNum();

    auto fulKvCacheIn = context_->GetInputShape(FULL_KV_CACHE_IDX);
    OPS_ERR_IF(fulKvCacheIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvCacheIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKvCacheInShape = fulKvCacheIn->GetStorageShape();
    OPS_ERR_IF(fulKvCacheInShape.GetDimNum() != CONST3,
        OPS_LOG_E(context_->GetNodeName(), "full_kv_cache dim should be 3."),
        return ge::GRAPH_FAILED);

    tilingData_.set_kvCacheDim(fulKvCacheInShape.GetDim(CONST2));

    if (dimsNFullKRope == CONST3) {
        tilingData_.set_fullKvBlockNum(fulKRopeShape.GetDim(0));
        tilingData_.set_fullKvBlockSize(fulKRopeShape.GetDim(1));
        tilingData_.set_kRopeDim(fulKRopeShape.GetDim(CONST2));
    } else {
        tilingData_.set_fullKvBlockNum(fulKvCacheInShape.GetDim(0));
        tilingData_.set_fullKvBlockSize(fulKvCacheInShape.GetDim(1));
        tilingData_.set_kRopeDim(0);
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetFullKvBlkTable()
{
    auto fulKvBlkTIn = context_->GetInputShape(FULL_KV_BLOCK_TABLE_IDX);
    OPS_ERR_IF(fulKvBlkTIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvBlkTIn nullptr."),
        return ge::GRAPH_FAILED);
    gert::Shape fulKvBlkTInShape = fulKvBlkTIn->GetStorageShape();
    OPS_ERR_IF(fulKvBlkTInShape.GetDimNum() != CONST2,
        OPS_LOG_E(context_->GetNodeName(), "full_kv_block_table dim should be 2."),
        return ge::GRAPH_FAILED);

    tilingData_.set_fullMaxBlockNum(fulKvBlkTInShape.GetDim(1));
    if (topKLayout_ == DataLayout::TND) {
        batchSize_ = fulKvBlkTInShape.GetDim(0);
        OPS_ERR_IF(batchSize_ == 0 || t_ % batchSize_ != 0,
            OPS_LOG_E(context_->GetNodeName(), "TND format t_:%ld must be multiple of batchSize_:%ld", t_, batchSize_),
            return ge::GRAPH_FAILED);
        seq_ = t_ / batchSize_;
    }

    tilingData_.set_rawSeq(seq_);
    tilingData_.set_headnum(headnum_);
    tilingData_.set_topk(topk_);
    tilingData_.set_layOut(static_cast<int64_t>(topKLayout_));
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetSeqLenIn()
{
    auto fulKvSeqIn = context_->GetInputShape(FULL_KV_ACTSEQ_IDX);
    OPS_ERR_IF(fulKvSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulKvSeqIn nullptr."),
        return ge::GRAPH_FAILED);
    auto fulQSeqIn = context_->GetInputShape(FULL_Q_ACTSEQ_IDX);
    OPS_ERR_IF(fulQSeqIn == nullptr, OPS_LOG_E(context_->GetNodeName(), "get fulQSeqIn nullptr."),
        return ge::GRAPH_FAILED);

    batchSize_ = batchSize_ * seq_;
    seq_ = 1;
    tilingData_.set_batchsize(batchSize_);
    tilingData_.set_seq(seq_);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::GetShapeAttrsInfo()
{
    OPS_ERR_IF(context_ == nullptr, OPS_LOG_E("FusedSparseAttentionOverlap", "context can not be nullptr."),
        return ge::GRAPH_FAILED);

    if (GetInputAttrs() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    if (GetQueryShape() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    if (GetSelKvCacheShape() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    if (GetSelBlockTable() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    if (GetTopkIndices() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    if (GetFullKvCacheShape() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    if (GetFullKvBlkTable() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    if (GetSeqLenIn() != ge::GRAPH_SUCCESS) { return ge::GRAPH_FAILED; }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::DoOpTiling()
{
    if (batchSize_ == 0) {
        tilingData_.set_usedCoreNum(0);
        context_->SetBlockDim(1);
        return ge::GRAPH_SUCCESS;
    }

    int64_t bsCoreFactor = CeilDiv(batchSize_, static_cast<int64_t>(coreNum_));
    int64_t bsCoreNum = CeilDiv(batchSize_, bsCoreFactor);
    tilingData_.set_usedCoreNum(bsCoreNum);

    // MIX_AIC_1_2: use CalcTschBlockDim like SFA does
    // Always use all available cores; kernel checks blkIdx_ >= usedCoreNum to skip idle blocks
    auto ascendcPlatformBD = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    uint32_t aicNum = ascendcPlatformBD.GetCoreNumAic();
    uint32_t aivNum = ascendcPlatformBD.GetCoreNumAiv();
    int64_t blockDim = ascendcPlatformBD.CalcTschBlockDim(aivNum, aicNum, aivNum);

    bsCoreFactor = CeilDiv(batchSize_, bsCoreNum);
    int64_t tailCoreBsFactor = batchSize_ - (bsCoreNum - 1) * bsCoreFactor;
    tilingData_.set_mainCoreBsLoopNum(bsCoreFactor);
    tilingData_.set_tailCoreBsLoopNum(tailCoreBsFactor);

    int64_t SH = seq_ * headnum_;
    int64_t selTopKUb = CeilAlign(SH * topk_ * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    int64_t topkStatUb = CeilAlign(SH * (topk_ + 1) * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    int64_t selKvBlockTabUbSize = CeilAlign(SH * tilingData_.get_selMaxBlockNum() *
                                            static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);
    int64_t selKvSeqLenUbSize = CeilAlign(SH * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_);

    // kRope and kvCache UB sizes (use actual dtype size from query input)
    int64_t kRopeUbSize = CeilAlign(selTopKBlockSize_ * tilingData_.get_kRopeDim() * dtypeSize_, ubBlockSize_);
    int64_t kvCacheUbSize = CeilAlign(selTopKBlockSize_ * tilingData_.get_kvCacheDim() * dtypeSize_, ubBlockSize_);

    // gather queue: same element size as kvCache queue, independent for overlap
    int64_t gatherQueueUbSize = kvCacheUbSize + kRopeUbSize;

    // outMissFloat: float buffer for miss-segment attention accumulation (kvCacheDim floats)
    int64_t outMissFloatUbSize = CeilAlign(tilingData_.get_kvCacheDim() * static_cast<int64_t>(sizeof(float)),
                                           static_cast<int64_t>(ubBlockSize_));

    tilingData_.set_kRopeUbSize(kRopeUbSize);
    tilingData_.set_kvCacheUbSize(kvCacheUbSize);
    tilingData_.set_gatherQueueUbSize(gatherQueueUbSize);
    tilingData_.set_outMissFloatUbSize(outMissFloatUbSize);
    tilingData_.set_buffNum(CONST2);

    // UB total usage check
    int64_t kvCacheQueueTotal = (kvCacheUbSize + kRopeUbSize) * static_cast<int64_t>(CONST2);
    int64_t gatherQueueTotal = gatherQueueUbSize * static_cast<int64_t>(CONST2);
    int64_t selTopKIdxTotal = selTopKUb;
    int64_t workBufTotal = selKvBlockTabUbSize + selKvSeqLenUbSize + topkStatUb +
        CeilAlign(topk_ * static_cast<int64_t>(sizeof(int32_t)), ubBlockSize_) +  // hitFlagSize
        CeilAlign(topk_ * static_cast<int64_t>(sizeof(int32_t)) * 4, ubBlockSize_);  // sortBufSize
    int64_t kvDimFloatAlign = CeilAlign(tilingData_.get_kvCacheDim() * static_cast<int64_t>(sizeof(float)), ubBlockSize_);
    int64_t ropeFloatAlign = CeilAlign(tilingData_.get_kRopeDim() * static_cast<int64_t>(sizeof(float)), ubBlockSize_);
    int64_t expBufSize = kvDimFloatAlign;  // large enough for Mul(Q*K) + tree reduce
    if (expBufSize < ubBlockSize_) expBufSize = ubBlockSize_;
    int64_t attnBufTotal = kvDimFloatAlign * 2 + outMissFloatUbSize + kvDimFloatAlign + ropeFloatAlign;  // queryFloat + outHitFloat + outMissFloat + tmpFloat + queryRopeFloat (expBuf is separate)
    int64_t totalUbUsage = kvCacheQueueTotal + gatherQueueTotal + selTopKIdxTotal + workBufTotal + attnBufTotal + expBufSize + 5 * ubBlockSize_;  // +5*BLOCK_BYTES for scoreBuf_, runMaxBuf_, runSumBuf_, oldMaxBuf_, reduceTmpBuf_
    OPS_ERR_IF(totalUbUsage > ubSize_,
        OPS_LOG_E(context_->GetNodeName(),
            "UB usage %ld exceeds capacity %ld (kvCacheQ=%ld, gatherQ=%ld, topkIdx=%ld, work=%ld, attn=%ld).",
            totalUbUsage, ubSize_, kvCacheQueueTotal, gatherQueueTotal, selTopKIdxTotal, workBufTotal, attnBufTotal),
        return ge::GRAPH_FAILED);

    // MIX_AIC_1_2 requires blockDim >= 2 (already computed above)
    context_->SetBlockDim(blockDim);
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::PostTiling()
{
    context_->SetTilingKey(0);
    size_t* workspaces = context_->GetWorkspaceSizes(1);
    OPS_ERR_IF(workspaces == nullptr, OPS_LOG_E(context_->GetNodeName(), "get workspaces nullptr."),
        return ge::GRAPH_FAILED);
    // workspace for intermediate attention scores and values (per-core isolated)
    int64_t BSH = batchSize_ * seq_ * headnum_;
    int64_t bsCoreNum = tilingData_.get_usedCoreNum();  // already >= 2 (forced in DoOpTiling)
    int64_t bsPerCore = tilingData_.get_mainCoreBsLoopNum();
    int64_t SH_perCore = bsPerCore * headnum_;

    // Per-core workspace layout:
    // [0 .. scoresSize)           : Mm1 scores (float), topk floats per head
    // [scoresSize .. +weightsSize): softmax weights in T format for Mm2
    // [+weightsSize .. +mm2Size)  : Mm2 output accumulation (float), kvCacheDim per head
    // [+mm2Size .. +commBufSize)  : GM comm buffer (1088 bytes, 32B aligned)
    // Mm1 Fixpipe outputs [mSizeAlign, nSizeAlign] floats where mSizeAlign=16 (M=1 padded).
    // We need space for the full Fixpipe output, not just topk floats.
    // nSizeAlign = CeilAlign(topk, 16) per N-block, but we allocate for the worst case
    // where all topk tokens are in one N-block (topk <= FSA_N_SPLIT_SIZE=128).
    // For topk > 128, tokens are split into multiple blocks, each <= 128.
    constexpr int64_t MM1_M_SIZE_ALIGN = 16;
    int64_t nSizeAlign = ((topk_ + 15) / 16) * 16;
    int64_t perCoreScoresSize = SH_perCore * MM1_M_SIZE_ALIGN * nSizeAlign * static_cast<int64_t>(sizeof(float));
    // Softmax weights are Cast to T format with columnCount = CeilAlign(nTokens, 128).
    // For topk <= 128, this is 128 elements. For topk > 128, each block has <= 128.
    // Allocate for the padded size.
    int64_t weightsColumnAlign = ((topk_ + 127) / 128) * 128;
    if (weightsColumnAlign < 128) weightsColumnAlign = 128;
    int64_t perCoreWeightsSize = SH_perCore * weightsColumnAlign * dtypeSize_;
    // Mm2 Fixpipe outputs [mSizeAlign, outNSizeAlign] floats where mSizeAlign=16.
    // outNSizeAlign = CeilAlign(kvCacheDim, 16). We need space for the full Fixpipe output.
    int64_t kvDimAlign = ((headDim_ + 15) / 16) * 16;  // 16-element aligned for Fixpipe
    int64_t perCoreMm2Size = SH_perCore * MM1_M_SIZE_ALIGN * kvDimAlign * static_cast<int64_t>(sizeof(float));
    int64_t commBufSize = 1088;  // 32B aligned comm buffer
    int64_t diagBufSize = 256 * static_cast<int64_t>(sizeof(float));  // 256 floats for diagnostics

    int64_t perCoreSize = perCoreScoresSize + perCoreWeightsSize + perCoreMm2Size + commBufSize + diagBufSize;
    tilingData_.set_commBufGmOffset(perCoreScoresSize + perCoreWeightsSize + perCoreMm2Size);
    tilingData_.set_mm2OutputGmOffset(perCoreScoresSize + perCoreWeightsSize);
    tilingData_.set_perCoreWorkspaceSize(perCoreSize);

    workspaces[0] = static_cast<size_t>(bsCoreNum * perCoreSize + DEFAULT_WORKSPACE_SIZE);

    OPS_ERR_IF(context_->GetRawTilingData() == nullptr,
        OPS_LOG_E(context_->GetNodeName(), "get tilingdata nullptr."),
        return ge::GRAPH_FAILED);
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus FusedSparseAttentionOverlapTiling::RunTiling()
{
    ge::graphStatus ret = GetShapeAttrsInfo();
    if (ret != ge::GRAPH_SUCCESS) { return ret; }
    ret = GetPlatformInfo();
    if (ret != ge::GRAPH_SUCCESS) { return ret; }
    ret = DoOpTiling();
    if (ret != ge::GRAPH_SUCCESS) { return ret; }
    return PostTiling();
}

ge::graphStatus Tiling4FusedSparseAttentionOverlap(gert::TilingContext* context)
{
    OPS_LOG_I(context->GetNodeName(), "TilingForFusedSparseAttentionOverlap running.");
    FusedSparseAttentionOverlapTiling tiling(context);
    return tiling.RunTiling();
}

ge::graphStatus TilingPrepare4FusedSparseAttentionOverlap(gert::TilingParseContext* context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(FusedSparseAttentionOverlap)
    .Tiling(Tiling4FusedSparseAttentionOverlap)
    .TilingParse<FusedSparseAttentionOverlapCompileInfo>(TilingPrepare4FusedSparseAttentionOverlap);

} // namespace optiling
