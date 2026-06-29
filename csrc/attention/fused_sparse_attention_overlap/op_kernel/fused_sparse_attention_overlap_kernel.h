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

#ifndef FUSED_SPARSE_ATTENTION_OVERLAP_KERNEL_H_
#define FUSED_SPARSE_ATTENTION_OVERLAP_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/tiling.h"

using namespace AscendC;

namespace FusedSparseAttentionOverlapNs {

constexpr int32_t BLOCK_BYTES = 32;
constexpr int32_t ONE_REPEAT_SORT_NUM = 32;
constexpr int32_t COMPARE_SCALAR_NUM = 256 / sizeof(int32_t);
constexpr int32_t SORT_BUF_FACTOR = 4;
constexpr int32_t HIT_FLAG = -2;
constexpr int32_t CHUNK_SIZE = 64;  // max slots per chunk for overlap processing (fits 32KB stack)
constexpr float NEG_INF = -3.4028235e+38f;  // -FLT_MAX, replaces -INFINITY for AscendC kernel

template <typename T>
class FusedSparseAttentionOverlapOp {
public:
    __aicore__ inline FusedSparseAttentionOverlapOp() {}

    __aicore__ inline void Init(
        TPipe* pipeIn,
        const FusedSparseAttentionOverlapTilingData* tilingIn,
        GM_ADDR query, GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
        GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status,
        GM_ADDR selection_topk_indices,
        GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
        GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq,
        GM_ADDR full_q_actual_seq,
        GM_ADDR hit_mask_out, GM_ADDR miss_indices_out,
        GM_ADDR attention_output, GM_ADDR selection_kv_actual_seq);
    __aicore__ inline void Process();

private:
    // --- Hit detection ---
    __aicore__ inline void CopyInTopKIndices(int64_t bsIdx);
    __aicore__ inline void CopyInBlockStatus(int64_t bsIdx, LocalTensor<int32_t>& statLocal);
    __aicore__ inline void CopyInSelKvBlockTable(int64_t bsIdx, LocalTensor<int32_t>& blkTableLocal);

    __aicore__ inline void DetectHitsForHead(
        int64_t hnIdx, LocalTensor<int32_t>& topkLocal,
        LocalTensor<int32_t>& statLocal, LocalTensor<int32_t>& hitFlagLocal);

    // --- Core processing per head: gather misses + compute attention ---
    __aicore__ inline void ComputeSlotScore(
        int64_t selKvCacheAddr, int64_t selKRopeAddr,
        LocalTensor<float>& queryFloat, LocalTensor<float>& queryRopeFloat,
        LocalTensor<float>& tmpFloat,
        int64_t kvDimFloatAlign, int64_t kRopeDimFloatAlign,
        LocalTensor<float>& scoreOut);

    // --- Process a single token's attention (score + softmax update + V accumulate) ---
    __aicore__ inline void ProcessSingleToken(
        int64_t tokenKvAddr, int64_t tokenRopeAddr,
        LocalTensor<float>& queryFloat, LocalTensor<float>& queryRopeFloat,
        LocalTensor<float>& tmpFloat, LocalTensor<float>& outHitFloat,
        LocalTensor<float>& scoreOut, LocalTensor<float>& runMaxTensor,
        LocalTensor<float>& runSumTensor, LocalTensor<float>& oldMaxTensor,
        LocalTensor<float>& expLocal, LocalTensor<float>& brcbFloat,
        int64_t kvDimFloatAlign, int64_t kRopeDimFloatAlign);

    // --- Core processing per head: hit detect + overlapped gather/attention ---
    __aicore__ inline void ProcessOneHead(
        int64_t bsIdx, int64_t seqIdx, int64_t hnIdx, int64_t curFullKvSeqModify,
        LocalTensor<int32_t>& statLocal, LocalTensor<int32_t>& topkLocal,
        LocalTensor<int32_t>& blkTableLocal, LocalTensor<int32_t>& seqLocal);

    // --- Overlap gather helper: issues a single async gather via gatherQue_ ---
    __aicore__ inline void IssueSingleGather(
        int32_t missIdx, int64_t* missGatherSizes,
        int64_t* missKvAddrs, int64_t* missKRopeAddrs,
        int64_t* missFullKvAddrs, int64_t* missFullKRopeAddrs);

    __aicore__ inline void CopyOutResults(
        int64_t bsIdx, LocalTensor<int32_t>& seqLocal,
        LocalTensor<int32_t>& statLocal);

    // --- Helpers ---
    template <HardEvent event>
    __aicore__ inline void SyncEvent(HardEvent evt)
    {
        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<event>(eventId);
        WaitFlag<event>(eventId);
    }

    template <typename U>
    __aicore__ inline U CeilAlign(U a, U b) { return (b == 0) ? 0 : ((a + b - 1) / b * b); }

    template <typename U>
    __aicore__ inline U CeilDiv(U a, U b) { return (b == 0) ? 0 : ((a + b - 1) / b); }

    // --- Members ---
    TPipe* pipe_ = nullptr;
    const FusedSparseAttentionOverlapTilingData* tiling_ = nullptr;

    int32_t blkIdx_ = -1;
    int64_t bsLoopNum_ = 0;
    int64_t rawSeq_ = 0;
    int64_t topkAlign_ = 0;
    int64_t topkSortAlign_ = 0;
    int64_t topkOneAlign_ = 0;
    int64_t topkOneSortAlign_ = 0;
    int64_t headDimAlign_ = 0;
    int32_t kRopeUbOffset_ = 0;

    // UB size tracking
    int32_t selTopKIdxUbSize_ = 0;
    int32_t selBlockStatUbSize_ = 0;
    int32_t selKvBlockTableUbSize_ = 0;
    int32_t selKvActSeqUbSize_ = 0;

    // GM tensors
    GlobalTensor<T> queryGm_;
    GlobalTensor<T> selKRopeGm_;
    GlobalTensor<T> selKvCacheGm_;
    GlobalTensor<int32_t> selKvBlockTableGm_;
    GlobalTensor<int32_t> selKvBlockStatusGm_;
    GlobalTensor<int32_t> selTopKIndicesGm_;
    GlobalTensor<T> fullKRopeGm_;
    GlobalTensor<T> fullKvCacheGm_;
    GlobalTensor<int32_t> fullKvBlockTableGm_;
    GlobalTensor<int32_t> fullKvActualSeqGm_;
    GlobalTensor<int32_t> fullQActualSeqGm_;
    GlobalTensor<int32_t> hitMaskOutGm_;
    GlobalTensor<int32_t> missIndicesOutGm_;
    GlobalTensor<T> attentionOutGm_;
    GlobalTensor<int32_t> selKvActualSeqGm_;

    // UB queues and buffers
    TQue<QuePosition::VECIN, 1> selTopKIdxQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> kvCacheQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 2> gatherQue_;  // independent queue for overlap gather
    TBuf<QuePosition::VECCALC> workBuf_;
    TBuf<QuePosition::VECCALC> attnBuf_;
    TBuf<QuePosition::VECCALC> expBuf_;      // for Exp/reduce operations (kvDimFloatAlign floats)
    TBuf<QuePosition::VECCALC> brcbBuf_;     // broadcast buffer for Brcb + Mul pattern
    TBuf<QuePosition::VECCALC> scoreBuf_;    // score output (BLOCK_FLOATS)
    TBuf<QuePosition::VECCALC> runMaxBuf_;   // running max (BLOCK_FLOATS)
    TBuf<QuePosition::VECCALC> runSumBuf_;   // running sum (BLOCK_FLOATS)
    TBuf<QuePosition::VECCALC> oldMaxBuf_;   // old max temp (BLOCK_FLOATS)
    TBuf<QuePosition::VECCALC> reduceTmpBuf_; // tmpBuffer for ReduceSum API (BLOCK_BYTES)

    // Work buffer sub-tensors (set in Process)
    LocalTensor<int32_t> hitFlagLocal_;
};

} // namespace FusedSparseAttentionOverlapNs

// ============================================================================
// Implementation
// ============================================================================
namespace FusedSparseAttentionOverlapNs {

template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::Init(
    TPipe* pipeIn,
    const FusedSparseAttentionOverlapTilingData* tilingIn,
    GM_ADDR query, GM_ADDR selection_k_rope, GM_ADDR selection_kv_cache,
    GM_ADDR selection_kv_block_table, GM_ADDR selection_kv_block_status,
    GM_ADDR selection_topk_indices,
    GM_ADDR full_k_rope, GM_ADDR full_kv_cache,
    GM_ADDR full_kv_block_table, GM_ADDR full_kv_actual_seq,
    GM_ADDR full_q_actual_seq,
    GM_ADDR hit_mask_out, GM_ADDR miss_indices_out,
    GM_ADDR attention_output, GM_ADDR selection_kv_actual_seq)
{
    pipe_ = pipeIn;
    tiling_ = tilingIn;
    blkIdx_ = GetBlockIdx();
    if (blkIdx_ >= tiling_->usedCoreNum) return;

    rawSeq_ = tiling_->rawSeq;
    int64_t SH = rawSeq_ * tiling_->headnum;

    topkAlign_ = CeilAlign(static_cast<int64_t>(tiling_->topk),
                           static_cast<int64_t>(BLOCK_BYTES / sizeof(int32_t)));
    topkSortAlign_ = CeilAlign(static_cast<int32_t>(tiling_->topk), ONE_REPEAT_SORT_NUM);
    topkOneAlign_ = CeilAlign(static_cast<int64_t>(tiling_->topk + 1),
                              static_cast<int64_t>(BLOCK_BYTES / sizeof(int32_t)));
    topkOneSortAlign_ = topkSortAlign_ > topkOneAlign_ ? topkSortAlign_ : topkOneAlign_;
    headDimAlign_ = CeilAlign(tiling_->headDim,
                              static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));

    kRopeUbOffset_ = tiling_->kvCacheUbSize / sizeof(T);
    pipe_->InitBuffer(kvCacheQue_, tiling_->buffNum,
                      tiling_->kvCacheUbSize + tiling_->kRopeUbSize);
    // Independent gather queue for overlap: decoupled from kvCacheQue_ to avoid FIFO conflicts
    pipe_->InitBuffer(gatherQue_, tiling_->buffNum, tiling_->gatherQueueUbSize);

    selTopKIdxUbSize_ = SH * topkSortAlign_ * sizeof(int32_t);
    pipe_->InitBuffer(selTopKIdxQue_, 1, selTopKIdxUbSize_);

    selKvBlockTableUbSize_ = CeilAlign(
        static_cast<int64_t>(SH * tiling_->selMaxBlockNum * sizeof(int32_t)),
        static_cast<int64_t>(BLOCK_BYTES));
    selKvActSeqUbSize_ = CeilAlign(
        static_cast<int64_t>(SH * sizeof(int32_t)),
        static_cast<int64_t>(BLOCK_BYTES));
    selBlockStatUbSize_ = SH * topkOneSortAlign_ * sizeof(int32_t);
    int64_t hitFlagSize = topkSortAlign_ * sizeof(int32_t);
    int64_t sortBufSize = topkSortAlign_ * sizeof(int32_t) * SORT_BUF_FACTOR;

    pipe_->InitBuffer(workBuf_, selKvBlockTableUbSize_ + selKvActSeqUbSize_ +
                      selBlockStatUbSize_ + hitFlagSize + sortBufSize);

    // Attention buffer: queryFloat(kvCacheDim) + outHitFloat + outMissFloat + tmpFloat + queryRopeFloat(kRopeDim)
    int64_t queryFloatSize = CeilAlign(tiling_->kvCacheDim * static_cast<int64_t>(sizeof(float)),
                                       static_cast<int64_t>(BLOCK_BYTES));
    int64_t outFloatSize = queryFloatSize;
    int64_t outMissFloatSize = tiling_->outMissFloatUbSize;
    int64_t tmpSize = queryFloatSize;
    int64_t ropeFloatSize = CeilAlign(tiling_->kRopeDim * static_cast<int64_t>(sizeof(float)),
                                      static_cast<int64_t>(BLOCK_BYTES));
    pipe_->InitBuffer(attnBuf_, queryFloatSize + outFloatSize + outMissFloatSize + tmpSize + ropeFloatSize);

    // Independent buffer for vector exp / Mul(Q*K) + tree reduce
    int64_t expBufSize = CeilAlign(tiling_->kvCacheDim * static_cast<int64_t>(sizeof(float)),
                                    static_cast<int64_t>(BLOCK_BYTES));
    if (expBufSize < BLOCK_BYTES) expBufSize = BLOCK_BYTES;
    pipe_->InitBuffer(expBuf_, expBufSize);

    // Broadcast buffer for Brcb + Mul pattern (kvDimFloatAlign floats)
    int64_t brcbBufSize = CeilAlign(tiling_->kvCacheDim * static_cast<int64_t>(sizeof(float)),
                                     static_cast<int64_t>(BLOCK_BYTES));
    if (brcbBufSize < BLOCK_BYTES) brcbBufSize = BLOCK_BYTES;
    pipe_->InitBuffer(brcbBuf_, brcbBufSize);

    // Independent TBufs for score, runMax, runSum, oldMax (each BLOCK_BYTES = 8 floats)
    pipe_->InitBuffer(scoreBuf_, BLOCK_BYTES);
    pipe_->InitBuffer(runMaxBuf_, BLOCK_BYTES);
    pipe_->InitBuffer(runSumBuf_, BLOCK_BYTES);
    pipe_->InitBuffer(oldMaxBuf_, BLOCK_BYTES);
    pipe_->InitBuffer(reduceTmpBuf_, expBufSize);  // same size as expBuf_ to prevent ReduceSum overflow

    bsLoopNum_ = (blkIdx_ == tiling_->usedCoreNum - 1)
        ? tiling_->tailCoreBsLoopNum : tiling_->mainCoreBsLoopNum;

    queryGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(query));
    selKRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selection_k_rope));
    selKvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(selection_kv_cache));
    selKvBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_kv_block_table));
    selKvBlockStatusGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_kv_block_status));
    selTopKIndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_topk_indices));
    fullKRopeGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(full_k_rope));
    fullKvCacheGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(full_kv_cache));
    fullKvBlockTableGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(full_kv_block_table));
    fullKvActualSeqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(full_kv_actual_seq));
    fullQActualSeqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(full_q_actual_seq));
    hitMaskOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(hit_mask_out));
    missIndicesOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(miss_indices_out));
    attentionOutGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(attention_output));
    selKvActualSeqGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(selection_kv_actual_seq));
}

// --- Process ---
template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::Process()
{
    if (blkIdx_ >= tiling_->usedCoreNum) return;

    LocalTensor<int32_t> workLocal = workBuf_.Get<int32_t>();
    LocalTensor<int32_t> selKvBlockTableLocal = workLocal;
    LocalTensor<int32_t> selKvActSeqLocal = selKvBlockTableLocal[selKvBlockTableUbSize_ / sizeof(int32_t)];
    LocalTensor<int32_t> selBlockStatLocal = selKvActSeqLocal[selKvActSeqUbSize_ / sizeof(int32_t)];
    hitFlagLocal_ = selBlockStatLocal[selBlockStatUbSize_ / sizeof(int32_t)];

    for (int64_t bsIdx = 0; bsIdx < bsLoopNum_; bsIdx++) {
        int64_t curBatchSize = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
        int64_t curFullKvSeqLen = fullKvActualSeqGm_.GetValue(curBatchSize);
        if (curFullKvSeqLen <= 0) continue;

        CopyInTopKIndices(bsIdx);
        LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.DeQue<int32_t>();

        SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        CopyInBlockStatus(bsIdx, selBlockStatLocal);
        CopyInSelKvBlockTable(bsIdx, selKvBlockTableLocal);
        SyncEvent<HardEvent::MTE2_S>(HardEvent::MTE2_S);

        int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
        int64_t offset = (rawSeq_ - 1) - curSeq;
        int64_t curFullKvSeqModify = curFullKvSeqLen - offset;

        if (curFullKvSeqModify > 0) {
            for (int64_t hnIdx = 0; hnIdx < tiling_->headnum; hnIdx++) {
                LocalTensor<int32_t> tmpTopk = selTopKIdxLocal[hnIdx * topkSortAlign_];
                ProcessOneHead(bsIdx, curSeq, hnIdx, curFullKvSeqModify,
                    selBlockStatLocal, tmpTopk, selKvBlockTableLocal, selKvActSeqLocal);
            }
        }

        selTopKIdxQue_.FreeTensor(selTopKIdxLocal);
        SyncEvent<HardEvent::S_MTE3>(HardEvent::S_MTE3);
        CopyOutResults(bsIdx, selKvActSeqLocal, selBlockStatLocal);
    }
}

// --- DMA Copy Methods (unchanged) ---
template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::CopyInTopKIndices(int64_t bsIdx)
{
    LocalTensor<int32_t> selTopKIdxLocal = selTopKIdxQue_.AllocTensor<int32_t>();
    int64_t SH = rawSeq_ * tiling_->headnum;
    uint8_t padCnt = topkAlign_ - tiling_->topk;
    uint32_t dstStride = (topkSortAlign_ - topkAlign_) / (BLOCK_BYTES / sizeof(int32_t));
    int64_t curBatch = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
    int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;

    DataCopyPadExtParams<int32_t> padParams{true, 0, padCnt, -1};
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(tiling_->headnum),
        static_cast<uint32_t>(tiling_->topk * sizeof(int32_t)), 0, dstStride, 0};
    DataCopyPad(selTopKIdxLocal,
        selTopKIndicesGm_[curBatch * SH * tiling_->topk + curSeq * tiling_->headnum * tiling_->topk],
        copyParams, padParams);
    selTopKIdxQue_.EnQue(selTopKIdxLocal);
}

template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::CopyInBlockStatus(
    int64_t bsIdx, LocalTensor<int32_t>& statLocal)
{
    uint8_t padCnt = topkOneAlign_ - (tiling_->topk + 1);
    uint32_t dstStride = (topkOneSortAlign_ - topkOneAlign_) / (BLOCK_BYTES / sizeof(int32_t));
    int64_t SH = rawSeq_ * tiling_->headnum;
    int64_t curBatch = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
    int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
    int64_t batchOff = curBatch * SH * (tiling_->topk + 1);

    DataCopyPadExtParams<int32_t> padParams{true, 0, padCnt, -1};
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(tiling_->seq * tiling_->headnum),
        static_cast<uint32_t>((tiling_->topk + 1) * sizeof(int32_t)), 0, dstStride, 0};
    DataCopyPad(statLocal,
        selKvBlockStatusGm_[batchOff + curSeq * tiling_->headnum * (tiling_->topk + 1)],
        copyParams, padParams);
}

template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::CopyInSelKvBlockTable(
    int64_t bsIdx, LocalTensor<int32_t>& blkTableLocal)
{
    int64_t curBatch = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
    int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;
    int64_t SH = rawSeq_ * tiling_->headnum;
    int64_t batchOff = curBatch * SH * tiling_->selMaxBlockNum;

    DataCopyPadExtParams<int32_t> padParams{false, 0, 0, 0};
    DataCopyExtParams copyParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(tiling_->seq * tiling_->headnum * tiling_->selMaxBlockNum * sizeof(int32_t)),
        0, 0, 0};
    DataCopyPad(blkTableLocal,
        selKvBlockTableGm_[batchOff + curSeq * tiling_->headnum * tiling_->selMaxBlockNum],
        copyParams, padParams);
}

template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::CopyOutResults(
    int64_t bsIdx, LocalTensor<int32_t>& seqLocal, LocalTensor<int32_t>& statLocal)
{
    int64_t curBatch = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;
    int64_t curSeq = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) % rawSeq_;

    DataCopyExtParams seqParams{
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(tiling_->seq * tiling_->headnum * sizeof(int32_t)), 0, 0, 0};
    DataCopyPad(selKvActualSeqGm_[(curBatch * rawSeq_) * tiling_->headnum + curSeq * tiling_->headnum],
                seqLocal, seqParams);

    int64_t SH = rawSeq_ * tiling_->headnum;
    uint32_t srcStride = (topkOneSortAlign_ - topkOneAlign_) / (BLOCK_BYTES / sizeof(int32_t));
    int64_t batchOff = curBatch * SH * (tiling_->topk + 1);
    DataCopyExtParams statParams{
        static_cast<uint16_t>(tiling_->headnum),
        static_cast<uint32_t>((tiling_->topk + 1) * sizeof(int32_t)), srcStride, 0, 0};
    DataCopyPad(selKvBlockStatusGm_[batchOff + curSeq * tiling_->headnum * (tiling_->topk + 1)],
                statLocal, statParams);
}

// --- Compute attention score for one slot: dot(Q, K_cache) + dot(Q_rope, K_rope) ---
template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::ComputeSlotScore(
    int64_t selKvCacheAddr, int64_t selKRopeAddr,
    LocalTensor<float>& queryFloat, LocalTensor<float>& queryRopeFloat,
    LocalTensor<float>& tmpFloat,
    int64_t kvDimFloatAlign, int64_t kRopeDimFloatAlign,
    LocalTensor<float>& scoreOut)
{
    constexpr int32_t BLOCK_FLOATS = BLOCK_BYTES / static_cast<int32_t>(sizeof(float));
    LocalTensor<float> expLocal = expBuf_.Get<float>();

    // 1. Load K_cache, Cast to tmpFloat
    LocalTensor<T> kvTensor = kvCacheQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyExtParams kvCopyParams{static_cast<uint16_t>(1),
        static_cast<uint32_t>(tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
    DataCopyPad(kvTensor, selKvCacheGm_[selKvCacheAddr], kvCopyParams, padParams);
    kvCacheQue_.EnQue(kvTensor);
    kvTensor = kvCacheQue_.DeQue<T>();
    Cast(tmpFloat, kvTensor, RoundMode::CAST_NONE, tiling_->kvCacheDim);
    pipe_barrier(PIPE_ALL);
    kvCacheQue_.FreeTensor(kvTensor);

    // Zero padding for clean reduction
    if (tiling_->kvCacheDim < kvDimFloatAlign) {
        Duplicate(tmpFloat[tiling_->kvCacheDim], 0.0f, kvDimFloatAlign - tiling_->kvCacheDim);
        pipe_barrier(PIPE_ALL);
    }

    // 2. Element-wise Q*K -> expLocal, then reduce to scalar
    Mul(expLocal, queryFloat, tmpFloat, kvDimFloatAlign);
    pipe_barrier(PIPE_V);

    // Two-stage WholeReduceSum (in-place on expLocal)
    constexpr int32_t REPEAT_FLOATS = 256 / static_cast<int32_t>(sizeof(float));  // 64
    int64_t repeatTimes = kvDimFloatAlign / REPEAT_FLOATS;
    if (repeatTimes > 1) {
        WholeReduceSum(expLocal, expLocal, REPEAT_FLOATS, static_cast<int32_t>(repeatTimes), 1, 8, 8);
        pipe_barrier(PIPE_V);
    }
    WholeReduceSum(expLocal, expLocal, BLOCK_FLOATS, 1, 1, 1, 1);
    pipe_barrier(PIPE_ALL);
    float cacheScore = expLocal.GetValue(0);
    if (cacheScore != cacheScore) cacheScore = 0.0f;
    Duplicate(scoreOut, cacheScore, BLOCK_FLOATS);
    pipe_barrier(PIPE_V);

    // 3. K_rope dot product
    if (tiling_->kRopeDim > 0 && tiling_->headDim > tiling_->kvCacheDim) {
        LocalTensor<T> ropeTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyExtParams ropeCopyParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(ropeTensor, selKRopeGm_[selKRopeAddr], ropeCopyParams, padParams);
        kvCacheQue_.EnQue(ropeTensor);
        ropeTensor = kvCacheQue_.DeQue<T>();
        Cast(tmpFloat, ropeTensor, RoundMode::CAST_NONE, tiling_->kRopeDim);
        pipe_barrier(PIPE_V);
        kvCacheQue_.FreeTensor(ropeTensor);

        if (tiling_->kRopeDim < kRopeDimFloatAlign) {
            Duplicate(tmpFloat[tiling_->kRopeDim], 0.0f, kRopeDimFloatAlign - tiling_->kRopeDim);
            pipe_barrier(PIPE_V);
        }

        Mul(expLocal, queryRopeFloat, tmpFloat, kRopeDimFloatAlign);
        pipe_barrier(PIPE_V);

        int64_t ropeRepeatTimes = kRopeDimFloatAlign / REPEAT_FLOATS;
        if (ropeRepeatTimes > 1) {
            WholeReduceSum(expLocal, expLocal, REPEAT_FLOATS, static_cast<int32_t>(ropeRepeatTimes), 1, 8, 8);
            pipe_barrier(PIPE_V);
        }
        WholeReduceSum(expLocal, expLocal, BLOCK_FLOATS, 1, 1, 1, 1);
        pipe_barrier(PIPE_ALL);
        float ropeVal = expLocal.GetValue(0);
        if (ropeVal != ropeVal) ropeVal = 0.0f;
        Duplicate(expLocal, ropeVal, BLOCK_FLOATS);
        pipe_barrier(PIPE_V);
        Add(scoreOut, scoreOut, expLocal, BLOCK_FLOATS);
        pipe_barrier(PIPE_V);
    }

    // 4. Multiply by scaleValue
    Duplicate(expLocal, tiling_->scaleValue, BLOCK_FLOATS);
    pipe_barrier(PIPE_V);
    Mul(scoreOut, scoreOut, expLocal, BLOCK_FLOATS);
    pipe_barrier(PIPE_V);
}

// --- Process a single token: score + online softmax update + V accumulate ---
template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::ProcessSingleToken(
    int64_t tokenKvAddr, int64_t tokenRopeAddr,
    LocalTensor<float>& queryFloat, LocalTensor<float>& queryRopeFloat,
    LocalTensor<float>& tmpFloat, LocalTensor<float>& outHitFloat,
    LocalTensor<float>& scoreOut, LocalTensor<float>& runMaxTensor,
    LocalTensor<float>& runSumTensor, LocalTensor<float>& oldMaxTensor,
    LocalTensor<float>& expLocal, LocalTensor<float>& brcbFloat,
    int64_t kvDimFloatAlign, int64_t kRopeDimFloatAlign)
{
    constexpr int32_t BLOCK_FLOATS = BLOCK_BYTES / static_cast<int32_t>(sizeof(float));

    ComputeSlotScore(tokenKvAddr, tokenRopeAddr, queryFloat, queryRopeFloat,
                     tmpFloat, kvDimFloatAlign, kRopeDimFloatAlign, scoreOut);

    // Save oldMax
    Adds(oldMaxTensor, runMaxTensor, 0.0f, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    // newMax = max(score, runMax)
    Max(runMaxTensor, scoreOut, runMaxTensor, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    // rescale = exp(clamp(oldMax - newMax, -80))
    Sub(expLocal, oldMaxTensor, runMaxTensor, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    Maxs(expLocal, expLocal, -80.0f, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    Exp(expLocal, expLocal, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    volatile float rescaleVal = expLocal.GetValue(0);
    // outHitFloat *= rescale
    Duplicate(brcbFloat, (float)rescaleVal, kvDimFloatAlign);
    pipe_barrier(PIPE_ALL);
    Mul(outHitFloat, outHitFloat, brcbFloat, kvDimFloatAlign);
    pipe_barrier(PIPE_ALL);
    // runSum *= rescale (both BLOCK_FLOATS, use expLocal directly)
    Mul(runSumTensor, runSumTensor, expLocal, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    // expScore = exp(clamp(score - newMax, -80))
    Sub(expLocal, scoreOut, runMaxTensor, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    Maxs(expLocal, expLocal, -80.0f, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    Exp(expLocal, expLocal, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);
    volatile float expScoreVal = expLocal.GetValue(0);
    // runSum += expScore
    Add(runSumTensor, runSumTensor, expLocal, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);

    // Load V and weight by expScore
    {
        LocalTensor<T> vTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> vPad{false, 0, 0, 0};
        DataCopyExtParams vParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(vTensor, selKvCacheGm_[tokenKvAddr], vParams, vPad);
        kvCacheQue_.EnQue(vTensor);
        vTensor = kvCacheQue_.DeQue<T>();
        Cast(tmpFloat, vTensor, RoundMode::CAST_NONE, tiling_->kvCacheDim);
        pipe_barrier(PIPE_ALL);
        kvCacheQue_.FreeTensor(vTensor);
    }
    Duplicate(brcbFloat, (float)expScoreVal, kvDimFloatAlign);
    pipe_barrier(PIPE_ALL);
    Mul(tmpFloat, tmpFloat, brcbFloat, kvDimFloatAlign);
    pipe_barrier(PIPE_ALL);
    Add(outHitFloat, outHitFloat, tmpFloat, kvDimFloatAlign);
    pipe_barrier(PIPE_ALL);
}

// --- ProcessOneHead: two-phase pipeline with overlap + online softmax ---
template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::ProcessOneHead(
    int64_t bsIdx, int64_t seqIdx, int64_t hnIdx, int64_t curFullKvSeqModify,
    LocalTensor<int32_t>& statLocal, LocalTensor<int32_t>& topkLocal,
    LocalTensor<int32_t>& blkTableLocal, LocalTensor<int32_t>& seqLocal)
{
    int64_t selBlkTableOff = hnIdx * tiling_->selMaxBlockNum;
    int32_t maxSelectionId = CeilDiv(curFullKvSeqModify, tiling_->selTopKBlockSize) - 1;
    int64_t lastGatherBlockSize = curFullKvSeqModify - maxSelectionId * tiling_->selTopKBlockSize;
    int64_t globalBsIdx = blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx;

    // --- Phase 1: Hit detection ---
    Duplicate(hitFlagLocal_, static_cast<int32_t>(-1), topkSortAlign_);
    PipeBarrier<PIPE_V>();

    LocalTensor<int32_t> curStatLocal = statLocal[hnIdx * topkOneSortAlign_];
    DetectHitsForHead(hnIdx, topkLocal, curStatLocal, hitFlagLocal_);

    // Prepare attention buffers (expanded: queryFloat + outHitFloat + outMissFloat + tmpFloat + queryRopeFloat)
    LocalTensor<float> attnLocal = attnBuf_.Get<float>();
    int64_t kvDimFloatAlign = CeilAlign(tiling_->kvCacheDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
    int64_t kRopeDimFloatAlign = CeilAlign(tiling_->kRopeDim, static_cast<int64_t>(BLOCK_BYTES / sizeof(float)));
    int64_t outMissFloatOff = tiling_->outMissFloatUbSize / sizeof(float);
    LocalTensor<float> queryFloat = attnLocal;
    LocalTensor<float> outHitFloat = queryFloat[kvDimFloatAlign];
    LocalTensor<float> outMissFloat = outHitFloat[kvDimFloatAlign];
    LocalTensor<float> tmpFloat = outMissFloat[outMissFloatOff];
    LocalTensor<float> queryRopeFloat = tmpFloat[kvDimFloatAlign];

    // Initialize output accumulators to zero
    Duplicate(outHitFloat, 0.0f, kvDimFloatAlign);
    Duplicate(outMissFloat, 0.0f, kvDimFloatAlign);
    PipeBarrier<PIPE_V>();

    // Load Q (kvCacheDim elements) and cast to float
    {
        int64_t queryGmOff = globalBsIdx * tiling_->headnum * tiling_->headDim + hnIdx * tiling_->headDim;
        LocalTensor<T> inTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(inTensor, queryGm_[queryGmOff], copyParams, padParams);
        kvCacheQue_.EnQue(inTensor);
        inTensor = kvCacheQue_.DeQue<T>();
        Cast(queryFloat, inTensor, RoundMode::CAST_NONE, tiling_->kvCacheDim);
        PipeBarrier<PIPE_V>();
        kvCacheQue_.FreeTensor(inTensor);
    }
    // Load Q_rope only if query contains rope part (headDim > kvCacheDim)
    if (tiling_->kRopeDim > 0 && tiling_->headDim > tiling_->kvCacheDim) {
        int64_t queryRopeGmOff = globalBsIdx * tiling_->headnum * tiling_->headDim
                                 + hnIdx * tiling_->headDim + tiling_->kvCacheDim;
        LocalTensor<T> inTensor = kvCacheQue_.AllocTensor<T>();
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(inTensor, queryGm_[queryRopeGmOff], copyParams, padParams);
        kvCacheQue_.EnQue(inTensor);
        inTensor = kvCacheQue_.DeQue<T>();
        Cast(queryRopeFloat, inTensor, RoundMode::CAST_NONE, tiling_->kRopeDim);
        PipeBarrier<PIPE_V>();
        kvCacheQue_.FreeTensor(inTensor);
    }

    // --- Global state across all chunks (tensor-based, no scalars) ---
    constexpr int32_t BLOCK_FLOATS = BLOCK_BYTES / static_cast<int32_t>(sizeof(float));
    LocalTensor<float> expLocal = expBuf_.Get<float>();
    // Use independent TBufs instead of partitioning expBuf_
    LocalTensor<float> scoreOut = scoreBuf_.Get<float>();
    LocalTensor<float> runMaxTensor = runMaxBuf_.Get<float>();
    LocalTensor<float> runSumTensor = runSumBuf_.Get<float>();
    LocalTensor<float> oldMaxTensor = oldMaxBuf_.Get<float>();
    LocalTensor<float> brcbFloat = brcbBuf_.Get<float>();

    // Initialize running max/sum tensors
    Duplicate(runMaxTensor, NEG_INF, BLOCK_FLOATS);
    Duplicate(runSumTensor, 0.0f, BLOCK_FLOATS);
    pipe_barrier(PIPE_ALL);

    int32_t insertIdx = 0;
    int32_t selActualSeqLen = 0;
    int32_t totalHitCount = 0, totalMissCount = 0;
    int64_t curBatch = (blkIdx_ * tiling_->mainCoreBsLoopNum + bsIdx) / rawSeq_;

    // === Phase 3: Chunked processing ===
    int64_t topkIdx = 0;
    while (topkIdx < tiling_->topk) {
        // 3a. Collect up to CHUNK_SIZE valid slots
        int32_t hitTopKIds[CHUNK_SIZE], missTopKIds[CHUNK_SIZE];
        int64_t hitGatherSizes[CHUNK_SIZE], missGatherSizes[CHUNK_SIZE];
        int64_t hitKvAddrs[CHUNK_SIZE], missKvAddrs[CHUNK_SIZE];
        int64_t hitKRopeAddrs[CHUNK_SIZE], missKRopeAddrs[CHUNK_SIZE];
        int32_t hitCount = 0, missCount = 0;

        for (; topkIdx < tiling_->topk && (hitCount + missCount) < CHUNK_SIZE; topkIdx++) {
            int32_t topKId = topkLocal.GetValue(topkIdx);
            if (topKId < 0 || topKId > maxSelectionId) continue;

            int64_t gatherBlockSize = (topKId == maxSelectionId) ? lastGatherBlockSize : tiling_->selTopKBlockSize;
            selActualSeqLen += gatherBlockSize;

            int64_t selKvBlkTableIdx = (insertIdx * tiling_->selTopKBlockSize) / tiling_->selKvBlockSize;
            int64_t selKvBlkSizeOff = (insertIdx * tiling_->selTopKBlockSize) % tiling_->selKvBlockSize;
            int32_t selKvBlockNumIdx = blkTableLocal.GetValue(selBlkTableOff + selKvBlkTableIdx);
            int64_t selKRopeAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kRopeDim
                                   + selKvBlkSizeOff * tiling_->kRopeDim;
            int64_t selKvCacheAddr = selKvBlockNumIdx * tiling_->selKvBlockSize * tiling_->kvCacheDim
                                     + selKvBlkSizeOff * tiling_->kvCacheDim;

            bool isHit = (hitFlagLocal_.GetValue(topkIdx) == HIT_FLAG);

            int64_t hmOff = globalBsIdx * tiling_->headnum * tiling_->topk + hnIdx * tiling_->topk + topkIdx;
            hitMaskOutGm_.SetValue(hmOff, isHit ? 1 : 0);
            missIndicesOutGm_.SetValue(hmOff, isHit ? -1 : topKId);

            if (isHit) {
                hitTopKIds[hitCount] = topKId;
                hitGatherSizes[hitCount] = gatherBlockSize;
                hitKvAddrs[hitCount] = selKvCacheAddr;
                hitKRopeAddrs[hitCount] = selKRopeAddr;
                hitCount++;
            } else {
                missTopKIds[missCount] = topKId;
                missGatherSizes[missCount] = gatherBlockSize;
                missKvAddrs[missCount] = selKvCacheAddr;
                missKRopeAddrs[missCount] = selKRopeAddr;
                missCount++;
            }

            curStatLocal.SetValue(insertIdx, topKId);
            insertIdx++;
        }

        if (hitCount + missCount == 0) continue;
        totalHitCount += hitCount;
        totalMissCount += missCount;

        // 3b. Pre-compute miss gather source addresses from full_kv_cache
        int64_t missFullKvAddrs[CHUNK_SIZE], missFullKRopeAddrs[CHUNK_SIZE];
        for (int32_t m = 0; m < missCount; m++) {
            int64_t kvBlkTableIdx = (missTopKIds[m] * tiling_->selTopKBlockSize) / tiling_->fullKvBlockSize;
            int64_t kvBlkSizeOff = (missTopKIds[m] * tiling_->selTopKBlockSize) % tiling_->fullKvBlockSize;
            int32_t kvBlockNumIdx = fullKvBlockTableGm_.GetValue(curBatch * tiling_->fullMaxBlockNum + kvBlkTableIdx);
            if (kvBlockNumIdx < 0) {
                missGatherSizes[m] = 0;
                missFullKvAddrs[m] = 0;
                missFullKRopeAddrs[m] = 0;
                continue;
            }
            missFullKvAddrs[m] = kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kvCacheDim
                                 + kvBlkSizeOff * tiling_->kvCacheDim;
            missFullKRopeAddrs[m] = kvBlockNumIdx * tiling_->fullKvBlockSize * tiling_->kRopeDim
                                    + kvBlkSizeOff * tiling_->kRopeDim;
        }

        // === Overlap mode: interleave gather(miss) with compute(hit) ===

        // Phase A: Kick off first gather, then alternate gather/compute
        int32_t gIdx = 0, hIdx = 0;
        if (missCount > 0 && missGatherSizes[0] > 0) {
            IssueSingleGather(0, missGatherSizes, missKvAddrs, missKRopeAddrs,
                              missFullKvAddrs, missFullKRopeAddrs);
        }
        gIdx = 1;

        while (hIdx < hitCount || gIdx < missCount) {
            // Issue next gather (wait for previous MTE3 first)
            if (gIdx < missCount) {
                SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
                if (missGatherSizes[gIdx] > 0) {
                    IssueSingleGather(gIdx, missGatherSizes, missKvAddrs, missKRopeAddrs,
                                      missFullKvAddrs, missFullKRopeAddrs);
                }
                gIdx++;
            }
            // Compute attention for one hit block (overlaps with async gather)
            if (hIdx < hitCount) {
                for (int64_t t = 0; t < hitGatherSizes[hIdx]; t++) {
                    ProcessSingleToken(
                        hitKvAddrs[hIdx] + t * tiling_->kvCacheDim,
                        hitKRopeAddrs[hIdx] + t * tiling_->kRopeDim,
                        queryFloat, queryRopeFloat, tmpFloat, outHitFloat,
                        scoreOut, runMaxTensor, runSumTensor, oldMaxTensor,
                        expLocal, brcbFloat, kvDimFloatAlign, kRopeDimFloatAlign);
                }
                hIdx++;
            }
        }

        // Wait for all outstanding gathers to complete
        if (missCount > 0) {
            SyncEvent<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
        }

        // Phase B: Compute attention for miss tokens (data now in selKvCache)
        for (int32_t m = 0; m < missCount; m++) {
            if (missGatherSizes[m] <= 0) continue;
            for (int64_t t = 0; t < missGatherSizes[m]; t++) {
                ProcessSingleToken(
                    missKvAddrs[m] + t * tiling_->kvCacheDim,
                    missKRopeAddrs[m] + t * tiling_->kRopeDim,
                    queryFloat, queryRopeFloat, tmpFloat, outHitFloat,
                    scoreOut, runMaxTensor, runSumTensor, oldMaxTensor,
                    expLocal, brcbFloat, kvDimFloatAlign, kRopeDimFloatAlign);
            }
        }

    } // end while chunks

    if (totalHitCount + totalMissCount == 0) {
        for (int32_t j = 0; j < topkOneSortAlign_; j++) curStatLocal.SetValue(j, -1);
        curStatLocal.SetValue(tiling_->topk, 0);
        seqLocal.SetValue(hnIdx, 0);
        return;
    }

    // === Phase 4: No separate merge needed (unified stream, outMissFloat unused) ===
    // All tokens accumulated into outHitFloat with runMaxTensor/runSumTensor

    // === Phase 5: Normalize and write output ===
    // Normalize: outHitFloat /= runSum  (via Reciprocal + Brcb + Mul)
    {
        Reciprocal(expLocal, runSumTensor, BLOCK_FLOATS);
        pipe_barrier(PIPE_ALL);
        volatile float invSum = expLocal.GetValue(0);
        if (invSum != invSum || invSum > 1e30f) invSum = 1.0f;
        Duplicate(brcbFloat, (float)invSum, kvDimFloatAlign);
        pipe_barrier(PIPE_ALL);
        Mul(outHitFloat, outHitFloat, brcbFloat, kvDimFloatAlign);
        pipe_barrier(PIPE_ALL);

        LocalTensor<T> outTensor = kvCacheQue_.AllocTensor<T>();
        Cast(outTensor, outHitFloat, RoundMode::CAST_RINT, tiling_->kvCacheDim);
        pipe_barrier(PIPE_ALL);

        int64_t outGmOff = globalBsIdx * tiling_->headnum * tiling_->headDim + hnIdx * tiling_->headDim;
        DataCopyExtParams outParams{static_cast<uint16_t>(1),
            static_cast<uint32_t>(tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(attentionOutGm_[outGmOff], outTensor, outParams);
        kvCacheQue_.FreeTensor(outTensor);
    }

    // Clean invalid status slots
    for (int32_t j = insertIdx; j < topkOneSortAlign_; j++) {
        curStatLocal.SetValue(j, -1);
    }
    curStatLocal.SetValue(tiling_->topk, selActualSeqLen);
    seqLocal.SetValue(hnIdx, selActualSeqLen);
}

// --- IssueSingleGather: async gather one miss block via gatherQue_ ---
template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::IssueSingleGather(
    int32_t missIdx, int64_t* missGatherSizes,
    int64_t* missKvAddrs, int64_t* missKRopeAddrs,
    int64_t* missFullKvAddrs, int64_t* missFullKRopeAddrs)
{
    LocalTensor<T> gBuf = gatherQue_.AllocTensor<T>();
    DataCopyPadExtParams<T> gPad{false, 0, 0, 0};

    // MTE2: full_kv_cache(GM) -> gBuf(UB)
    DataCopyExtParams kvP{static_cast<uint16_t>(1),
        static_cast<uint32_t>(missGatherSizes[missIdx] * tiling_->kvCacheDim * sizeof(T)), 0, 0, 0};
    DataCopyPad(gBuf, fullKvCacheGm_[missFullKvAddrs[missIdx]], kvP, gPad);

    if (tiling_->kRopeDim > 0) {
        DataCopyExtParams rP{static_cast<uint16_t>(1),
            static_cast<uint32_t>(missGatherSizes[missIdx] * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(gBuf[kRopeUbOffset_], fullKRopeGm_[missFullKRopeAddrs[missIdx]], rP, gPad);
    }

    gatherQue_.EnQue(gBuf);
    gBuf = gatherQue_.DeQue<T>();

    // MTE3: gBuf(UB) -> selection_kv_cache(GM) -- async, Vector can proceed
    DataCopyPad(selKvCacheGm_[missKvAddrs[missIdx]], gBuf, kvP);
    if (tiling_->kRopeDim > 0) {
        DataCopyExtParams rP{static_cast<uint16_t>(1),
            static_cast<uint32_t>(missGatherSizes[missIdx] * tiling_->kRopeDim * sizeof(T)), 0, 0, 0};
        DataCopyPad(selKRopeGm_[missKRopeAddrs[missIdx]], gBuf[kRopeUbOffset_], rP);
    }
    gatherQue_.FreeTensor(gBuf);
}

// --- Hit Detection ---
template <typename T>
__aicore__ inline void FusedSparseAttentionOverlapOp<T>::DetectHitsForHead(
    int64_t hnIdx, LocalTensor<int32_t>& topkLocal,
    LocalTensor<int32_t>& statLocal, LocalTensor<int32_t>& hitFlagLocal)
{
    for (int64_t i = 0; i < tiling_->topk; i++) {
        int32_t topkVal = topkLocal.GetValue(i);
        if (topkVal < 0) continue;

        for (int64_t j = 0; j < tiling_->topk; j++) {
            int32_t statVal = statLocal.GetValue(j);
            if (topkVal == statVal) {
                hitFlagLocal.SetValue(i, HIT_FLAG);
                break;
            }
        }
    }
}

} // namespace FusedSparseAttentionOverlapNs

#endif // FUSED_SPARSE_ATTENTION_OVERLAP_KERNEL_H_
