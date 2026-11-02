// SPDX-License-Identifier: Apache-2.0
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t WIDTH = 128;
constexpr uint32_t BLOCK = 8;
constexpr uint32_t TILE_BLOCKS = 32;
constexpr uint32_t SCALE_PITCH = 16;

class CandidateGather {
public:
    __aicore__ inline void Init(GM_ADDR key, GM_ADDR scale, GM_ADDR candidates, GM_ADDR table,
                               GM_ADDR cuQ, GM_ADDR seqK, GM_ADDR packedKey, GM_ADDR packedScale,
                               GM_ADDR positions, GM_ADDR lengths, const QliCandidateGatherTilingData &data)
    {
        data_ = data;
        key_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(key));
        scale_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(scale));
        candidates_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(candidates));
        table_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(table));
        cuQ_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(cuQ));
        seqK_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(seqK));
        packedKey_.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t *>(packedKey));
        packedScale_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(packedScale));
        positions_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(positions));
        lengths_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(lengths));
        const uint32_t requestPitch = (data_.batch + 8) / 8 * 8;
        pipe_.InitBuffer(requestBuf_, requestPitch * sizeof(int32_t) * 2);
        pipe_.InitBuffer(candidateBuf_, data_.candidates * sizeof(int32_t));
        pipe_.InitBuffer(tableBuf_, ((data_.pagesPerBatch + 7) / 8 * 8) * sizeof(int32_t));
        pipe_.InitBuffer(keyBuf_, TILE_BLOCKS * BLOCK * WIDTH);
        pipe_.InitBuffer(scaleBuf_, TILE_BLOCKS * SCALE_PITCH * sizeof(half));
        pipe_.InitBuffer(positionBuf_, TILE_BLOCKS * BLOCK * sizeof(int32_t));
        pipe_.InitBuffer(smallBuf_, 64);
        cuQLocal_ = requestBuf_.Get<int32_t>();
        seqKLocal_ = cuQLocal_[requestPitch];
        CopyMetadata(cuQLocal_, cuQ_, data_.batch + 1);
        CopyMetadata(seqKLocal_, seqK_, data_.batch);
        SetFlag<HardEvent::MTE2_S>(0);
        WaitFlag<HardEvent::MTE2_S>(0);
        auto offsets = smallBuf_.Get<int32_t>();
        // Bounded initialization of the within-block index vector.
        for (uint32_t i = 0; i < BLOCK; ++i) {
            offsets.SetValue(i, i);
        }
        SetFlag<HardEvent::S_V>(0);
        WaitFlag<HardEvent::S_V>(0);
    }

    __aicore__ inline void Process()
    {
        auto candidates = candidateBuf_.Get<int32_t>();
        auto table = tableBuf_.Get<int32_t>();
        auto key = keyBuf_.Get<int8_t>();
        auto scale = scaleBuf_.Get<half>();
        auto positions = positionBuf_.Get<int32_t>();
        auto offsets = smallBuf_.Get<int32_t>();
        auto length = offsets[8];
        int64_t cachedRow = -1;
        int32_t cachedBatch = -1;
        int32_t visible = 0;
        const int64_t blocks = data_.tokens * data_.candidates;
        for (int64_t begin = GetBlockIdx() * TILE_BLOCKS; begin < blocks;
             begin += GetBlockNum() * TILE_BLOCKS) {
            const int64_t row = begin / data_.candidates;
            const int64_t candidateStart = begin % data_.candidates;
            if (row != cachedRow) {
                const int32_t batch = RequestForRow(row);
                visible = batch < data_.batch ? seqKLocal_.GetValue(batch) - cuQLocal_.GetValue(batch + 1) + row + 1 : 0;
                if (batch < data_.batch && batch != cachedBatch) {
                    CopyMetadata(table, table_[static_cast<int64_t>(batch) * data_.pagesPerBatch], data_.pagesPerBatch);
                    cachedBatch = batch;
                }
                CopyMetadata(candidates, candidates_[row * data_.candidates], data_.candidates);
                SetFlag<HardEvent::MTE2_S>(0);
                WaitFlag<HardEvent::MTE2_S>(0);
                cachedRow = row;
            }
            if (candidateStart == 0) {
                int32_t low = 0;
                int32_t high = data_.candidates;
                while (low < high) {
                    const int32_t mid = (low + high) / 2;
                    if (static_cast<int64_t>(candidates.GetValue(mid)) * BLOCK < visible) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
                int32_t valid = 0;
                if (low > 0 && visible > 0) {
                    const int32_t tail = visible - candidates.GetValue(low - 1) * BLOCK;
                    valid = (low - 1) * BLOCK + (tail < BLOCK ? tail : BLOCK);
                }
                Duplicate(length, valid, 8);
                SetFlag<HardEvent::V_MTE3>(0);
                WaitFlag<HardEvent::V_MTE3>(0);
                DataCopyPad(lengths_[row], length, DataCopyExtParams{1, sizeof(int32_t), 0, 0, 0});
            }
            // Each scalar read supplies a DMA address, never a payload element.
            for (uint32_t i = 0; i < TILE_BLOCKS; ++i) {
                const int32_t block = candidates.GetValue(candidateStart + i);
                const int64_t logical = static_cast<int64_t>(block) * BLOCK;
                Adds(positions[i * BLOCK], offsets, static_cast<int32_t>(logical), BLOCK);
                const int64_t logicalPage = logical / data_.pageSize;
                int64_t keyRow = -1;
                int64_t scaleRow = -1;
                if (logical >= 0 && logical < visible && logicalPage < data_.pagesPerBatch) {
                    const int32_t physicalPage = table.GetValue(logicalPage);
                    if (physicalPage >= 0) {
                        keyRow = static_cast<int64_t>(physicalPage) * data_.keyPageStride + logical % data_.pageSize;
                        scaleRow = static_cast<int64_t>(physicalPage) * data_.scalePageStride + logical % data_.pageSize;
                    }
                }
                if (keyRow >= 0 && keyRow + BLOCK <= data_.keyRows &&
                    scaleRow >= 0 && scaleRow + BLOCK <= data_.scaleRows) {
                    DataCopy(key[i * BLOCK * WIDTH], key_[keyRow * WIDTH], BLOCK * WIDTH);
                    DataCopyPad(scale[i * SCALE_PITCH], scale_[scaleRow],
                                DataCopyExtParams{1, BLOCK * sizeof(half), 0, 0, 0},
                                DataCopyPadExtParams<half>{false, 0, 0, 0});
                } else {
                    Duplicate(key[i * BLOCK * WIDTH].ReinterpretCast<int16_t>(), static_cast<int16_t>(0),
                              BLOCK * WIDTH / sizeof(int16_t));
                    Duplicate(scale[i * SCALE_PITCH], static_cast<half>(0), SCALE_PITCH);
                }
            }
            SetFlag<HardEvent::V_MTE3>(0);
            WaitFlag<HardEvent::V_MTE3>(0);
            SetFlag<HardEvent::MTE2_MTE3>(0);
            WaitFlag<HardEvent::MTE2_MTE3>(0);
            DataCopy(packedKey_[begin * BLOCK * WIDTH], key, TILE_BLOCKS * BLOCK * WIDTH);
            DataCopyPad(packedScale_[begin * BLOCK], scale,
                        DataCopyExtParams{TILE_BLOCKS, BLOCK * sizeof(half), 0, 0, 0});
            DataCopy(positions_[begin * BLOCK], positions, TILE_BLOCKS * BLOCK);
            SetFlag<HardEvent::MTE3_MTE2>(0);
            WaitFlag<HardEvent::MTE3_MTE2>(0);
            SetFlag<HardEvent::MTE3_V>(0);
            WaitFlag<HardEvent::MTE3_V>(0);
            SetFlag<HardEvent::S_MTE2>(0);
            WaitFlag<HardEvent::S_MTE2>(0);
        }
    }

private:
    __aicore__ inline void CopyMetadata(LocalTensor<int32_t> dst, GlobalTensor<int32_t> src, uint32_t count)
    {
        DataCopyPad(dst, src, DataCopyExtParams{1, count * static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0},
                    DataCopyPadExtParams<int32_t>{false, 0, 0, 0});
    }

    __aicore__ inline int32_t RequestForRow(int32_t row)
    {
        int32_t low = 0;
        int32_t high = data_.batch;
        while (low < high) {
            const int32_t mid = (low + high) / 2;
            if (cuQLocal_.GetValue(mid + 1) <= row) {
                low = mid + 1;
            } else {
                high = mid;
            }
        }
        return low;
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> requestBuf_, candidateBuf_, tableBuf_, keyBuf_, scaleBuf_, positionBuf_, smallBuf_;
    LocalTensor<int32_t> cuQLocal_, seqKLocal_;
    GlobalTensor<int8_t> key_, packedKey_;
    GlobalTensor<half> scale_, packedScale_;
    GlobalTensor<int32_t> candidates_, table_, cuQ_, seqK_, positions_, lengths_;
    QliCandidateGatherTilingData data_;
};
} // namespace

extern "C" __global__ __aicore__ void qli_candidate_gather(
    GM_ADDR key, GM_ADDR scale, GM_ADDR candidates, GM_ADDR table, GM_ADDR cuQ, GM_ADDR seqK,
    GM_ADDR packedKey, GM_ADDR packedScale, GM_ADDR positions, GM_ADDR lengths,
    GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(data, tiling);
    KERNEL_TASK_TYPE(0, KERNEL_TYPE_AIV_ONLY);
    CandidateGather op;
    op.Init(key, scale, candidates, table, cuQ, seqK, packedKey, packedScale, positions, lengths, data);
    op.Process();
}
