// SPDX-License-Identifier: Apache-2.0
#include "qli_candidate_gather_tiling.h"
#include <algorithm>
#include <register/op_impl_registry.h>
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
ge::graphStatus TilingQliCandidateGather(gert::TilingContext *context)
{
    const auto &key = context->GetInputShape(0)->GetStorageShape();
    const auto &scale = context->GetInputShape(1)->GetStorageShape();
    const auto &candidates = context->GetInputShape(2)->GetStorageShape();
    const auto &table = context->GetInputShape(3)->GetStorageShape();
    const auto &cuQ = context->GetInputShape(4)->GetStorageShape();
    const auto &seqK = context->GetInputShape(5)->GetStorageShape();
    constexpr int64_t WIDTH = 128;
    constexpr int64_t BLOCK = 8;
    constexpr int64_t TILE_BLOCKS = 32;
    const auto *attrs = context->GetAttrs();
    const int64_t keyPageStride = *attrs->GetAttrPointer<int64_t>(0);
    const int64_t scalePageStride = *attrs->GetAttrPointer<int64_t>(1);
    const int64_t pageSize = *attrs->GetAttrPointer<int64_t>(2);
    if (key.GetDimNum() != 2 || key.GetDim(1) != WIDTH || scale.GetDimNum() != 1 ||
        candidates.GetDimNum() != 2 || table.GetDimNum() != 2 || cuQ.GetDimNum() != 1 ||
        seqK.GetDimNum() != 1 || candidates.GetDim(0) <= 0 || candidates.GetDim(1) <= 0 ||
        candidates.GetDim(1) > 2048 || candidates.GetDim(1) % TILE_BLOCKS != 0 ||
        table.GetDim(0) <= 0 || table.GetDim(0) > 2048 || table.GetDim(1) <= 0 || table.GetDim(1) > 8192 ||
        cuQ.GetDim(0) != table.GetDim(0) + 1 || seqK.GetDim(0) != table.GetDim(0) ||
        pageSize < BLOCK || pageSize % BLOCK != 0 || keyPageStride < pageSize || scalePageStride < pageSize) {
        return ge::GRAPH_FAILED;
    }
    const int64_t tokens = candidates.GetDim(0);
    const int64_t count = candidates.GetDim(1);
    const int64_t expected[] = {tokens * count * BLOCK * WIDTH, tokens * count * BLOCK,
                                tokens * count * BLOCK, tokens};
    for (size_t i = 0; i < 4; ++i) {
        if (context->GetOutputShape(i)->GetStorageShape().GetShapeSize() != expected[i]) {
            return ge::GRAPH_FAILED;
        }
    }
    platform_ascendc::PlatformAscendC platform(context->GetPlatformInfo());
    context->SetBlockDim(std::min<int64_t>(platform.GetCoreNumAiv(), tokens * count / TILE_BLOCKS));
    context->SetTilingKey(0);
    context->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize();
    QliCandidateGatherTilingData data;
    data.set_tokens(tokens);
    data.set_candidates(count);
    data.set_batch(table.GetDim(0));
    data.set_pagesPerBatch(table.GetDim(1));
    data.set_keyPageStride(keyPageStride);
    data.set_scalePageStride(scalePageStride);
    data.set_pageSize(pageSize);
    data.set_keyRows(key.GetDim(0));
    data.set_scaleRows(scale.GetDim(0));
    data.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(data.GetDataSize());
    return ge::GRAPH_SUCCESS;
}
IMPL_OP_OPTILING(QliCandidateGather).Tiling(TilingQliCandidateGather);
} // namespace optiling
