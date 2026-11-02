// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <register/tilingdata_base.h>

namespace optiling {
BEGIN_TILING_DATA_DEF(QliCandidateGatherTilingData)
TILING_DATA_FIELD_DEF(int64_t, tokens);
TILING_DATA_FIELD_DEF(int64_t, candidates);
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, pagesPerBatch);
TILING_DATA_FIELD_DEF(int64_t, keyPageStride);
TILING_DATA_FIELD_DEF(int64_t, scalePageStride);
TILING_DATA_FIELD_DEF(int64_t, pageSize);
TILING_DATA_FIELD_DEF(int64_t, keyRows);
TILING_DATA_FIELD_DEF(int64_t, scaleRows);
END_TILING_DATA_DEF;
REGISTER_TILING_DATA_CLASS(QliCandidateGather, QliCandidateGatherTilingData)
} // namespace optiling
