// SPDX-License-Identifier: Apache-2.0
#pragma once

namespace vllm_ascend {

// Inputs are contiguous views over the physical cache storage. The explicit
// page strides preserve Hybrid-cache gaps without materializing the full cache.
inline void qli_candidate_gather(
    const at::Tensor &key, const at::Tensor &scale, const at::Tensor &candidates,
    const at::Tensor &table, const at::Tensor &cuQ, const at::Tensor &seqK,
    int64_t keyPageStride, int64_t scalePageStride, int64_t pageSize,
    at::Tensor &packedKey, at::Tensor &packedScale, at::Tensor &positions, at::Tensor &lengths)
{
    TORCH_CHECK(key.dim() == 2 && key.size(1) == 128 && key.scalar_type() == at::kChar,
                "QLI candidate gather requires INT8 key rows with width 128");
    TORCH_CHECK(scale.dim() == 1 && scale.scalar_type() == at::kHalf,
                "QLI candidate gather requires a flattened FP16 scale cache");
    TORCH_CHECK(candidates.dim() == 2 && candidates.scalar_type() == at::kInt &&
                candidates.size(1) > 0 && candidates.size(1) <= 2048 && candidates.size(1) % 32 == 0,
                "QLI candidate gather requires 32-aligned sorted INT32 candidate rows, at most 2048 blocks");
    TORCH_CHECK(table.dim() == 2 && table.scalar_type() == at::kInt &&
                table.size(0) > 0 && table.size(0) <= 2048 && table.size(1) > 0 && table.size(1) <= 8192,
                "QLI candidate gather requires an INT32 block table with at most 2048 requests and 8192 pages");
    TORCH_CHECK(cuQ.dim() == 1 && cuQ.size(0) == table.size(0) + 1 && cuQ.scalar_type() == at::kInt &&
                seqK.dim() == 1 && seqK.size(0) == table.size(0) && seqK.scalar_type() == at::kInt,
                "QLI candidate gather requires INT32 TND query offsets and per-request key lengths");
    TORCH_CHECK(pageSize >= 8 && pageSize % 8 == 0 && keyPageStride >= pageSize && scalePageStride >= pageSize,
                "QLI candidate gather requires 8-aligned pages and valid physical page strides");
    const int64_t rows = candidates.size(0);
    const int64_t capacity = candidates.size(1) * 8;
    TORCH_CHECK(packedKey.dim() == 2 && packedKey.size(0) == rows * capacity && packedKey.size(1) == 128 &&
                packedKey.scalar_type() == at::kChar && packedScale.dim() == 1 &&
                packedScale.numel() == rows * capacity && packedScale.scalar_type() == at::kHalf &&
                positions.dim() == 2 && positions.size(0) == rows && positions.size(1) == capacity &&
                positions.scalar_type() == at::kInt && lengths.dim() == 1 && lengths.size(0) == rows &&
                lengths.scalar_type() == at::kInt, "QLI candidate gather output shapes or dtypes do not match");
    for (const auto &tensor : {key, scale, candidates, table, cuQ, seqK, packedKey, packedScale, positions, lengths}) {
        TORCH_CHECK(tensor.device() == key.device() && tensor.is_contiguous(),
                    "QLI candidate gather requires contiguous tensors on the same device");
    }
    if (rows == 0 || key.device().is_meta()) {
        return;
    }
    EXEC_NPU_CMD(aclnnQliCandidateGather, key, scale, candidates, table, cuQ, seqK,
                 keyPageStride, scalePageStride, pageSize, packedKey, packedScale, positions, lengths);
}
} // namespace vllm_ascend
