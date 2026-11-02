# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded, fixed-address storage for one forward's shared indexer candidates."""

import torch

INDEX_WIDTH = 128
CANDIDATE_BLOCK_SIZE = 8
PACKED_PAGE_SIZE = 128
MAX_CANDIDATES = 2048
MAX_REQUESTS = 2048
MAX_TABLE_PAGES = 8192
INVALID_CANDIDATE = (1 << 24) - 1
CANDIDATE_BUFFER_BUDGET = 512 * 1024 * 1024


def candidate_reuse_layers(topology, local_layers):
    """Find a local candidate source and consumers sharing uncompressed K.

    Non-index-source layers already reuse TopK and need no packed K. A group
    must include its candidate source and KV owner on this pipeline stage;
    require at least two consumers to amortize preparing the shared buffer.
    """
    local_layers = frozenset(local_layers)
    source = topology.layer(topology.candidate_source_layer_id)
    count = topology.candidate_topk_blocks
    if (
        source.layer_idx not in local_layers
        or source.kv_source_layer not in local_layers
        or not source.is_index_source
        or not source.is_candidate_source
        or source.compress_ratio != 1
        or topology.candidate_block_size != CANDIDATE_BLOCK_SIZE
        or not 0 < count <= MAX_CANDIDATES
        or count % 64
        or (count - 1) * CANDIDATE_BLOCK_SIZE + 1 < topology.index_topk
    ):
        return ()
    consumers = tuple(
        role.layer_idx
        for role in topology.layers
        if role.layer_idx in local_layers
        and role.layer_idx > source.layer_idx
        and role.is_index_source
        and role.uses_candidate_filter
        and role.kv_source_layer == source.kv_source_layer
        and role.compress_ratio == source.compress_ratio
    )
    return (source.layer_idx, *consumers) if len(consumers) >= 2 else ()


class CandidateKBuffer:
    """The candidate source overwrites active rows on every forward/replay.

    Consumers use the same storage with their own Q/weights. No readiness flag
    or Python generation counter participates in device replay. Callers must
    establish matching cache ownership and request coordinates before sharing.
    """

    def __init__(
        self, max_tokens, heads, topk, candidate_blocks, device, *, metadata_op=None, gather_op=None, indexer_op=None
    ):
        if not 0 < candidate_blocks <= MAX_CANDIDATES or candidate_blocks % 64:
            raise ValueError("candidate_blocks must be a multiple of 64, at most 2048")
        if (candidate_blocks - 1) * CANDIDATE_BLOCK_SIZE + 1 < topk:
            raise ValueError("candidate capacity must cover TopK even with a partial final block")
        self.capacity = candidate_blocks * CANDIDATE_BLOCK_SIZE
        self.candidate_blocks = candidate_blocks
        self.heads = heads
        self.topk = topk
        row_bytes = self.capacity * (INDEX_WIDTH + 2 + 4) + 4
        self.max_rows = min(max_tokens, CANDIDATE_BUFFER_BUDGET // row_bytes)
        if self.max_rows <= 0:
            raise ValueError("candidate buffer requires a positive token capacity")
        self.gather_op = gather_op if gather_op is not None else torch.ops._C_ascend.qli_candidate_gather
        self.indexer_op = indexer_op if indexer_op is not None else torch.ops._C_ascend.npu_quant_lightning_indexer_v3
        if metadata_op is None:
            metadata_op = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata
        self.key = torch.empty((self.max_rows * self.capacity, INDEX_WIDTH), dtype=torch.int8, device=device)
        self.scale = torch.empty((self.max_rows * self.capacity,), dtype=torch.float16, device=device)
        self.positions = torch.empty((self.max_rows, self.capacity), dtype=torch.int32, device=device)
        self.lengths = torch.empty((self.max_rows,), dtype=torch.int32, device=device)
        self.query_offsets = torch.arange(self.max_rows + 1, dtype=torch.int32, device=device)
        pages_per_row = self.capacity // PACKED_PAGE_SIZE
        self.block_table = torch.arange(self.max_rows * pages_per_row, dtype=torch.int32, device=device).reshape(
            self.max_rows, pages_per_row
        )
        max_lengths = torch.full((self.max_rows,), self.capacity, dtype=torch.int32, device=device)
        # A3 metadata only assigns batches/query rows. Upper-bound K is safe:
        # the consumer kernel reads the packed lengths produced on device.
        # Build once at model initialization, never inside forward or capture.
        self.metadata = tuple(
            metadata_op(
                heads,
                1,
                INDEX_WIDTH,
                topk,
                2,
                cu_seqlens_q=self.query_offsets[: rows + 1],
                seqused_k=max_lengths[:rows],
                batch_size=rows,
                max_seqlen_q=1,
                max_seqlen_k=self.capacity,
                layout_q="TND",
                layout_k="PA_BBND",
                mask_mode=0,
                cmp_ratio=1,
            )
            for rows in range(1, self.max_rows + 1)
        )

    def supports(self, tokens, source_cache, source_metadata, compress_ratio):
        if not 0 < tokens <= self.max_rows or compress_ratio != 1 or len(source_cache) != 2:
            return False
        key, scale = source_cache
        scale = scale.squeeze(-1)
        if (
            key.ndim != 4
            or key.shape[0] <= 0
            or key.shape[1] <= 0
            or key.shape[2:] != (1, INDEX_WIDTH)
            or key.dtype != torch.int8
            or key.device != self.key.device
            or key.stride()[1:] != (INDEX_WIDTH, INDEX_WIDTH, 1)
            or key.stride(0) % INDEX_WIDTH
            or key.stride(0) < key.shape[1] * INDEX_WIDTH
            or scale.ndim != 3
            or scale.shape != key.shape[:3]
            or scale.dtype != torch.float16
            or scale.stride(1) != 1
            or scale.stride(0) < scale.shape[1]
            or scale.device != key.device
            or key.shape[1] % CANDIDATE_BLOCK_SIZE
        ):
            return False
        table = source_metadata.block_table
        ql = source_metadata.query_start_loc
        kl = source_metadata.cache_seq_lens
        return (
            table.ndim == 2
            and 0 < table.shape[0] <= MAX_REQUESTS
            and 0 < table.shape[1] <= MAX_TABLE_PAGES
            and ql.shape == (table.shape[0] + 1,)
            and kl.shape == (table.shape[0],)
            and all(t.dtype == torch.int32 and t.device == key.device and t.is_contiguous() for t in (table, ql, kl))
        )

    def prepare(self, candidates, source_cache, source_metadata):
        """Pack source-generated unique blocks; candidate padding is -1."""
        rows = candidates.shape[0]
        key, scale = source_cache
        scale = scale.squeeze(-1)
        blocks = candidates.reshape(rows, self.candidate_blocks)
        # FP32 exactly represents the admitted block IDs. Integer sort on A3
        # is much slower; the finite sentinel also avoids INT32 overflow.
        blocks = torch.where(blocks >= 0, blocks, INVALID_CANDIDATE).float().sort(dim=-1).values.int()
        key_stride = key.stride(0) // INDEX_WIDTH
        scale_stride = scale.stride(0)
        key_rows = key.as_strided(((key.shape[0] - 1) * key_stride + key.shape[1], INDEX_WIDTH), (INDEX_WIDTH, 1))
        scale_rows = scale.as_strided(((scale.shape[0] - 1) * scale_stride + scale.shape[1],), (1,))
        self.gather_op(
            key_rows,
            scale_rows,
            blocks,
            source_metadata.block_table,
            source_metadata.query_start_loc,
            source_metadata.cache_seq_lens,
            key_stride,
            scale_stride,
            key.shape[1],
            self.key[: rows * self.capacity],
            self.scale[: rows * self.capacity],
            self.positions[:rows],
            self.lengths[:rows],
        )

    def select(self, query, weights, query_scale):
        rows = query.shape[0]
        indices, _, _ = self.indexer_op(
            query,
            self.key[: rows * self.capacity].view(-1, PACKED_PAGE_SIZE, 1, INDEX_WIDTH),
            weights,
            query_scale,
            self.scale[: rows * self.capacity].view(-1, PACKED_PAGE_SIZE, 1),
            self.topk,
            2,
            cu_seqlens_q=self.query_offsets[: rows + 1],
            seqused_k=self.lengths[:rows],
            block_table=self.block_table[:rows],
            metadata=self.metadata[rows - 1],
            max_seqlen_q=1,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=0,
            cmp_ratio=1,
            candidate_mode=5,
            candidate_topk_blocks=self.candidate_blocks,
            candidate_block_size=CANDIDATE_BLOCK_SIZE,
        )
        indices = indices.squeeze(1)
        original = torch.gather(self.positions[:rows], 1, indices.clamp(min=0).long())
        return torch.where(indices >= 0, original, -1)
