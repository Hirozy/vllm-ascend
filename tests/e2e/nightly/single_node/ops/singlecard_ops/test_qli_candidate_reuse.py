# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.models.deepseek_v41.candidate_reuse import CandidateKBuffer
from vllm_ascend.utils import bootstrap_custom_op_env, enable_custom_op

bootstrap_custom_op_env(include_vendor_lib=True)
import vllm_ascend.vllm_ascend_C  # type: ignore[import-untyped] # noqa: E402,F401

enable_custom_op()


@pytest.mark.parametrize("tokens", [0, 2])
def test_candidate_gather_meta_outputs_and_validation(tokens):
    count, capacity = 64, 64 * 8
    args = [
        torch.empty((128, 128), dtype=torch.int8, device="meta"),
        torch.empty(128, dtype=torch.float16, device="meta"),
        torch.empty((tokens, count), dtype=torch.int32, device="meta"),
        torch.empty((1, 2), dtype=torch.int32, device="meta"),
        torch.empty(2, dtype=torch.int32, device="meta"),
        torch.empty(1, dtype=torch.int32, device="meta"),
        64,
        64,
        64,
        torch.empty((tokens * capacity, 128), dtype=torch.int8, device="meta"),
        torch.empty(tokens * capacity, dtype=torch.float16, device="meta"),
        torch.empty((tokens, capacity), dtype=torch.int32, device="meta"),
        torch.empty(tokens, dtype=torch.int32, device="meta"),
    ]
    assert torch.ops._C_ascend.qli_candidate_gather(*args) is None
    args[0] = torch.empty((128, 127), dtype=torch.int8, device="meta")
    with pytest.raises(RuntimeError, match="width 128"):
        torch.ops._C_ascend.qli_candidate_gather(*args)


@pytest.mark.parametrize("page,batch", [(128, 1), (64, 3)])
@torch.inference_mode()
def test_candidate_source_and_four_consumers_refresh_during_graph_replay(page, batch):
    """Change source Q, four consumer Qs, cache, page table and ragged lengths.

    The captured source regenerates candidates on every replay. Compare all
    consumers against full-cache mode 4 with fresh metadata and candidates,
    including equal scores, empty requests, causal tails and padded queries.
    """
    if not get_current_hardware_profile().supports(HardwareCapability.QLI_CANDIDATE_K_REUSE):
        pytest.skip("requires the candidate K reuse operator stack")
    torch.manual_seed(71)
    tokens, heads, width, pages, topk, count = 6, 32, 128, 1100, 512, 2048
    key = torch.empty_strided(
        (pages, page, 1, width), ((page + 1026) * width, width, width, 1), dtype=torch.int8, device="npu"
    )
    scale = torch.empty_strided((pages, page, 1), (page + 73728, 1, 1), dtype=torch.float16, device="npu")
    table = torch.empty((batch, 1024), dtype=torch.int32, device="npu")
    ql = torch.empty(batch + 1, dtype=torch.int32, device="npu")
    kl = torch.empty(batch, dtype=torch.int32, device="npu")
    queries = [torch.empty((tokens, heads, width), dtype=torch.int8, device="npu") for _ in range(5)]
    weights = [torch.empty((tokens, heads), dtype=torch.float16, device="npu") for _ in range(5)]
    qs = torch.full((tokens, heads), 0.01, dtype=torch.float16, device="npu")
    candidates = torch.full((tokens, 1, count), -1, dtype=torch.int32, device="npu")
    original_metadata = torch.empty(1024, dtype=torch.int32, device="npu")
    source_metadata = SimpleNamespace(block_table=table, query_start_loc=ql, cache_seq_lens=kl)
    cache = (key, scale.unsqueeze(-1))
    shared = CandidateKBuffer(tokens, heads, topk, count, key.device)
    buffers = (shared.key, shared.scale, shared.positions, shared.lengths)
    addresses = [value.data_ptr() for value in buffers]

    def full(layer, mode, blocks=None):
        return torch.ops._C_ascend.npu_quant_lightning_indexer_v3(
            queries[layer],
            key,
            weights[layer],
            qs,
            scale,
            topk,
            2,
            cu_seqlens_q=ql,
            seqused_k=kl,
            block_table=table,
            metadata=original_metadata,
            max_seqlen_q=tokens,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
            candidate_mode=mode,
            candidate_topk_index=blocks,
            candidate_topk_blocks=count,
            candidate_block_size=8,
        )

    def update(iteration):
        # TND query offsets cover the physical rows, including the zero-K
        # dummy request, as the model's graph metadata builder does.
        offsets = [[0, 2, 4, 6], [0, 0, 3, 6], [0, 1, 5, 6], [0, 6, 6, 6], [0, 2, 4, 6]][iteration]
        lengths = [[65535, 511, 16385], [7, 513, 4095], [9, 32767, 0], [262, 0, 0], [0, 0, 0]][iteration]
        if batch == 1:
            offsets, lengths = [0, tokens], [lengths[0]]
        ql.copy_(torch.tensor(offsets, dtype=torch.int32))
        kl.copy_(torch.tensor(lengths, dtype=torch.int32))
        table.copy_(torch.stack([torch.randperm(pages, dtype=torch.int32)[:1024] for _ in range(batch)]))
        key.copy_(torch.randint(-8, 9, key.shape, dtype=torch.int8))
        scale.copy_(torch.rand(scale.shape, dtype=torch.float16) * 0.02)
        for layer in range(5):
            queries[layer].copy_(torch.randint(-8, 9, queries[layer].shape, dtype=torch.int8))
            # Indexer projection weights are signed, unlike quantization scales.
            weights[layer].copy_(torch.rand(weights[layer].shape, dtype=torch.float16) * 2 - 1)
            if iteration == 3:
                weights[layer].zero_()
        new_metadata = torch.ops._C_ascend.npu_quant_lightning_indexer_v2_metadata(
            heads,
            1,
            width,
            topk,
            2,
            cu_seqlens_q=ql,
            seqused_k=kl,
            batch_size=batch,
            max_seqlen_q=tokens,
            max_seqlen_k=max(1, max(lengths)),
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=3,
            cmp_ratio=1,
        )
        original_metadata.copy_(new_metadata)

    def chain():
        candidates.copy_(full(0, 1)[2])
        shared.prepare(candidates, cache, source_metadata)
        return [shared.select(queries[layer], weights[layer], qs) for layer in range(1, 5)]

    update(0)
    assert shared.supports(tokens, cache, source_metadata, 1)
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        outputs = chain()
    for iteration in range(5):
        update(iteration)
        source_candidates = full(0, 1)[2]
        expected = [full(layer, 4, source_candidates)[0].squeeze(1).cpu().sort().values for layer in range(1, 5)]
        for actual, reference in zip(chain(), expected):
            torch.testing.assert_close(actual.cpu().sort().values, reference, rtol=0, atol=0)
        graph.replay()
        for actual, reference in zip(outputs, expected):
            torch.testing.assert_close(actual.cpu().sort().values, reference, rtol=0, atol=0)
        torch.testing.assert_close(candidates.cpu(), source_candidates.cpu(), rtol=0, atol=0)
        assert [value.data_ptr() for value in buffers] == addresses
        print(f"PASS page={page} batch={batch} changing replay={iteration}", flush=True)
    torch.npu.synchronize()
