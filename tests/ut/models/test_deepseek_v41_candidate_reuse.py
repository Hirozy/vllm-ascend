# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.models.deepseek_v41 import candidate_reuse as reuse


@pytest.fixture
def topology():
    roles = [
        SimpleNamespace(
            layer_idx=i,
            kv_source_layer=20,
            compress_ratio=1,
            is_index_source=i in (20, 24, 28, 32, 36),
            is_candidate_source=i == 20,
            uses_candidate_filter=i >= 24,
        )
        for i in range(40)
    ]
    return SimpleNamespace(
        layers=roles,
        layer=roles.__getitem__,
        candidate_source_layer_id=20,
        candidate_topk_blocks=2048,
        candidate_block_size=8,
        index_topk=512,
    )


@pytest.mark.parametrize(
    "local,expected",
    [(range(40), (20, 24, 28, 32, 36)), (range(20, 30), (20, 24, 28)), (range(24, 40), ()), (range(24), ())],
)
def test_reuse_requires_local_source_owner_and_multiple_consumers(topology, local, expected):
    assert reuse.candidate_reuse_layers(topology, local) == expected
    topology.layer(20).kv_source_layer = 14
    assert reuse.candidate_reuse_layers(topology, range(20, 40)) == ()


def test_reuse_excludes_other_cache_owners_and_compression_ratios(topology):
    topology.layer(24).kv_source_layer = 24
    topology.layer(32).compress_ratio = 2
    assert reuse.candidate_reuse_layers(topology, range(40)) == (20, 28, 36)
    topology.layer(20).is_index_source = False
    assert reuse.candidate_reuse_layers(topology, range(40)) == ()


@pytest.mark.parametrize("count,block,topk", [(63, 8, 32), (4096, 8, 512), (2048, 16, 512), (64, 8, 512)])
def test_reuse_rejects_unsupported_or_insufficient_candidate_capacity(topology, count, block, topk):
    topology.candidate_topk_blocks, topology.candidate_block_size, topology.index_topk = count, block, topk
    assert reuse.candidate_reuse_layers(topology, range(40)) == ()


def make_buffer(rows=2, topk=3):
    metadata = Mock(side_effect=lambda *args, **kw: torch.tensor([kw["batch_size"]], dtype=torch.int32))
    gather = Mock()
    indexer = Mock()
    buffer = reuse.CandidateKBuffer(
        rows, 32, topk, 64, "cpu", metadata_op=metadata, gather_op=gather, indexer_op=indexer
    )
    return buffer, metadata, gather, indexer


def cache_and_metadata():
    # A slice through larger allocations models the Hybrid-cache page gaps.
    key = torch.zeros((8, 64, 1, 128), dtype=torch.int8)[::2]
    scale = torch.ones((12, 64, 1, 1), dtype=torch.float16)[::3]
    metadata = SimpleNamespace(
        block_table=torch.tensor([[1, 0, 3, 2]], dtype=torch.int32),
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        cache_seq_lens=torch.tensor([129], dtype=torch.int32),
        max_cache_seq_len=129,
    )
    return (key, scale), metadata


def test_payload_is_bounded_and_metadata_is_built_only_once(monkeypatch):
    bytes_per_row = 64 * 8 * (128 + 2 + 4) + 4
    monkeypatch.setattr(reuse, "CANDIDATE_BUFFER_BUDGET", bytes_per_row * 3)
    buffer, metadata, _, _ = make_buffer(rows=100000)
    assert buffer.max_rows == 3
    assert metadata.call_count == 3
    assert sum(t.numel() * t.element_size() for t in (buffer.key, buffer.scale, buffer.positions, buffer.lengths)) == (
        bytes_per_row * 3
    )


def test_eligibility_uses_layout_capacity_and_device_without_host_length_dispatch():
    buffer, _, _, _ = make_buffer()
    cache, metadata = cache_and_metadata()
    assert buffer.supports(2, cache, metadata, 1)
    metadata.max_cache_seq_len = 0
    assert buffer.supports(2, cache, metadata, 1)
    assert not buffer.supports(0, cache, metadata, 1)
    assert not buffer.supports(3, cache, metadata, 1)
    assert not buffer.supports(2, cache, metadata, 2)
    assert not buffer.supports(2, (cache[0].float(), cache[1]), metadata, 1)
    assert not buffer.supports(2, tuple(t.to("meta") for t in cache), metadata, 1)
    metadata.query_start_loc = metadata.query_start_loc.long()
    assert not buffer.supports(2, cache, metadata, 1)


def test_prepare_sorts_padding_and_keeps_physical_strides_and_output_addresses():
    buffer, metadata_op, gather, _ = make_buffer()
    cache, metadata = cache_and_metadata()
    candidates = torch.full((2, 1, 64), -1, dtype=torch.int32)
    candidates[:, 0, :3] = torch.tensor([16, 0, 7])
    pointers = [tensor.data_ptr() for tensor in (buffer.key, buffer.scale, buffer.positions, buffer.lengths)]
    for _ in range(2):
        buffer.prepare(candidates, cache, metadata)
        args = gather.call_args.args
        assert args[0].data_ptr() == cache[0].data_ptr()
        assert args[1].data_ptr() == cache[1].data_ptr()
        torch.testing.assert_close(args[2][:, :3], torch.tensor([[0, 7, 16]] * 2, dtype=torch.int32))
        assert (args[2][:, 3:] == reuse.INVALID_CANDIDATE).all()
        assert args[6:9] == (128, 192, 64)
        assert [tensor.data_ptr() for tensor in args[9:]] == pointers
    assert metadata_op.call_count == 2


def test_each_consumer_uses_own_query_and_restores_indices_and_invalid_slots():
    buffer, metadata_op, _, indexer = make_buffer()
    buffer.positions.copy_(torch.arange(buffer.capacity).repeat(2, 1) + 1000)
    indexer.return_value = (torch.tensor([[[511, 0, -1]]] * 2, dtype=torch.int32), None, None)
    weights = torch.ones((2, 32), dtype=torch.float16)
    scales = weights.clone()
    for value in range(4):
        query = torch.full((2, 32, 128), value, dtype=torch.int8)
        actual = buffer.select(query, weights, scales)
        assert indexer.call_args.args[0] is query
        assert indexer.call_args.kwargs["metadata"] is buffer.metadata[1]
        assert indexer.call_args.kwargs["seqused_k"].data_ptr() == buffer.lengths.data_ptr()
        assert indexer.call_args.kwargs["candidate_mode"] == 5
        torch.testing.assert_close(actual, torch.tensor([[1511, 1000, -1]] * 2, dtype=torch.int32))
    assert metadata_op.call_count == 2
