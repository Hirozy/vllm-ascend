# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.attention import dsa_v41
from vllm_ascend.models.deepseek_v41 import indexer, model


@pytest.mark.parametrize(
    "enabled,supported,width", [(True, True, 128), (False, True, 128), (True, False, 128), (True, True, 64)]
)
def test_model_allocates_one_shared_buffer_only_for_eligible_local_layers(monkeypatch, enabled, supported, width):
    class Decoder:
        def __init__(self, layer_idx):
            self.self_attn = SimpleNamespace(role=SimpleNamespace(layer_idx=layer_idx), candidate_k_buffer=None)

    layers = [Decoder(i) for i in range(20, 40)]
    self = SimpleNamespace(
        config=SimpleNamespace(index_head_dim=width, index_n_heads=32),
        layers=layers,
        topk_indices_buffer=torch.empty(1),
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=262144), scheduler_config=SimpleNamespace(max_num_batched_tokens=158)
    )
    topology = SimpleNamespace(index_topk=512, candidate_topk_blocks=2048)
    factory = Mock()
    monkeypatch.setattr(model.ascend_envs, "VLLM_ASCEND_QLI_CANDIDATE_REUSE", enabled)
    monkeypatch.setattr(model, "get_current_hardware_profile", lambda: SimpleNamespace(supports=lambda cap: supported))
    monkeypatch.setattr(model, "DeepseekV41DecoderLayer", Decoder)
    monkeypatch.setattr(model, "candidate_reuse_layers", lambda topology, local: (20, 24, 28, 32, 36))
    monkeypatch.setattr(model, "CandidateKBuffer", factory)
    model.DeepseekV41Model._initialize_candidate_reuse(self, vllm_config, topology)
    eligible = enabled and supported and width == 128
    assert factory.call_count == int(eligible)
    for layer in layers:
        expected = factory.return_value if eligible and layer.self_attn.role.layer_idx % 4 == 0 else None
        assert layer.self_attn.candidate_k_buffer is expected


@pytest.mark.parametrize("source,eligible", [(True, True), (False, True), (True, False), (False, False)])
def test_source_packs_before_consumers_and_overwrites_each_forward(monkeypatch, source, eligible):
    cache = (torch.zeros((2, 64, 1, 128), dtype=torch.int8), torch.ones((2, 64, 1, 1), dtype=torch.float16))
    tokens, count, topk = 2, 64, 4
    shared = SimpleNamespace(
        candidates=torch.full((tokens, 1, count), -1, dtype=torch.int32),
        topk_indices=torch.empty((tokens, topk), dtype=torch.int32),
        candidate_lengths=None,
        topk_lengths=None,
    )
    packed = Mock()
    packed.supports.return_value = eligible
    produced = torch.arange(tokens * count, dtype=torch.int32).reshape(tokens, 1, count)

    def pack(candidates, source_cache, metadata):
        assert source_cache is cache
        assert candidates.data_ptr() == shared.candidates.data_ptr()
        torch.testing.assert_close(candidates, produced)

    packed.prepare.side_effect = pack
    selection = Mock(return_value=(torch.ones((tokens, topk), dtype=torch.int32), produced))
    attn = SimpleNamespace(
        shared_state=shared,
        candidate_k_buffer=packed,
        indexer=SimpleNamespace(packed_cache_ops=None, select_projected=selection),
    )
    role = SimpleNamespace(
        has_long_context=True,
        is_index_source=True,
        is_candidate_source=source,
        uses_candidate_filter=not source,
        compress_ratio=1,
    )
    impl = object.__new__(dsa_v41.AscendDSAV41Impl)
    impl.role, impl.topology, impl.index_k_source_prefix = (
        role,
        SimpleNamespace(candidate_topk_blocks=count, candidate_block_size=8),
        "owner.indexer.k_cache",
    )
    monkeypatch.setattr(
        dsa_v41,
        "get_forward_context",
        lambda: SimpleNamespace(no_compile_layers={impl.index_k_source_prefix: SimpleNamespace(kv_cache=[cache])}),
    )
    metadata = SimpleNamespace(swa=SimpleNamespace(num_actual_tokens=tokens), indexer=SimpleNamespace(cache=object()))
    prepared = SimpleNamespace(
        query=object(), weights=object(), quantized_query=object(), query_scale=object(), quantize_done=None
    )
    for iteration in range(2):
        produced.add_(count * iteration)
        actual = impl._select_sparse_indices(attn, torch.empty(tokens, 8), None, None, None, None, metadata, prepared)
        assert actual.data_ptr() == shared.topk_indices.data_ptr()
        assert selection.call_args.kwargs["candidate_reuse"] is (packed if eligible and not source else None)
    assert packed.prepare.call_count == (2 if eligible and source else 0)


def test_indexer_packed_dispatch_preserves_output_processing(monkeypatch):
    selected = torch.tensor([[9, 2, -1, 5]], dtype=torch.int32)
    reuse = Mock()
    reuse.select.return_value = selected
    output = torch.empty_like(selected)
    lengths = torch.empty((1, 1), dtype=torch.int32)
    postprocess = Mock(return_value=output)
    monkeypatch.setattr(indexer, "prepare_indexer_indices", postprocess)
    self = SimpleNamespace(index_topk=4, compress_ratio=1, packed_cache_ops=None)
    q, w, qs = torch.zeros((1, 32, 128), dtype=torch.int8), torch.ones(1, 32), torch.ones(1, 32)
    positions = torch.tensor([12])
    candidates = torch.zeros((1, 1, 64), dtype=torch.int32)
    result, returned_candidates = indexer.DeepseekV41Indexer.select_projected(
        self,
        q,
        w,
        positions,
        None,
        SimpleNamespace(max_cache_seq_len=13),
        is_candidate_source=False,
        uses_candidate_filter=True,
        candidate_topk_blocks=64,
        candidate_block_size=8,
        candidates=candidates,
        quantized_query=q,
        query_scale=qs,
        candidate_reuse=reuse,
        indices_output=output,
        topk_lengths=lengths,
    )
    assert result is output and returned_candidates is candidates
    assert reuse.select.call_args.args[0] is q
    assert reuse.select.call_args.args[1].dtype == torch.float16
    assert reuse.select.call_args.args[2] is qs
    postprocess.assert_called_once_with(selected, positions, 1, indices_output=output, lengths_output=lengths)


@pytest.mark.parametrize("skip,reuse,mode", [(False, False, 2), (True, False, 4), (False, True, 4)])
def test_ineligible_consumer_respects_fallback_switches(monkeypatch, skip, reuse, mode):
    monkeypatch.setattr(indexer.envs, "VLLM_ASCEND_QLI_CANDIDATE_SKIP", skip)
    monkeypatch.setattr(indexer.envs, "VLLM_ASCEND_QLI_CANDIDATE_REUSE", reuse)
    native = Mock(return_value=(torch.zeros((1, 1, 4), dtype=torch.int32), None, None))
    monkeypatch.setattr(torch.ops._C_ascend, "npu_quant_lightning_indexer_v3", native, raising=False)
    monkeypatch.setattr(indexer, "prepare_indexer_indices", lambda selected, *args, **kw: selected)
    monkeypatch.setattr(indexer, "wait_for_device_metadata", Mock())
    self = SimpleNamespace(index_topk=4, compress_ratio=1, packed_cache_ops=None)
    metadata = SimpleNamespace(
        max_cache_seq_len=13,
        query_start_loc=None,
        cache_seq_lens=None,
        cmp_residual=None,
        max_query_len=1,
        qli_metadata=None,
        block_table=None,
    )
    q, w = torch.zeros((1, 32, 128), dtype=torch.int8), torch.ones(1, 32)
    indexer.DeepseekV41Indexer.select_projected(
        self,
        q,
        w,
        None,
        (torch.zeros((1, 64, 1, 128), dtype=torch.int8), torch.ones(1, 64, 1, 1)),
        metadata,
        is_candidate_source=False,
        uses_candidate_filter=True,
        candidate_topk_blocks=64,
        candidate_block_size=8,
        candidates=torch.zeros((1, 1, 64), dtype=torch.int32),
        quantized_query=q,
        query_scale=w,
    )
    assert native.call_args.kwargs["candidate_mode"] == mode
