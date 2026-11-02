# QliCandidateGather

Packs visible, source-generated QLI candidate blocks from paged INT8 K and FP16
scale caches into fixed-address output tensors. The A3 model path shares these
outputs across indexer consumers within one forward. This operator does not
compute QK scores or TopK.

## Interface

```python
torch.ops._C_ascend.qli_candidate_gather(
    key, scale, sorted_candidates, block_table, cu_seqlens_q, seq_lens_k,
    key_page_stride, scale_page_stride, page_size,
    packed_key, packed_scale, positions, lengths,
)
```

All tensors must be contiguous and on the same device. Outputs are supplied by
the caller and mutated in place; the return value is `None`.

| Tensor | Shape | Dtype | Meaning |
| --- | --- | --- | --- |
| `key` | `[physical_rows, 128]` | int8 | Raw physical K storage, including page gaps |
| `scale` | `[physical_scale_elements]` | float16 | Raw physical scale storage, including page gaps |
| `sorted_candidates` | `[T, C]` | int32 | Unique block IDs in ascending order, each covering eight tokens |
| `block_table` | `[B, max_pages]` | int32 | Logical to physical page mapping |
| `cu_seqlens_q` | `[B+1]` | int32 | TND query prefix sums |
| `seq_lens_k` | `[B]` | int32 | Per-request uncompressed K lengths |
| `packed_key` | `[T*C*8, 128]` | int8 | Gathered candidate K |
| `packed_scale` | `[T*C*8]` | float16 | Gathered scales |
| `positions` | `[T, C*8]` | int32 | Original token positions |
| `lengths` | `[T]` | int32 | Valid prefix length for each packed query |

`key_page_stride` is measured in 128-element rows; `scale_page_stride` is in
FP16 elements. Both must be at least `page_size`, which must be a positive
multiple of eight. C is a positive multiple of 32, at most 2048; B is at most
2048; `max_pages` is at most 8192. The QLI consumer further requires C to be a
multiple of 64. A zero T returns without launching the gather kernel.

The caller converts `-1` candidate padding to `2^24-1` before sorting. This
sentinel and admitted nonnegative block IDs are exactly representable by the
helper's FP32 sort. Logical positions and page mappings of valid candidates
must be in range; the native kernel bounds physical reads but does not validate
all input values or deduplicate candidates.

For query row `r` in request `b`, visibility ends at
`seq_lens_k[b] - cu_seqlens_q[b+1] + r + 1`. The output valid prefix excludes
causal-tail tokens and padding. A partial last block may copy extra physical
tokens; consumers must honor `lengths`. Entirely invisible blocks have zero K
and scale. `positions` outside the valid prefix must not be consumed.

## Sharing contract

The source overwrites the active output rows on every forward or captured graph
replay before consumers read them on the dependent stream. Consumers must share
the same candidate source, cache owner, page table, query row order and
uncompressed coordinate system. They retain separate Q and weights. Storage
must not be shared across concurrently executing forwards.

The `CandidateKBuffer` helper initializes packed metadata outside forward and
bounds the payload allocation. Its `select` method invokes QLI mode 5 and
restores original positions. See the [QLI model integration](../quant_lightning_indexer_v2/README.md)
for eligibility, the opt-in switch and fallback behavior.
