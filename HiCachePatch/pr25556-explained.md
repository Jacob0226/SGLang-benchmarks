# Why `--page-size` is invisible to the AITER MLA kernel, and where the bug lives

Companion explainer for `pr25556-aiter-mla-page-size.sh`. Walks through
a 4096-token request to make concrete what `page_size=1` vs `page_size=64`
actually changes — and why the kernel doesn't care, but the framework's
preallocated buffers absolutely do.

## Setup

KV cache pool with 100,000 slots. Each slot stores one token's KV.
A request needs 4096 tokens worth of KV space.

## Allocator layer (this layer DOES care about page_size)

### `page_size = 1`

Pull 4096 individual free slots from the free list. They are scattered
because nothing forces contiguity:

```
slot_ids = [12, 7, 88, 4, 99, 23, 7421, 33, ..., 1023]   # shape (4096,)
```

Free-list bookkeeping: 4096 operations.

### `page_size = 64`

Free list is grouped into "pages" of 64 slots. Total pool has
`100,000 / 64 ≈ 1562` pages. For 4096 tokens we need 64 pages
(`4096 / 64`). Pull 64 free pages:

```
page_ids = [5, 17, 42, 3, ..., 88]                       # shape (64,)
```

Note: pages are still scattered relative to each other (page 5 is not
next to page 17), but each page's 64 internal slots are by definition
contiguous.

Free-list bookkeeping: 64 operations — 64× cheaper than `page_size=1`.

## Allocator's last step: flatten page ids into token slot ids

This is the key step the PR description highlights.
`PagedTokenToKVPoolAllocator.alloc` returns:

```python
out_indices = out_pages[:, None] * page_size + arange(page_size)
out_indices = out_indices.reshape(-1)
```

### `page_size = 1`

```
out_pages = [12, 7, 88, ..., 1023]                       # shape (4096,)

out_pages[:, None] * 1 + arange(1)
  = [[12], [7], [88], ..., [1023]]                       # shape (4096, 1)

reshape:
  = [12, 7, 88, ..., 1023]                               # shape (4096,)
```

### `page_size = 64`

```
out_pages = [5, 17, 42, ..., 88]                         # shape (64,)

out_pages[:, None] * 64 + arange(64)
  = [[5*64+0, 5*64+1, ..., 5*64+63],     # page 5  → slots 320..383
     [17*64+0, ..., 17*64+63],           # page 17 → slots 1088..1151
     ...
     [88*64+0, ..., 88*64+63]]           # page 88 → slots 5632..5695
  shape (64, 64)

reshape:
  = [320, 321, ..., 383,                 # page 5
     1088, 1089, ..., 1151,              # page 17
     2688, ..., 2751,                    # page 42
     ...
     5632, ..., 5695]                    # shape (4096,)
```

**The output shape is identical**: a 1D `int32[4096]` array of absolute
token slot IDs. Page identity is gone. The only structural difference is
that the `page_size=64` ID sequence happens to show local contiguity in
groups of 64 — but that's a value-distribution property, not a
shape/granularity property.

## `req_to_token` (scheduler / metadata layer)

Whatever the allocator returned gets stored verbatim:

```
req_to_token[req_id, :4096] = [12, 7, 88, ..., 1023]                 # page_size=1
req_to_token[req_id, :4096] = [320, 321, ..., 383, 1088, ...]        # page_size=64
```

**Same shape, both `[max_context_len]` entries per request, both stored
at token granularity.**

## What the AITER MLA kernel actually sees

When a decode (or prefill) step runs, the scheduler hands the kernel a
1D `kv_indices` array:

```
kv_indices = [12, 7, 88, ..., 1023]              # page_size=1
kv_indices = [320, 321, ..., 5695]               # page_size=64
shape: (4096,)
```

The kernel's loop is:

```
for i in 0..4096:
    K[i] = KV_pool[kv_indices[i]]      # gather one token
    V[i] = ...
# attention math on K, V
```

This loop has **no awareness of pages**. It does 4096 gather operations
regardless. The slot ID values being locally contiguous (page_size=64)
vs scattered (page_size=1) has zero impact on shape, computation count,
or correctness — at most it slightly affects L1/L2 cache hit rate, but
that's a memory-subsystem side-effect, not a kernel-logic difference.

## So what changes across page sizes?

| Layer | Sensitive to page_size? | Why |
|---|---|---|
| Allocator free list | **Yes** — `page_size=64` is ~64× cheaper bookkeeping | larger granular units, fewer operations |
| `out_indices` shape | No | always flattened to 1D token-granularity array |
| `req_to_token` shape | No | always `[max_context_len]` per req, token-granularity |
| `kv_indices` shape into kernel | No | always 1D, token-granularity |
| AITER MLA gather kernel | No | gathers N tokens by ID, doesn't read page IDs |
| HBM/L2 access pattern | **Yes, indirectly** | larger pages → groups of 64 contiguous loads → better cache locality |

## Why PR #25556 says "decode flat, prefill +21%"

PR's measured numbers:

| Conc | decode page=1 | decode page=64 | Δ | prefill page=1 | prefill page=64 | Δ |
|---|---|---|---|---|---|---|
| 64  | 2611 | 2615 | +0.2% | 34407 | 41786 | **+21.4%** |
| 128 | 3251 | 3318 | +2.1% | 40944 | 42148 | +2.9% |
| 256 | 3832 | 3897 | +1.7% | 40788 | 42625 | +4.5% |

- **Decode is flat** (≤2% across all conc): each decode step adds 1
  token per request. Almost no allocator / metadata work happens
  per step — the kernel inner-loop dominates, and that's identical.
- **Prefill is noticeably faster at low concurrency** with
  `page_size=64`: a single 8k prefill triggers a lot of allocator
  operations (8k token slots to find). The 64× allocator speedup
  matters here. At higher conc the prefill is more batched and the
  win shrinks to single digits.
- This is exactly what the PR claims: page_size only changes
  allocator / metadata overhead, not kernel work.

## Why the buggy buffer crashes anyway (fix #2)

"Kernel is page-agnostic" is the logical truth. But to make the kernel
work, the framework has to **pre-allocate** the `kv_indices` buffer
that the kernel reads. And here, the buggy code in
`init_cuda_graph_state` was inconsistent:

```python
# in init_cuda_graph_state — wrong for AITER MLA:
max_num_blocks_per_seq = ceil(self.max_context_len / self.page_size)
self.cuda_graph_kv_indices = torch.zeros(max_bs * max_num_blocks_per_seq, ...)
```

With `max_context_len=163840` and `page_size=64`:

```
max_num_blocks_per_seq = ceil(163840 / 64) = 2560
buffer size = max_bs * 2560   # block granularity
```

But the buffer is filled by `create_flashinfer_kv_indices_triton`
which writes at **token granularity** using `kv_indptr = cumsum(seq_lens)`.
For a single 8192-token decode context that's 8192 entries. The buffer
reserved 2560 entries per seq → **3.2× over-write per seq → silent
out-of-bounds → GPU memory access fault on all TP ranks**.

With `page_size=1`:

```
max_num_blocks_per_seq = ceil(163840 / 1) = 163840
buffer size = max_bs * 163840   # already token granularity — correct
```

The bug never fires.

So the duality is:

- The **kernel is correct** to not know about pages: it always sees a
  flat token-granularity gather list.
- The **framework is wrong** to size the buffer with `page_size` in the
  denominator: the kernel never used that granularity, so dividing by
  it just makes the buffer too small.

Fix #2 forces the buffer to token granularity for the MLA path, which
matches what the kernel actually requires. PR upstream took it out
during review; this repo's `pr25556-aiter-mla-page-size.sh` puts it
back with the gemini-bot review suggestion (extend the condition to
also catch the MLA + unified-attention case).

## Local verification recap

DeepSeek-R1-0528 FP8, MI355X TP=8, AITER backend,
`SGLANG_AITER_FP8_PREFILL_ATTN=0`, ISL=8192 OSL=1024 conc=16,
num-prompts=160, warmup=32:

| Setup | Median TTFT | GPU faults |
|---|---|---|
| `page_size=1` (InferenceX baseline) | 204.85 ms | 0 |
| `page_size=64`, no patches | — | 8 / 8 ranks |
| `page_size=64`, PR upstream fix #1 only | — | 8 / 8 ranks |
| `page_size=64`, this repo's `pr25556-aiter-mla-page-size.sh` | 207.64 ms | 0 |

`page_size=64` ends at +1.4% TTFT vs `page_size=1` — within noise.
Confirms the PR's analytical claim that, once metadata is consistent
with the kernel's token-granularity view, page size is a pure
allocator-side knob.
