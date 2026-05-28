# HiCachePatch

Local patches for sglang on ROCm MI355X to remove silent behaviour
that breaks reproducible benchmarking and to fix a real bug that
crashes the AITER MLA backend with `--page-size > 1`.

All patches are **idempotent shell scripts** that edit a sglang
checkout in place. Default target is `/sgl-workspace/sglang` (the
container's editable install); pass a path to patch a different
checkout.

**Do NOT run on `~/PR/sglang`** -- that is the host edit repo where
your branches originate; the patches must not pollute pushes.

## Files

| File | What it does |
|---|---|
| `apply-all.sh` | Apply all patches below |
| `no-aiter-mem-fraction.sh` | Remove the AMD aiter long-context `mem_fraction_static *= 0.85` silent adjustment in `server_args.py`. Without this, MI355X and B200 use different KV pools for the same `--mem-fraction-static` flag value. |
| `pr25556-aiter-mla-page-size.sh` | Apply both fixes from sglang PR #25556 (fix #1 from upstream + fix #2 manually re-applied with gemini-bot's `use_mla` suggestion). Required to use `--page-size > 1` with AITER MLA backend without crashing. |
| `pr25556-explained.md` | Concrete walkthrough (4096-token example) of how `page_size` flows through allocator → metadata → kernel, why the kernel is page-agnostic, and where the buggy buffer makes the framework disagree with the kernel. |

## PR #25556 patch details

Upstream PR sgl-project/sglang#25556 titled _"Fix correctness for
AITER MLA backend with `--page-size > 1`"_ promised to fix two bugs
but the version currently on the PR (1 commit, `6db65bd`) only
contains fix #1. Fix #2 was force-pushed away during code review --
gemini-bot pointed out the conditional missed an MLA case, and the
author dropped fix #2 to let fix #1 merge first.

### fix #1 (from upstream, `python/sglang/srt/layers/attention/aiter_backend.py: make_mla_prefill_ps_meta_data`)

- `kvlen_granularity = max(128, self.page_size)` → `kvlen_granularity = 128`
- `block_size = self.page_size` → `block_size = 1`
- Only affects the fp8 prefill attention path (gated by
  `SGLANG_AITER_FP8_PREFILL_ATTN`, default `True` on gfx95).
- Without it: `gsm8k` 0.975 → 0.005 when `page_size > 1` and fp8
  prefill is on.

### fix #2 (re-applied, `python/sglang/srt/layers/attention/aiter_backend.py: init_cuda_graph_state`)

`cuda_graph_kv_indices` was sized at block granularity
(`max_bs * ceil(max_context_len / page_size)`) but is filled by
`create_flashinfer_kv_indices_triton` at token granularity
(`kv_indptr = cumsum(seq_lens)`). With `page_size > 1` the buffer is
`page_size`× too small → silent Triton overrun on any non-trivial
decode context → **GPU memory access fault on all TP ranks**.

The bug fires regardless of `SGLANG_AITER_FP8_PREFILL_ATTN` since it
is in the cuda graph decode buffer, not the fp8 prefill metadata.

Re-applied with gemini-bot's review suggestion: extend the
token-granularity condition to also cover the case where MLA is used
even if unified attention is enabled
(`if not self.use_triton_unified_attention or self.use_mla:`).

## Reproducer (MI355X, lmsysorg/sglang:v0.5.12-rocm700-mi35x)

DeepSeek-R1-0528 FP8, TP=8, AITER backend, `SGLANG_AITER_FP8_PREFILL_ATTN=0`,
ISL=8192 OSL=1024 conc=16, num-prompts=160, warmup=32:

| Setup | Median TTFT | Result |
|---|---|---|
| page_size=1 (InferenceX baseline) | 204.85 ms | ✓ |
| page_size=64, no patches | — | ✗ Memory access fault 8/8 GPUs |
| page_size=64, only PR upstream fix #1 | — | ✗ Memory access fault 8/8 GPUs |
| page_size=64, this `pr25556-aiter-mla-page-size.sh` | 207.64 ms | ✓ |

Within-noise match (+1.4%) confirms PR's claim that decode kernel
performance is flat across page sizes once the metadata bugs are
fixed.

## Usage

```bash
# Apply everything (recommended):
bash /home/jacchang/local-patches/HiCachePatch/apply-all.sh

# Revert (inside the container, on the sglang checkout):
cd /sgl-workspace/sglang
git checkout -- python/sglang/srt/server_args.py \
                python/sglang/srt/layers/attention/aiter_backend.py
```
