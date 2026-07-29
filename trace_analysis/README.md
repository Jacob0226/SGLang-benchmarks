# trace_analysis/

Kernel-level PyTorch-profiler trace analysis for SGLang, and the two comparison
flows built on it. All scripts are run from the `SGLang-benchmarks/` root.

```
sglang/       SGLang-side analyzer (the core; everything else consumes its step3 xlsx)
atom/         ATOM-side analyzer, mapped onto the same step3 schema
compare/      cross-run / cross-stack comparison workbooks
diagnostics/  verify a workbook, or dig into one window / one kernel / one stream
```

## The step3 workbook is the interface

`sglang/analyze_trace.py` needs two traces of the same run: the **graph-ON** one for
timing (CUDA graph is how the model really runs) and the **graph-OFF** one for
structure (only that one carries the `nn.Module` python_function tree). It emits

| step | file | content |
|------|------|---------|
| 1 | `step1_kernel_stats*.xlsx` | per kernel NAME: count, Σ, avg, % of the forward |
| 2 | — | layer structure printed to stdout (module classes, layer count) |
| 3 | `step3_layer_breakdown*.xlsx` | one row per **call site**, aggregated over all layers |

Every consumer below reads step3, so its columns are worth knowing:

- `LayerType` — which layer types reach this call site, as tags (`full+MLP`,
  `full+MoE`, `shared+MoE`, or `all`); the **LayerTypes** sheet spells the tags out
  with their layer indices.
- `LaunchCount` (`LaunchCnt` in the side-by-side workbooks) — launches of **this call
  site** in the forward. On the SGLang side that is one per layer, so it equals the
  number of layers of the types in `LayerType`, and `Σ = Avg × LaunchCount` covers the
  forward without extrapolating from one representative layer.
- `KernelCount_fwd` / `KernelSum_ms_fwd` (`KernelCnt` / `KernelΣ_ms`) — the kernel
  **name**'s totals in the same forward regardless of call site, repeated on every row
  sharing the name, so never sum them. They are independent ground truth from the
  graph-ON trace: `Σ` of the rows should match, and `analyze_trace.py` prints a
  coverage report listing any kernel where it does not. GLM-5.2 lands at ~99.8% of a
  prefill forward; the gap is kernels outside the decoder layers (lm_head, sampling).

Layer types come from **what a layer actually runs**, not from its sub-module
signature — which is why GLM-5.2 shows three (`full+MLP` / `full+MoE` /
`shared+MoE`) instead of the two that `DeepseekV2MLP` vs `DeepseekV2MoE` suggests.
See the `glm5.2-dsa-layer-structure` skill for the architecture behind that.

**Always pass `--forward-match`.** Without it each kernel is averaged over the whole
trace, which blends different batch/token sizes: the GLM-5.2 TP all-reduce came out
as 713 µs/call instead of the real 861 µs/call for the `bs=3` prefill forward.

## Flow A — same stack, two GPUs (MI355X vs B200)

| step | script |
|------|--------|
| 1 | `sglang/analyze_trace.py` on each machine's trace pair |
| 2 | `compare/compare_breakdown.py` — LCS-aligns the two step3 workbooks row by row |

## Flow B — same GPU, two stacks (SGLang vs ATOM)

ATOM needs its own analyzer: its traces carry no `nn.Module` tree and mark forwards
with `prefill[]` / `decode[]` annotations, and the two stacks use different module
names, which defeats `compare_breakdown.py`'s row alignment.

| step | script | purpose |
|------|--------|---------|
| 1 | `sglang/analyze_trace.py` | SGLang step1 + step3 |
| 2 | `atom/analyze_atom_trace.py` | ATOM step1 + step3, mapped to SGLang's classification |
| 3 | `compare/compare_glm52_sglang_atom.py` | per-forward bucket comparison, CSV + `--xlsx` call-order workbook (authoritative timing; each kernel counted once) |
| 4 | `compare/callorder_sidebyside.py` | side-by-side call-order xlsx with module / call-site labels (consumes step3 + the bucket CSV) |
| driver | `regen_glm52_sidebyside.sh` | rebuilds everything: prefill + decode, SGLANG old/new + ATOM, both call-order workbooks |

```bash
cd ~/SGLang-benchmarks
bash trace_analysis/regen_glm52_sidebyside.sh
```

One specialized comparator also lives in `compare/`:
`glm52_full_layer_sidebyside.py` puts **two concrete neighbouring decode layers**
side by side (a shared-indexer one next to a full-indexer one, ATOM | SGLang) by
segmenting the step positionally — `fused_qk_rmsnorm` anchors each layer, the TP
all-reduce before it is the boundary — because neither stack's no-graph decode trace
has a per-layer module tree. step3 gives you the same numbers averaged per layer type
for SGLang (330 µs shared vs 426 µs full on MI355X, which this tool reproduces), so
reach for it when you want the two kernel lists in call order, or anything at all on
the ATOM side. It warns when kernels of the chosen layer have no timing in the timed
window: that happens when the no-graph trace only captured a prefill forward (true of
some ATOM runs), and the subtotals of that column are then an undercount.

## diagnostics/

| script | use it when |
|--------|-------------|
| `verify_step3_against_trace.py` | after changing an analyzer or switching traces: checks every kernel's avg / count / Σ in a step3 workbook against the raw trace |
| `window_kernel_diff.py` | find which kernels differ between two windows (two ranks, or two stacks); `--stats` for per-call distributions, `--show-args` for launch configs |
| `comm_skew_split.py` | a comm bucket looks slow: splits all-reduce into pure communication vs waiting for the slowest rank (needs all TP ranks' traces) |
| `trace_kernel_summary.py` | quick per-kernel totals of one trace, no structure needed |
| `extract_stream.py` | inspect kernels of a specific CUDA stream, or compare two streams |
| `analyze_trace_overlap.py` | measure stream concurrency / bubble time, e.g. dual-stream MoE |
