# trace_analysis/

Kernel-level PyTorch-profiler trace analysis for SGLang, and the two comparison
flows built on it. All scripts are run from the `SGLang-benchmarks/` root.

```
analyze/sglang_trace.py       SGLang-side analyzer (the core; everything else consumes its step3 xlsx)
analyze/atom_trace.py         ATOM-side analyzer, mapped onto the same step3 schema
compare/                      cross-run / cross-stack comparison workbooks
diagnostics/                  verify a workbook, or dig into one window / one kernel / one stream
run_glm52_sglang_vs_atom.sh   driver: SGLang vs ATOM on one GPU (Flow B)
run_glm52_sglang_vs_sglang.sh driver: same stack on two GPUs (Flow A)
```

Only the two analyzers read raw traces. Every comparison below reads their
workbooks, so no comparison re-implements forward segmentation and all of them
describe exactly the forward the analyzer isolated.

## The step3 workbook is the interface

`analyze/sglang_trace.py` needs two traces of the same run: the **graph-ON** one for
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
  graph-ON trace: `Σ` of the rows should match, and `analyze/sglang_trace.py` prints a
  coverage report listing any kernel where it does not. GLM-5.2 lands at ~99.8% of a
  prefill forward; the gap is kernels outside the decoder layers (lm_head, sampling).

Layer types come from **what a layer actually runs**, not from its sub-module
signature — which is why GLM-5.2 shows three (`full+MLP` / `full+MoE` /
`shared+MoE`) instead of the two that `DeepseekV2MLP` vs `DeepseekV2MoE` suggests.
See the `glm5.2-dsa-layer-structure` skill for the architecture behind that.

**Always pass `--forward-match`.** Without it each kernel is averaged over the whole
trace, which blends different batch/token sizes: the GLM-5.2 TP all-reduce came out
as 713 µs/call instead of the real 861 µs/call for the `bs=3` prefill forward.

## Flow A — two SGLang runs (MI355X vs B200, or before vs after a change)

| step | script |
|------|--------|
| 1 | `analyze/sglang_trace.py` on each run's trace pair |
| 2 | `compare/side_by_side.py --align lcs` — LCS-aligns the two step3 workbooks row by row |

```bash
python3 trace_analysis/compare/side_by_side.py --align lcs \
    --src MI355X MI355X/step3_layer_breakdown.xlsx \
    --src B200   B200/step3_layer_breakdown.xlsx \
    --out comparison.xlsx
```

`--align lcs` does not care what the two workbooks are, as long as the module names
match: two GPUs, or the same GPU before and after a PR. Add `--trace LABEL trace.gz`
to also report that side's union-busy time and overlap factor. An ATOM step3 workbook
is readable too (`analyze/atom_trace.py` writes the same schema on purpose), but the
module names differ, so the rows misalign — that is what Flow B exists for.

`run_glm52_sglang_vs_sglang.sh` is the worked example: MI355X 8PR vs B200
stock at conc4 and conc64, prefill and decode, into
`analysis_GLM5.2/Docker0729_8PR_SGLang_i8k_conc{4,64}/`. Two things it does that
a bucket table alone cannot, and that any cross-GPU comparison wants:

- `diagnostics/forward_overlap.py` per forward, because Σ kernel does not say
  whether a forward is slow from slow kernels or from serialised ones. GLM-5.2
  conc64 decode is the case in point — the two GPUs' Σ kernel are within 2%
  while wall time differs by 25%, all of it B200's 45-stream overlap.
- `diagnostics/comm_skew_split.py`, since all-reduce is prefill's largest gap
  and a collective's duration includes waiting for the slowest rank.

It also emits the buckets at **both** scopes. step3 matches graph-ON kernels to
the graph-OFF module tree by name, which fails when the two runs used different
batch sizes and the library picks shape-specialised kernels (B200's conc64
no-graph decode ran at bs=40, so its bs=64 nvjet GEMMs found no call site and
step3 reported 1.11 ms against MI355X's 5.73 ms; step1 needs no attribution and
says 6.05 vs 5.83 ms). Where the scopes agree, either is fine; where they do
not, step1 is ground truth.

## Flow B — same GPU, two stacks (ROCm SGLang vs ROCm ATOM)

ATOM needs its own analyzer: its traces carry no `nn.Module` tree and mark forwards
with `prefill[]` / `decode[]` annotations, and the two stacks use different module
names, which defeats `--align lcs`'s row alignment.

| step | script | purpose |
|------|--------|---------|
| 1 | `analyze/sglang_trace.py` | SGLang step1 + step3 |
| 2 | `analyze/atom_trace.py` | ATOM step1 + step3, mapped to SGLang's classification |
| 3 | `compare/glm52_buckets.py` | **how much**: sorts both sides' kernels into ~9 GLM-5.2 functional buckets (sparse-MLA attn, DSA indexer+topk, dense GEMM, MoE up/gate, MoE down, MoE routing, all-reduce, rmsnorm/quant, rope/kv-cache, other) → CSV |
| 4 | `compare/side_by_side.py --align none` | **which kernel, called from where**: lists each side's call sites in its own execution order with `Section > LeafModule`, `CallSite`, `LayerType`, `LaunchCnt`. No cross-side alignment — you match them by eye. `--summary-csv` embeds step 3's buckets at the bottom so detail and totals share one scale |
| driver | `run_glm52_sglang_vs_atom.sh` | rebuilds all of it, one run per SGLang variant: prefill + decode, buckets, call-order workbooks |

```bash
cd ~/SGLang-benchmarks
bash trace_analysis/run_glm52_sglang_vs_atom.sh                  # every variant
bash trace_analysis/run_glm52_sglang_vs_atom.sh Docker0729_8PR   # just one
FORCE_ATOM=1 bash trace_analysis/run_glm52_sglang_vs_atom.sh     # re-derive the ATOM reference
```

The ATOM side is derived once into `analysis_GLM5.2/.atom_reference_i8k_conc64/`
and copied into every variant, so the reference column is identical across all of
them. Its forward is selected by shape and then by median rather than pinned to a
label: an ATOM prefill is labelled with its context lengths (`prefill[bs=3
tok=16384 ctx=[8063, 7153, 1168]]`), which no other profile reproduces, so a
pinned label silently stops matching. Matching `bs=3 tok=16384` leaves 44
candidates whose middle 50% sit in 640.4–643.4 ms and `--pick median` takes
642.1 ms. `--forward-pick mean` would report 629.8 ms here, dragged down by a
single 128.5 ms fragment at a trace boundary — averaging only helps when the
matched forwards really are the same shape, which is why the SGLang analyzer
prints their Σ spread and warns when it is wide.

Bucket scope is whichever workbook you hand `glm52_buckets.py`, and it prints which
one it got: a **step3** file means decoder layers only, a **step1** file means the
whole forward with the lm_head / sampling tail included. Mixing the two across sides
is refused. Bucket rules live in `glm52_buckets.py` and `side_by_side.py` imports
them, so the summary at the bottom of a workbook and the CSV always agree.

`compare/glm52_full_layer_sidebyside.py` is the only hand-run tool left.

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
| `forward_overlap.py` | one forward is slower than the other but its kernels are not: splits wall into Σ kernel, union-busy, overlap factor and idle. Unlike `per_forward_wall.py` it takes kernels by timestamp rather than correlation id, so it works on graph-ON traces |
