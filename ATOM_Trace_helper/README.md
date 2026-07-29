# ATOM_Trace_helper

Helper scripts for **SGLang vs ATOM** kernel-level trace comparison on MI355X
(GLM-5.2-MXFP4). All scripts are run from the `SGLang-benchmarks/` root.

> NOTE: `analyze_trace.py` intentionally stays at the repo root — it is the shared
> core also used by the MI355X-vs-B200 flow and by several `tools/` scripts.

Two analyzers are needed (unlike the MI355X-vs-B200 flow, which only needs
`analyze_trace.py` + `compare_breakdown.py`) because ATOM traces carry no
`nn.Module` python_function tree and mark forwards with `prefill[]` / `decode[]`
annotations, and because the two stacks use different module names, which defeats
`compare_breakdown.py`'s LCS row alignment.

## Pipeline

| step | script | purpose |
|------|--------|---------|
| 1 | `../analyze_trace.py` | SGLang step1 + step3 (graph-on timing + graph-off structure, per call site) |
| 2 | `analyze_atom_trace.py` | ATOM step1 + step3, mapped to SGLang's classification |
| 3 | `compare_glm52_sglang_atom.py` | per-forward bucket comparison, CSV + `--xlsx` call-order workbook (authoritative timing; each kernel counted once) |
| 4 | `callorder_sidebyside.py` | side-by-side call-order xlsx with module / call-site labels (consumes step3 + the bucket CSV) |
| driver | `../tools/regen_glm52_sidebyside.sh` | rebuilds everything: prefill + decode, SGLANG old/new + ATOM, both call-order workbooks |

Steps 1 and 2 **must** be given `--forward-match` so every number describes ONE
forward; without it each kernel is averaged over the whole trace, which mixes
different batch/token sizes.

## Example

```bash
cd ~/SGLang-benchmarks
bash tools/regen_glm52_sidebyside.sh
```

## Diagnostics (in `../tools/`)

| script | use it when |
|--------|-------------|
| `verify_step3_against_trace.py` | after changing an analyzer or switching traces: checks every kernel's avg / count / Σ in a step3 workbook against the raw trace |
| `comm_skew_split.py` | a comm bucket looks slow: splits all-reduce into pure communication vs waiting for the slowest rank (needs all TP ranks' traces) |
| `window_kernel_diff.py` | find which kernels differ between two windows (two ranks, or two stacks); `--stats` for per-call distributions, `--show-args` for launch configs |
