# ATOM_Trace_helper

Helper scripts for **SGLang vs ATOM** kernel-level trace comparison on MI355X
(GLM-5.2-MXFP4). All scripts are run from the `SGLang-benchmarks/` root.

> NOTE: `analyze_trace.py` intentionally stays at the repo root — it is the shared
> core also used by the MI355X-vs-B200 flow and by several `tools/` scripts.

## Pipeline

| step | script | purpose |
|------|--------|---------|
| 1 | `trim_trace_to_forward.py` | trim a trace to ONE forward (union CPU+GPU step window) |
| 2 | `../analyze_trace.py` | SGLang step3 (graph-on timing + graph-off structure, per-call-site) |
| 3 | `analyze_atom_trace.py` | ATOM step3 (mapped to SGLang's classification) |
| 4 | `compare_glm52_sglang_atom.py` | per-forward bucket 3-way CSV (authoritative timing; each kernel counted once) |
| 5 | `callorder_sidebyside.py` | side-by-side call-order xlsx (consumes step3 + summary CSV) |
| driver | `regen_sglang_step3.sh` | regenerate the 4 SGLANG step3 files in one shot |

A fuller end-to-end driver (prefill+decode, SGLANG old/new + ATOM + callorder) lives
at `tools/regen_glm52_sidebyside.sh`.

## Example

```bash
cd ~/SGLang-benchmarks
bash ATOM_Trace_helper/regen_sglang_step3.sh
```
