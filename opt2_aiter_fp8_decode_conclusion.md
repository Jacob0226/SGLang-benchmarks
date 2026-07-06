# opt#2 — Porting ATOM's fp8 MLA decode core into SGLang: conclusion

**Date:** 2026-07-06  **HW:** MI355X (gfx950), ROCm  **Model:** GLM-5.2-MXFP4, TP4
**Container:** `jacchang_GLM5` (SGLang)  **Branch:** `Jacob0226/sglang`
`jacob/glm-mla-fp8-aiter-decode-wip` (commit `1461f11d`)

--------------------------------------------------------------------------------
## TL;DR

- The long-standing GPU fault when enabling SGLang's **aiter fp8 MLA decode core**
  (`--nsa-decode-backend aiter` + `--kv-cache-dtype fp8_e4m3`) is **fixed**. It now
  runs correctly: **GSM8K 0.945**, 0 invalid.
- But it is **~69% slower** than the current tilelang decode on MI355X:
  **median TPOT 24.45 ms vs 14.52 ms**.
- Root cause: **tilelang is the tuned MI355X default**, and ATOM's real advantage
  comes from its *whole* attention subsystem (seg-MLA `page_size>1` kernel + fused
  projections + shape tuning), **not the decode core kernel alone**. Swapping only
  the core kernel replaces a fast tilelang path with a slower generic aiter asm
  path → regression.
- => opt#2 is a **correctness / enablement** result, **not** a perf win on MI355X.
  Kept on a separate WIP branch; the clean +5% opt#1 branch was not touched.

--------------------------------------------------------------------------------
## What was changed (4 edits in `dsa_backend.py`)

Saved as `opt2_aiter_fp8_decode.patch`.

1. **`intra_batch_mode=True → False`** in all three spots (buffer sizing
   `get_mla_metadata_info_v1`, runtime `get_mla_metadata_v1`, and the kwarg passed
   into `mla_decode_fwd`) **and drop `topk=`**. The `True + topk=2048` path put the
   asm kernel into a sparse "intra-batch" schedule that assumes a **topk-strided**
   work layout, while the indices/indptr handed in are **compacted CSR** → the
   kernel addressed past the buffer → the opaque GPU OOB. ATOM's decode uses
   neither flag.
2. **`q_scale = ones` when Q is fp8** — `asm_mla.cu:227 mla_decode_stage1_asm_fwd:
   fp8 Q requires q_scale and kv_scale`. SGLang only set `kv_scale`.
3. **MLA output `o` forced bf16** — `kn_mla_reduce_v1 doesn't support output type
   Float8_e4m3fn`. `o` had inherited fp8 from `q`.
4. **`aiter_dsa_max_split_per_batch 64 → 16`** (match ATOM). No measurable effect
   (see below), i.e. splitting is not the bottleneck.

--------------------------------------------------------------------------------
## Validation (GLM-5.2-MXFP4, TP4, MI355X, isl1024 / osl512, conc4)

| decode backend | GSM8K (200q) | Median TPOT | Output tok/s |
|---|---|---|---|
| tilelang (baseline)      | —          | **14.52 ms** | 259.9 |
| aiter fp8 opt#2 (64 spl) | 0.945      | 24.55 ms     | 158.8 |
| aiter fp8 opt#2 (16 spl) | 0.945      | 24.45 ms     | 158.7 |

Identical launch flags except `--nsa-decode-backend {tilelang|aiter}`.

--------------------------------------------------------------------------------
## Why ATOM's MLA kernel is faster, but the SGLang integration got 69% slower

This is the key question. The short answer: **I only swapped one kernel, but
ATOM's speed comes from a whole tuned pipeline — and the kernel I swapped *to* is
slower than the tilelang kernel I swapped *from*.**

### 1. "ATOM is faster" was measured on the *whole attention block*, not the core
In the side-by-side (`analysis_GLM5.2/...SideBySide_GLM5.2.xlsx`), the per-decode-
layer **MLA section** is ATOM ~60 µs vs SGLang(tilelang) ~87 µs. That number sums
**everything** in the attention block: input norm, q/k/v projections
(`q_proj_and_k_up_proj`, `v_up_proj_and_o_proj`), RoPE + KV write, the decode core,
and the reduce. ATOM wins that *aggregate* because of several subsystem-level
choices, only one of which is the decode core:

- **seg-MLA (`page_size>1`) asm kernel.** ATOM runs `use_seg_mla` (gated on
  `ATOM_MLA_PAGE_SIZE>1`) with a padded q row stride and `num_kv_splits=None`
  (kernel auto). This is a *different, newer asm kernel* than the `page_size=1`
  persistent path. SGLang's aiter DSA decode uses the **`page_size=1`** path — the
  slower variant.
- **Fused projections** absorbed into fewer, well-shaped GEMMs.
- **Shape-specific tuning** (ATOM ships tuned configs for its exact GLM shapes).

### 2. What opt#2 actually did: replace *only* the core kernel
opt#2 changed SGLang's decode from **tilelang** → **aiter `mla_decode_fwd`
(page_size=1)**. Everything else (projections, norms, RoPE/KV write, the
compacted-CSR index build) stayed as SGLang's. So we did **not** import ATOM's
seg-MLA kernel, its fused projections, or its tuning — we imported the *generic*
aiter asm decode core on the `page_size=1` path.

### 3. The kernel we swapped *from* (tilelang) is the tuned MI355X default
SGLang/InferenceX use **tilelang NSA decode as the MI355X default** precisely
because it is well-tuned for gfx950 GLM decode. The aiter `mla_decode_fwd`
`page_size=1` path is a generic fallback, not tuned for this shape/HW. So the swap
went **fast tuned kernel → slow generic kernel** for this config.

### 4. Evidence that the core kernel path is the bottleneck (not scheduling)
`num_kv_splits 64 → 16` (ATOM's value) changed TPOT by <0.1 ms (24.55 → 24.45).
If over-splitting/reduce scheduling were the cost, 16 would have helped a lot at
bs≈4. It didn't → the cost is inside the **`page_size=1` asm decode kernel itself**,
consistent with it being the un-tuned/non-seg variant.

### 5. Net
Faster ATOM attention ≠ faster aiter decode core. To actually beat tilelang on
MI355X you'd need to port the **seg-MLA (`page_size>1`) path + fused projections +
tuned configs**, i.e. the whole subsystem — a much larger change than the 4-line
metadata/dtype fix that unblocked correctness. The 4 fixes are still valuable:
they make the aiter fp8 decode path *functionally usable* in SGLang (which it
wasn't before), and they document the exact ATOM↔SGLang contract mismatch.

--------------------------------------------------------------------------------
## Caveats / not yet done
- Only one bench point (conc4, isl1024/osl512). Higher concurrency / longer
  contexts could shift the tilelang-vs-aiter gap, but tilelang is expected to keep
  the lead on MI355X.
- Did **not** capture a kernel trace of the aiter-in-SGLang run; §4 above is
  inferred from the split-count experiment + architecture, not from a per-kernel
  profile. A trace would confirm the core-kernel attribution.
- seg-MLA (`page_size>1`) path not attempted in SGLang.

--------------------------------------------------------------------------------
## Artifacts
- Patch: `opt2_aiter_fp8_decode.patch`
- Full spec / diagnosis: `GLM52_fp8_decode_port_spec.md`
- WIP branch: `Jacob0226/sglang` `jacob/glm-mla-fp8-aiter-decode-wip` (`1461f11d`)
- opt#1 (landed, +5%): `Jacob0226/sglang` `jacob/glm-mla-fp8-absorbed-bmm`
