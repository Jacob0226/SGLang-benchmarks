# GLM-5.3-Flash MXFP4 — MoE router GEMM tuning on MI355X (gfx950)

Target: the `gemm_a16w16` call behind `DeepseekV2MoE > MoEGate`, i.e. the
**router / gate** GEMM. Shape at decode conc4 is **M=4, N=288, K=4096, bf16**,
42 launches per forward (one per MoE layer).

This is *not* the expert GEMM. GLM-5.3-Flash-Quark-MXFP4 runs its experts
through `mfma_moe1_silu_mul_afp4_wfp4_bf16_*` / `mfma_moe2_afp4_wfp4_bf16_*`
(a4w4), which are untouched here and identical between the 0914 and 0923
images (0.648 → 0.647 ms and 0.340 → 0.348 ms).

## Why this kernel

It is the entire low-concurrency regression between
`rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260914` and `...-20260923`:

| | 0914 | 0923 | delta |
|---|---|---|---|
| `_gemm_a16_w16_kernel` (42 launches) | 0.353 ms | 1.182 ms | **+0.829 ms** |
| total decode bucket delta | | | +0.834 ms |
| decode wall delta | | | +0.83 ms |

Same kernel, same 42 launches, same tile — 8.4 → 28.1 µs per launch (3.35x).

## Root cause

gfx950 ships tuned `GEMM-A16W16` configs for N=128/256/384/640/1280/2880/5120
at various K, but **not for N=288, K=4096**. The lookup falls through to
`DEFAULT.json`'s `M_LEQ_8`:

```
BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, NUM_KSPLIT=1,
num_warps=8, num_stages=3, waves_per_eu=8, cache_modifier=".cg"
```

Two problems at M=4:

- `BLOCK_SIZE_M=16` wastes 12 of every 16 rows — the problem is 4 rows tall.
- `NUM_KSPLIT=1` with `BLOCK_SIZE_N=16` gives a grid of **18 workgroups on 256
  CUs**, so 93% of the GPU is idle.

The tuned neighbour `N=128-K=4096` already knows this and uses
`BLOCK_SIZE_M=4, NUM_KSPLIT=8`.

Note these are two separate facts. DEFAULT was *always* wrong for this shape
(even 0914's 8.4 µs is 1.7x off what the hardware can do); something in the
0923 image then made the same config 3.35x worse again. The image moved ROCm
7.2→10.0, triton 3.7→3.8, AITER `4ad99832`→`acf8fdf93` and python 3.10→3.12 in
one step, so which of the four caused the 3.35x is not established. Tuning the
config makes the question moot.

## Microbenchmark

`tools/glm53_bench_router_gemm.py`. M=4, N=288, K=4096, bf16, 42 rotating
weight buffers (94.5 MiB working set, matching the 42 distinct router weights
the model actually streams). Timing is **HIP graph replay**, not a python loop:
aiter's `gemm_a16w16` wrapper JSON-serializes the config on every call, which
costs ~27 µs of CPU and completely masks the GPU kernel. Every config is
checked against `F.linear` before it is timed.

1060 configs measured on the 0923 image:

| | median per launch |
|---|---|
| production default (0923) | **27.23 µs** |
| production default (0914, from trace) | 8.4 µs |
| `wvSpltK` | 10.62 µs |
| aiter skinny `wv_splitk_small_fp16_bf16` | 7.15 µs |
| `torch.mm` (hipblaslt) | 6.34 µs |
| **best triton config** | **4.89 µs** |

The winner:

```json
{
    "BLOCK_SIZE_M": 4,
    "BLOCK_SIZE_N": 16,
    "BLOCK_SIZE_K": 512,
    "GROUP_SIZE_M": 1,
    "NUM_KSPLIT": 4,
    "num_warps": 2,
    "num_stages": 3,
    "waves_per_eu": 0,
    "matrix_instr_nonkdim": 16,
    "cache_modifier": null
}
```

5.57x vs the 0923 default, 1.72x vs 0914, and it also beats both hipblaslt and
the aiter skinny GEMM, so no kernel swap is needed — a config file is enough.
`BLOCK_SIZE_M=4` matches the problem height exactly and `NUM_KSPLIT=4` lifts
the grid from 18 to 72 workgroups.

The top of the space is a plateau (4.89–5.05 µs across ~12 neighbouring
configs), not a fragile single point, so this should survive a compiler bump.

Expected decode saving: 42 × (27.23 − 4.89) = **0.94 ms per forward**.

## Installed config

`tools/glm53_install_router_config.py` writes
`GEMM-A16W16-N=288-K=4096.json` into aiter's gfx950 config dir
(`--revert` removes it). A specialized config file **replaces DEFAULT.json
wholesale** in `_get_gemm_config_cached`, so the file is a full copy of DEFAULT
with only `M_LEQ_1/4/8/16` overridden. Verified:

```
M=    1 is_tuned=True   BM=4   BN=16  BK=512 KSPLIT=4 warps=2
M=    4 is_tuned=True   BM=4   BN=16  BK=512 KSPLIT=4 warps=2
M=    8 is_tuned=True   BM=4   BN=16  BK=512 KSPLIT=4 warps=2
M=   16 is_tuned=True   BM=4   BN=16  BK=512 KSPLIT=4 warps=2
M=   64 is_tuned=True   BM=64  BN=32  BK=256 KSPLIT=1 warps=8   (DEFAULT, unchanged)
M= 8192 is_tuned=False  BM=256 BN=128 BK=64  KSPLIT=1 warps=8   (DEFAULT "any", unchanged)
N=512 K=4096 M=4        BM=16  BK=256                            (unrelated shape, unchanged)
```

## End-to-end result

i8k, TP4, 20 requests per cell, 0923 image, same node (`crsuse2-m2m-093`):

| cell | metric | baseline | tuned config | delta |
|---|---|---|---|---|
| conc4 | TTFT | 237.35 ms | 231.95 ms | −2.3% |
| conc4 | **TPOT** | 10.35 ms | **9.49 ms** | **−8.3%** |
| conc4 | **ITL** | 9.75 ms | **8.88 ms** | **−8.9%** |
| conc64 | TTFT | 351.68 ms | 342.63 ms | −2.6% |
| conc64 | TPOT | 26.71 ms | 26.67 ms | −0.1% |
| conc64 | ITL | 14.38 ms | 14.35 ms | −0.2% |

The whole low-concurrency regression is recovered. conc64 is flat, which is the
control the change predicts: only `M_LEQ_1/4/8/16` were overridden and conc64's
router GEMM runs at M=64, so it still takes DEFAULT's `M_LEQ_64`. That the
effect appears exactly where the config changed and nowhere else is what rules
out run-to-run drift.

The measured ITL saving of 0.87 ms matches the microbenchmark prediction of
42 × (28.16 − 4.89) µs = 0.98 ms.

GSM8K: 96.36% (baseline) → 97.04% (tuned), i.e. unchanged within noise.

### Trace verification

Re-profiled with the config installed
(`prof-Fixed-MXFP4-TP4-0923-routercfg-prof`), TP0 decode, 5 forwards:

| | kernel | median | sum / 210 launches |
|---|---|---|---|
| before | `BLOCK_SIZE_M_16 … BLOCK_SIZE_K_256, NUM_KSPLIT_1, EVEN_MN_0, cache_modifier_CG` | 28.16 µs | 5.910 ms |
| after | `BLOCK_SIZE_M_4 … BLOCK_SIZE_K_512, NUM_KSPLIT_4, EVEN_MN_1, cache_modifier_NONE` | **4.36 µs** | **0.917 ms** |

Exactly the installed config, so the lookup does reach this call site.
`EVEN_MN` also flipped to 1: M=4 is now divisible by `BLOCK_SIZE_M` and N=288
by `BLOCK_SIZE_N`, so the masking is gone.

`NUM_KSPLIT=4` adds a second kernel, `_gemm_a16w16_reduce_kernel`, also 210
launches at 4.32 µs (0.908 ms). Net per forward:

```
before: 1.182 ms (GEMM) + 0     (no reduce) = 1.182 ms
after:  0.183 ms (GEMM) + 0.182 ms (reduce) = 0.365 ms
                                      saving  0.817 ms
```

which is the measured ITL delta of 0.87 ms.

### Next: the split-K reduce is now half the cost

At 4.32 µs the reduce kernel is as expensive as the GEMM it serves, so a
`NUM_KSPLIT=1` config that avoids it entirely may win outright. The best
split-K-free configs in the sweep were:

| median | config |
|---|---|
| 5.74 µs | BM=1 BN=16 BK=512 warps=1 stages=2 |
| 5.79 µs | BM=4 BN=16 BK=512 warps=2 stages=2 |

5.8 µs in one kernel versus 8.68 µs in two would save a further ~0.12 ms per
forward (~1.3% of ITL). Untested end-to-end — the microbenchmark timed the
GEMM and its reduce inside one HIP graph and came out at 4.89 µs for the
split-K pair, below the 8.68 µs the trace shows, so the two-kernel cost was
under-counted there and this comparison should be redone against trace
numbers rather than the sweep CSV.

### Measurement note

The first attempt at this comparison was invalid and was discarded
(`bench-Fixed-MXFP4-TP4-0923-routercfg-INVALID-doublefire`). `tools/nrun.sh`
retries when the srun client hangs, but the `docker exec -d` from the first
attempt had already started, so the benchmark ran twice against one server —
two `bench_serving` clients meant the real concurrency was 8, not 4, and every
number came out worse (TTFT +47%, which no small-M GEMM config could cause).
`tools/run_glm53_bench_routercfg_guarded.sh` now takes an flock before
launching.

## Relationship to AITER PR5599

PR5599 (`6b35e2b9c [GLM5.3] Add routed MoE tuned configs for gfx950`) does not
overlap with this work, on three counts:

1. **Not in the image.** `git merge-base --is-ancestor 6b35e2b9c HEAD` is false;
   the 0923 image's AITER is at `acf8fdf93` (PR #5414).
2. **Different kernel.** It adds two files,
   `aiter/configs/model_configs/a8w8_blockscale_{tuned,untuned}_fmoe_glm5_3_flash_gfx950.csv`
   — expert (`fmoe`) GEMM configs, not the `gemm_a16w16` router GEMM, which
   lives in `aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/`.
3. **Different quantisation.** It tunes `a8w8_blockscale` (FP8). Our routed
   experts are MXFP4 (`afp4_wfp4`), so that path is never taken.

Separately: nothing in aiter reads `configs/model_configs/*.csv` at runtime
(grep finds zero call sites) — those are offline tuning inputs/outputs. The
runtime fmoe lookup uses `aiter/configs/tuned_fmoe.csv`.

## Open

- Phase-2 sweep of the remaining M buckets (M=64 for conc64, prefill M) —
  `tools/glm53_bench_router_gemm_allM.py`, not yet run to completion.
- `aiter.tuned_gemm` separately reports untuned shapes at runtime, e.g.
  `M:512, N:4096, K:1536 ... not found tuned config in bf16_tuned_gemm.csv,
  using torch solution:0`. Different lookup table, possibly another easy win.
