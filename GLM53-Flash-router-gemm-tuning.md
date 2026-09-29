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

## Root cause: `waves_per_eu=8` makes the kernel spill registers

gfx950 ships tuned `GEMM-A16W16` configs for N=128/256/384/640/1280/2880/5120
at various K, but **not for N=288, K=4096**, so the lookup falls through to
`DEFAULT.json`'s `M_LEQ_8`:

```
BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256, NUM_KSPLIT=1,
num_warps=8, num_stages=3, waves_per_eu=8, cache_modifier=".cg"
```

An earlier version of this document blamed the tile — `BLOCK_SIZE_M=16` for a
4-row problem, and a grid of 18 workgroups on 256 CUs. **That was wrong.** The
tile is fine. Holding it fixed at `BM=16/BN=16/BK=256, NUM_KSPLIT=1` and
walking only (`num_warps`, `num_stages`, `waves_per_eu`, `cache_modifier`) at
M=4 (`tools/glm53_ablate_default_knobs.py`, gfx950 / triton 3.8):

| warps | stages | waves_per_eu | cache_modifier | us | n_regs | n_spills | |
|---|---|---|---|---|---|---|---|
| 8 | 3 | **8** | `.cg` | **27.24** | 64 | **16** | DEFAULT `M_LEQ_8` |
| 8 | 2 | **8** | None | 25.38 | 64 | **21** | |
| 4 | 2 | **8** | `.cg` | 22.47 | 64 | **24** | |
| 4 | 3 | 6 | `.cg` | 7.40 | 80 | 0 | DEFAULT `M_LEQ_16` |
| 4 | 3 | 0 | None | **5.76** | 82 | 0 | fastest on this tile |

Two independent effects, both visible in the table:

1. **`waves_per_eu=8` caps the register budget at 64 and forces 16–24 spills**,
   costing 3–4x. Every `wpe=8` row spills; every `wpe<=6` row has zero spills
   and 80–88 registers. This is what makes the router GEMM 27 us.
2. **`cache_modifier=".cg"` costs a further ~25%** independently of spilling
   (5.76→7.27, 8.24→11.36, 12.65→15.55).

So the same DEFAULT tile reaches 5.76 us once those two knobs are corrected —
close to the 5.17 us found by sweeping tiles as well. The fix is about the
occupancy hint, not the tile shape.

This is not specific to N=288: any shape landing on DEFAULT's `M_LEQ_8` or
`M_LEQ_32` (both `waves_per_eu=8`) will spill. Measured on the router GEMM
shape, `M_LEQ_32` is the worst bucket of all at **35.20 us**, and `M_LEQ_16`
(`waves_per_eu=6`) is 3.8x faster than `M_LEQ_8` on an almost identical tile.

The 0914→0923 image step moved ROCm 7.2→10.0, triton 3.7→3.8, AITER
`4ad99832`→`acf8fdf93` and python 3.10→3.12 together, so which of the four
turned the spills from tolerable into 3.35x worse is not established. Removing
the spills makes the question moot.

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

## 0928 image and the conc24..64 question

Re-checked on `rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928`: AITER is still
`acf8fdf93` (#5414), `GEMM-A16W16-N=288-K=4096.json` is still absent, and
triton/torch/ROCm/python are unchanged from 0923. Only sglang moved
(`0318a8d0af`). So the gap is still open upstream.

### Per-bucket sweep

`tools/glm53_sweep_router_buckets.py`, N=288/K=4096, one M per bucket, GEMM and
split-K reduce timed in separate HIP graphs:

| M | bucket | DEFAULT | best (split-K allowed) | best single-kernel |
|---|---|---|---|---|
| 4 | `M_LEQ_4` | 27.11 us | 4.72 us (2 kernels) | 5.44 us |
| 8 | `M_LEQ_8` | 27.41 us | 4.72 us (2 kernels) | 5.19 us |
| 16 | `M_LEQ_16` | 7.12 us | 4.84 us (2 kernels) | 5.17 us |
| 32 | `M_LEQ_32` | 35.20 us | 5.04 us (2 kernels) | 5.27 us |
| 64 | `M_LEQ_64` | 13.68 us | 5.34 us (2 kernels) | 5.56 us |

### Single kernel beats split-K in the model

The decode graph has a soft per-kernel floor of ~3.7–3.9 us
(`trace_analysis/diagnostics/kernel_duration_floor.py`: 25 of 84 kernels have a
median under 4 us, but p5/p10 pile up at 3.76/3.84 us). `NUM_KSPLIT>1` pays it
twice. Trace numbers at conc4:

| config | kernels | in-model |
|---|---|---|
| DEFAULT | 1 | 28.04 us |
| split-K (`BM=4 BK=512 KSPLIT=4`) | 2 | 4.36 + 4.32 = 8.68 us |
| **single (`BM=1 BK=512 KSPLIT=1`)** | **1** | **6.04 us** |

The microbenchmark had the split-K pair at 4.72 us, below the 8.68 us the trace
shows, because back-to-back graph nodes pipeline the launch overhead in a
microbenchmark in a way they do not in the model. Config choice has to be
confirmed against traces.

### conc24..64 is a different code path

The installed config does nothing for conc64, and not because the lookup is
wrong. Instrumenting `get_gemm_config` in the running server
(`tools/glm53_trace_config_lookups.sh`) shows `GEMM-A16W16` is only ever asked
for N=288 at **M=1, 2, 4, 8, 16** — never at M>=24. Above that the router GEMM
leaves the triton path and goes through `aiter.tuned_gemm`, which finds no
N=288 row in `aiter/configs/bf16_tuned_gemm.csv` and falls back to
`torch solution:0` (hipblaslt). At conc64 it lands in the hipblaslt
`Cijk_..._MT16x16x1024` group, 87 launches per forward at 8.48 us, 3.66 ms total.

A caution about an earlier reading of the conc64 trace: the triton kernel
visible there (`BM=64 BN=32 BK=256`, 34 launches per forward) is **not** the
router. 34 is the KDA linear-attention layer count; the router fires 42 times,
matching the MoE layer count. It is N=6144/K=4096, a KDA projection.

So "improve conc4..64" needs two different changes:

| concurrency | M | router GEMM path | lever |
|---|---|---|---|
| 4..16 | <=16 | triton `gemm_a16w16` | the JSON config |
| 24..64 | >=24 | `tuned_gemm` -> hipblaslt | two rows in `glm53_bf16_tuned_gemm.csv` |

### Routing conc24..64 to triton

`aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv` already holds
N=288/K=4096 rows at M=2/4/16 with `libtype=triton` (6.76 / 6.83 / 5.39 us).
`get_config_file` merges every `model_configs/*bf16_tuned_gemm*.csv` into
`/tmp/aiter_configs/bf16_tuned_gemm.csv`, which is what a running server reads,
so those three rows *are* live — that is why conc4..16 reaches triton at all.
There is simply nothing above M=16.

`tuned_gemm` matches on exact M, then `get_padded_m(M,N,K,0)`, then
`get_padded_m(M,N,K,1)`, so two rows cover the whole range:

| decode bs | exact | padded gl=0 | padded gl=1 | matches |
|---|---|---|---|---|
| 24, 32 | 24 / 32 | 32 | 32 | **M=32 row** |
| 40, 48 | 40 / 48 | 48 | **64** | **M=64 row** |
| 56, 64 | 56 / 64 | 64 | 64 | **M=64 row** |

`tools/glm53_add_tunedgemm_rows.sh` appends them (`--revert` restores). Since
`triton_gemm()` calls `gemm_a16w16()` without an explicit config, the tile still
comes from the JSON above, so the two changes compose. Verified: M=1..64 all
report `libtype=triton`, M=72/128 deliberately still `torch`.

Trace result at conc64 — the hipblaslt group loses exactly the 42 router
launches:

| conc64, per forward | stock | + JSON | + JSON + CSV |
|---|---|---|---|
| `Cijk_..._MT16x16x1024` | 87 launches, 0.736 ms | 87, 0.732 ms | **45, 0.390 ms** |
| all `Cijk` | 1.468 ms (200) | 1.468 ms (200) | 1.132 ms (158) |
| triton `a16w16` K=4096 | 0.549 ms | 0.560 ms | 0.834 ms |
| **total** | **2.017 ms** | 2.028 ms | **1.966 ms** |

Router GEMM per launch: hipblaslt 8.52 us -> triton **6.48 us**. Net saving
0.05–0.086 ms per forward, i.e. **0.4–0.6% of a 14.3 ms conc64 ITL**.

That is an order of magnitude less than conc4 gets, and for a clear reason:
hipblaslt was already doing a reasonable job at M=64 (8.5 us), whereas at M<=16
the triton DEFAULT path was catastrophic (28 us). The mechanism is right and
the direction is right, but at conc64 the effect sits at the edge of end-to-end
noise.

### Same-image trace A/B (0928)

| | conc4 router | conc64 router |
|---|---|---|
| stock | 28.04 us (triton DEFAULT) | 8.52 us (hipblaslt) |
| + JSON | **6.04 us** | 8.48 us (unchanged) |
| + JSON + CSV | **5.96 us** | **6.48 us** |

### End-to-end

i8k, TP4, 20 requests per cell. The 0928 numbers use the single-kernel JSON
plus the two CSV rows; the 0923 rows are the earlier split-K JSON and its
baseline. **There is no same-image 0928 stock e2e baseline yet** — the cluster
token expired before it could run — so the cross-image comparison below is
supporting evidence only, and the same-image trace A/B above is the primary
result.

| image / config | conc4 TPOT | conc4 ITL | conc64 TPOT | conc64 ITL |
|---|---|---|---|---|
| 0923 stock | 10.35 ms | 9.75 ms | 26.71 ms | 14.38 ms |
| 0923 + split-K JSON | 9.49 ms | 8.88 ms | 26.67 ms | 14.35 ms |
| **0928 + single JSON + CSV** | **9.22 ms** | **8.61 ms** | 26.73 ms | 14.32 ms |

conc4 TTFT is unchanged throughout (237.35 / 231.95 / 237.19 ms), as it must be
— nothing here touches the prefill M. GSM8K 96.36% / 97.04% / 96.82%.

## Open

- Add N=288/K=4096 rows to the runtime `bf16_tuned_gemm.csv` for M>=24 and
  measure whether routing them to triton beats hipblaslt's 8.48 us.
- Propose the `waves_per_eu=8` spilling problem in DEFAULT's `M_LEQ_8` /
  `M_LEQ_32` upstream. It is shape-independent, so it needs wider measurement
  than this one GEMM before changing DEFAULT.
- `aiter.tuned_gemm` reports many other untuned shapes at runtime, e.g.
  `M:512, N:4096, K:1536 ... using torch solution:0`.
