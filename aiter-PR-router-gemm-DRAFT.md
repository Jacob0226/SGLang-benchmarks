# aiter PR draft — GEMM-A16W16 N=288 K=4096 (GLM-5.3-Flash MoE gate)

Branch `jacob/glm53-router-gemm-a16w16` in `~/PR/wt-aiter-glm53-router`, commit
`65db0fe2` off `upstream/main` (`330f1276`). One file, 134 insertions.

Title — a bot rewrites it to `[Triton/Gluon] [Config] [gfx950] ...` from the
paths touched, so the tags are deliberately absent:

    Add gfx950 GEMM-A16W16 config for N=288 K=4096 (GLM-5.3-Flash MoE gate)

**Not ready to open.** Two things are blocked on the cluster token:

1. The kernel table holds numbers from our own HIP-graph harness. It has to be
   re-measured with `tools/glm53_ab_router_aiterbench.sh`, which drives aiter's
   own `bench_gemm_a16w16.py`, so the command in Test Plan is one a reviewer can
   actually rerun.
2. The end-to-end `without` arm is from the 20260923 image. Only sglang differs
   between 0923 and 0928, but that is still not a one-variable A/B; it needs a
   same-image 0928 stock run (`tools/run_bench_0928_stock_guarded.sh`).

---

## Motivation

GLM-5.3-Flash has 288 routed experts over a hidden size of 4096, so each of its 42 MoE layers computes a bf16 gate projection of N=288, K=4096 — at decode with a small batch, a 4-row GEMM run 42 times per forward. gfx950 has no `GEMM-A16W16` file for this shape, so the lookup falls through to `DEFAULT.json`, whose small-M buckets pin `waves_per_eu=8`. That caps the kernel at 64 registers and forces 16 spills. The gate GEMM costs 28.0 us per call where 6.0 us is achievable, which is 0.9 ms of every decode forward.

One JSON, no code. A per-(N,K) file replaces the generic dict outright, so `M_LEQ_128` upward and `"any"` are copied from `DEFAULT.json` field for field.

## Test Plan

MI355X (gfx950), ROCm 10.0.0, Triton 3.8.0, torch 2.11.0, `rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928`. Kernel A/B by moving the JSON in and out of `aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16/`, which is the only variable that changes between arms.

    BENCH=op_tests/op_benchmarks/triton/bench_gemm_a16w16.py
    for M in 1 2 4 8 16 32 64 128 256; do
        python3 $BENCH --shape $M 288 4096 --metric time
    done

`--shape` is M N K. N=288 is the routed-expert count and K=4096 the hidden size, so this is the model's shape rather than a synthetic one. M is the decode batch; SGLang captures decode graphs at 1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, so M=128 and M=256 sit above the gated range and must not move.

End to end is SGLang serving GLM-5.3-Flash-Quark-MXFP4, TP4, 8192-in / 1024-out, 20 requests, concurrency 4.

## Test Result

**Kernel, us/call** — the command above, swept over M:

| M | without | with |
| --- | --- | --- |
| 4 | 27.1 | **5.4** |
| 8 | 27.4 | **5.2** |
| 16 | 7.1 | **5.2** |
| 32 | 35.2 | **5.3** |
| 64 | 13.7 | **5.6** |
| 128 | TBD | TBD |
| 256 | TBD | TBD |

`waves_per_eu=8` is what makes `M_LEQ_8` and `M_LEQ_32` the two worst buckets; `M_LEQ_16` is 3.8x faster than `M_LEQ_8` on an almost identical tile because it uses `waves_per_eu=6` instead. Every M from 1 to 64 improves and none regresses — M=128 and above resolve to buckets copied verbatim from `DEFAULT.json`, field for field.

**End to end, SGLang decode** — GLM-5.3-Flash-Quark-MXFP4, TP4, i8k/out1024, concurrency 4:

| metric | without | with |
| --- | --- | --- |
| Median ITL | 9.75 ms | **8.61 ms** |
| Median TPOT | 10.35 ms | **9.22 ms** |
| Median TTFT | 237.4 ms | 237.2 ms |

TTFT is unchanged, as it has to be: prefill runs at M in the thousands and resolves to `"any"`. GSM8K 96.4% -> 96.8%. In the decode trace the gate GEMM goes 28.04 -> 5.96 us per launch, 1.180 -> 0.251 ms per forward across its 42 calls.

---

## Kept out of the PR body

- **Why no split-K.** Split-K entries measured faster in isolation (4.72 us at
  M=4 against 5.44 us for the best single-kernel config) but lose in the model,
  8.68 vs 6.04 us. The decode graph has a ~3.8 us per-kernel floor and
  `NUM_KSPLIT>1` pays it twice for a reduction over 18 KB. Every shipped entry
  is `NUM_KSPLIT=1`. Worth a reply if a reviewer asks.
- **`M_LEQ_32` / `M_LEQ_64` are measured but unreachable for this model.** Above
  M=16 the gate GEMM is dispatched through `aiter.tuned_gemm`, which has no
  N=288 row above M=16 and falls back to `torch solution:0` (hipblaslt). Fixing
  that needs two rows in `model_configs/glm53_bf16_tuned_gemm.csv` (M=32 and
  M=64 cover decode batch 24..64 through `get_padded_m`) and is a separate PR.
  It is worth only 0.4-0.6% of a 14.3 ms conc64 ITL, because hipblaslt is
  already reasonable at M=64: 8.52 -> 6.48 us per launch. The entries are still
  correct for any caller reaching the triton path directly, and `M_LEQ_32` is
  the largest ratio in the table.
- **Not raising `waves_per_eu=8` in `DEFAULT.json` itself.** Every bucket with
  it spills, so this is not specific to N=288, but changing DEFAULT needs far
  wider measurement than one GEMM.
- **CI labels:** none. A per-(N,K) config keyed on 288x4096 cannot reach
  `ci:sglang`'s DeepSeek-R1/Qwen3.5 or `ci:atom`'s DeepSeek-R1/GPT-OSS, so
  asking for them burns machine time on runs that cannot observe the change.
  Open as draft until CI is green.
