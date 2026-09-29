# Handover — GLM-5.3-Flash router GEMM tuning (MI355X / gfx950)

Everything needed to pick this up on a fresh node. The work is essentially done
and measured; what remains is one reproducibility measurement and opening an
aiter PR.

---

## 1. What this is, in one paragraph

GLM-5.3-Flash has 288 routed experts over hidden size 4096, so every one of its
42 MoE layers runs a bf16 gate projection of **N=288, K=4096** — at decode with
a small batch, a 4-row GEMM, 42 times per forward. aiter has no tuned
`GEMM-A16W16` entry for that shape on gfx950, so it falls back to
`DEFAULT.json`, whose small-M buckets set `waves_per_eu=8`. That caps the kernel
at 64 registers and forces 16–24 spills. On the 0928 image the gate GEMM costs
**28 us per call instead of ~6**, which is the whole low-concurrency decode
regression against the 0914 image. A tuned config file fixes it: **i8k conc4 ITL
9.51 → 8.63 ms (−9.3%), output throughput 360 → 407 tok/s (+13%)**.

Two sessions worked on this independently and reached the same end result from
different configs; both sets of numbers are in section 4.

---

## 2. Environment

### Image and model

| | |
|---|---|
| image | `rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928` |
| ROCm / Triton / torch / python | 10.0.0 / 3.8.0 / 2.11.0 / 3.12.3 |
| AITER in image | `acf8fdf93` (PR #5414) — **PR5599 is *not* in it** |
| sglang in image | `0318a8d0af` (0.5.20.dev20260928) |
| model (host) | `/mnt/m2m_nobackup/models/amd/GLM-5.3-Flash-Quark-MXFP4` |
| model (in container) | `/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4` |

The image's own sglang already contains all ten GLM-5.3-Flash Day-0 PRs, so
**no PYTHONPATH tree and no PR stack to apply** — unlike the 0914 workflow.

### Getting a node

```bash
~/SGLang-benchmarks/tools/check_cluster.sh          # one-shot: auth / partition / job / node
~/SGLang-benchmarks/tools/check_cluster.sh --watch  # poll every 60s until CLUSTER_READY
```

It reports the four things that must line up, in the order they fail. The node
check distinguishes three cases, because they need different responses: the srun
client hanging (normal, retry), `spurd` missing the spur auth plugin (admin
issue, retrying never helps — this is what blocked 2026-09-29), and a reachable
node that simply has no container.

All node commands go through the retry wrapper, because the srun client hangs
roughly half the time:

```bash
NRUN_JOBID=<jobid> ~/SGLang-benchmarks/tools/nrun.sh '<command>'
```

### Starting the container

```bash
NRUN_JOBID=<jobid> ~/SGLang-benchmarks/tools/nrun.sh \
    'bash /home/jacchang/SGLang-benchmarks/tools/start_glm53_0928.sh'
```

Starts `jacchang_GLM53-Flash-1` **detached** (`-d` plus `tail -f /dev/null`).
Do not use `-it --rm`: the container dies with the srun session.

`$HOME` is bind-mounted at the same path inside the container, so scripts under
`~/SGLang-benchmarks/tools/` run directly and write results back to the shared
home.

### Cold cache warning

A new node means a cold JIT cache. Last time the first server start took about
**20 minutes**, of which 7 were CUDA-graph capture over 52 batch sizes. Budget
for it; nothing is wrong. Caches live in
`~/SGLang-benchmarks/tmp/cache-glm53-0928-rocm10/`.

---

## 3. Where everything is

| What | Path |
|---|---|
| aiter branch, ready to push | `~/PR/wt-aiter-glm53-router`, commit `65db0fe2`, branch `jacob/glm53-router-gemm-a16w16` off `upstream/main` |
| PR body draft | `SGLang-benchmarks/aiter-PR-router-gemm-DRAFT.md` |
| Full technical record | `SGLang-benchmarks/GLM53-Flash-router-gemm-tuning.md` |
| Config candidate A (single-kernel) | `analysis_GLM5.3/router_gemm_0928/GEMM-A16W16-N=288-K=4096.best_single.json` |
| Config candidate B (split-K, other session) | `tools/aiter-GEMM-A16W16-N=288-K=4096.json` |
| Sweep data | `analysis_GLM5.3/router_gemm_0928/bucket_sweep.csv` (~3000 configs) |
| Spill ablation | `analysis_GLM5.3/router_gemm_0928/default_knob_ablation.json` |
| Traces | `results/.../20260928/prof-Fixed-MXFP4-TP4-0928-{stock,single,csv}-prof/` |
| E2E results | `results/.../20260928/bench-Fixed-MXFP4-TP4-0928-{Routing-baseline,Routing-opt,full}/` |

### Key scripts

| Script | Job |
|---|---|
| `tools/start_glm53_0928.sh` | start the container detached |
| `tools/glm53_install_candidate.sh best_single \| --revert` | install/remove the JSON, prints the resolved config per M |
| `tools/glm53_add_tunedgemm_rows.sh [--revert]` | add the two CSV rows that route conc24..64 to triton |
| `tools/glm53_sweep_router_buckets.py` | per-bucket config sweep, emits both candidates |
| `tools/glm53_ablate_default_knobs.py` | the spill ablation (fixed tile, walks wpe / stages / cache_modifier) |
| `tools/glm53_ab_router_aiterbench.sh` | **the one still to run** — A/B via aiter's own bench |
| `tools/run_glm53_prof_0928_stock.sh` | profile conc4+conc64 (edit tag per arm) |
| `tools/run_bench_0928_stock_guarded.sh` | e2e i8k conc4+64, flock-guarded |
| `trace_analysis/diagnostics/identify_a16w16.py <trace.gz> <substr>` | kernel name, median, per-forward cost |
| `trace_analysis/diagnostics/kernels_by_launch_count.py <trace.gz> [N]` | kernels firing exactly N times per forward |

Launch-count fingerprints for this model: **42** = MoE layers (router, expert
GEMMs, sorting), **34** = KDA linear-attention layers, **11** = MLA/DSA sparse
layers, **3** = dense MLP, **45** = all layers. This is how to tell the router
GEMM from the KDA projection in a trace — they are both `_gemm_a16_w16_kernel`.

---

## 4. Results already in hand

### Root cause (fixed tile `BM=16 BN=16 BK=256 KSPLIT=1`, M=4, gfx950/triton 3.8)

| warps | stages | waves_per_eu | cache_modifier | us | n_regs | n_spills | |
|---|---|---|---|---|---|---|---|
| 8 | 3 | **8** | `.cg` | **27.24** | 64 | **16** | DEFAULT `M_LEQ_8` |
| 4 | 2 | **8** | `.cg` | 22.47 | 64 | **24** | |
| 4 | 3 | 6 | `.cg` | 7.40 | 80 | 0 | DEFAULT `M_LEQ_16` |
| 4 | 3 | 0 | None | **5.76** | 82 | 0 | fastest on this tile |

`waves_per_eu` is the variable, not `num_warps`: at `wpe=8` both 4 and 8 warps
are terrible, at `wpe=6` both are fine. It also explains the bucket ordering —
`M_LEQ_32` (`wpe=8`) is the worst at 35.2 us, `M_LEQ_64` (`wpe=4`) only mildly
bad at 13.7 us. `cache_modifier=".cg"` costs a further ~25% independently.

### Per-bucket sweep (N=288, K=4096, our HIP-graph harness, 42 rotating weights)

| M | bucket | DEFAULT | best single-kernel |
|---|---|---|---|
| 4 | `M_LEQ_4` | 27.11 | **5.44** |
| 8 | `M_LEQ_8` | 27.41 | **5.19** |
| 16 | `M_LEQ_16` | 7.12 | **5.17** |
| 32 | `M_LEQ_32` | 35.20 | **5.27** |
| 64 | `M_LEQ_64` | 13.68 | **5.56** |

### Trace, same image (0928), conc4 / conc64

| | conc4 router | conc64 router |
|---|---|---|
| stock | 28.04 us (triton DEFAULT) | 8.52 us (hipblaslt) |
| + JSON | **6.04 us** | 8.48 us (unchanged) |
| + JSON + CSV rows | **5.96 us** | **6.48 us** |

With the CSV rows the hipblaslt `Cijk_..._MT16x16x1024` group drops from 87 to
45 launches per forward — exactly the 42 router calls.

### End to end, all on the 0928 image

| cell | baseline | config B (split-K) | config A (single) + CSV |
|---|---|---|---|
| i8k conc4 ITL | 9.51 ms | **8.63** | **8.61** |
| i8k conc4 TPOT | 10.13 ms | **9.23** | **9.22** |
| i8k conc4 out tok/s | 360.3 | **407.4** | **407.4** |
| i8k conc64 ITL | 14.38 ms | 14.37 | **14.32** |
| i8k conc64 TPOT | 26.98 ms | 26.75 | 26.73 |
| i70k conc4 ITL | 9.62 ms | **8.74** | not run |
| i70k conc4 TTFT | 2098.7 ms | 2064.1 | not run |

Two things this settles:

- **The two configs are indistinguishable at conc4** (8.63 vs 8.61; 407.35 vs
  407.42 tok/s). A predicted 1.3% advantage for the single-kernel config from
  the per-kernel floor argument did **not** appear.
- **conc64 only moves with the CSV rows** (14.38 → 14.32, −0.4%), matching the
  0.4–0.6% the trace predicted. Config alone leaves conc64 flat because above
  M=16 the router GEMM is dispatched through `aiter.tuned_gemm` to hipblaslt.
  0.4% is not worth shipping — see section 5. The scope of this work is
  therefore **conc <= 16**, where it is worth 9%.

GSM8K across runs: 96.36 / 96.44 / 96.82 / 96.89 / 97.04 — no trend, spread
consistent with sampling noise on 1319 questions.

---

## 5. What still needs doing, in order

### Task 1 — decide which config ships

They are equal end-to-end, so the tiebreak is on other grounds.

| | config A (single-kernel, `analysis_GLM5.3/router_gemm_0928/...best_single.json`) | config B (split-K, `tools/aiter-GEMM-A16W16-N=288-K=4096.json`) |
|---|---|---|
| approach | removes `waves_per_eu=8` and `.cg` | keeps them, adds `NUM_KSPLIT=8` |
| kernels per call | 1 | 2 (GEMM + split-K reduce) |
| diff shape | per-bucket entries | one uniform entry for `M_LEQ_1..64` |

**Recommendation: config A.** It removes the thing that is actually wrong rather
than masking it — B works because `NUM_KSPLIT=8` shortens the K loop enough that
`waves_per_eu=8` stops spilling, which is a coincidence that a future compiler
change can undo. A also avoids a second kernel launch.

One loose end in B either way: its own notes say `M_LEQ_16` should keep DEFAULT
(the ks8 entry measured 5.67 vs 4.80 us there), but the file still overrides it.

### Task 2 — the reviewer-reproducible kernel table

Our sweep harness is not something an aiter reviewer can rerun. Redo the table
with aiter's own benchmark:

```bash
NRUN_JOBID=<jobid> ~/SGLang-benchmarks/tools/nrun.sh \
    'docker exec -d jacchang_GLM53-Flash-1 bash /home/jacchang/SGLang-benchmarks/tools/glm53_ab_router_aiterbench.sh'
# watch: tmp/logs/ab_router_aiterbench.log ; ends with AB_DONE and a markdown table
```

It moves the JSON in and out of the config directory and sweeps
**M = 1 2 4 8 16 32 64 128 256**. M=128/256 are deliberate: those buckets are
copied verbatim from `DEFAULT.json` and the table has to show they do not move.

The command a reviewer would run is:

```bash
python3 op_tests/op_benchmarks/triton/bench_gemm_a16w16.py --shape <M> 288 4096 --metric time
```

### Task 3 — open the aiter PR

```bash
cd ~/PR/wt-aiter-glm53-router
git push origin jacob/glm53-router-gemm-a16w16
gh pr create --repo ROCm/aiter --draft --base main \
    --head Jacob0226:jacob/glm53-router-gemm-a16w16 --title "..." --body "..."
```

Body is drafted in `aiter-PR-router-gemm-DRAFT.md` (Motivation / Test Plan /
Test Result, each ≤100 words of prose, detail in tables). Fill in the Task 2
numbers and the section 4 end-to-end table first.

- Do **not** hand-write component tags in the title; a bot prefixes
  `[Triton/Gluon] [Config] [gfx950]` from the paths touched.
- **No CI labels.** A per-(N,K) config keyed on 288x4096 cannot reach
  `ci:sglang`'s DeepSeek-R1/Qwen3.5 or `ci:atom`'s GPT-OSS.
- Open as draft until CI is green.

### conc24..64 — investigated, deliberately dropped

**Not doing this.** Recorded here so nobody re-derives it.

Above M=16 the router GEMM leaves the triton path: `aiter.tuned_gemm` finds no
N=288 row above M=16 and falls back to `torch solution:0` (hipblaslt). Two rows
in `aiter/configs/model_configs/glm53_bf16_tuned_gemm.csv` route it back —
`tuned_gemm` matches exact M, then `get_padded_m(M,N,K,0)`, then
`get_padded_m(M,N,K,1)`, so M=32 and M=64 cover decode batch 24..64:

| decode bs | exact | gl=0 | gl=1 | matches |
|---|---|---|---|---|
| 24, 32 | – | 32 | 32 | M=32 row |
| 40, 48 | – | 48 | **64** | M=64 row |
| 56, 64 | – | 64 | 64 | M=64 row |

It works — the hipblaslt group loses exactly its 42 router launches and the
kernel goes 8.52 → 6.48 us — but it buys **0.4% of conc64 ITL** (14.38 → 14.32
ms), because hipblaslt is already reasonable at M=64. That is at or below
end-to-end noise, and not worth a second PR and review cycle.
`tools/glm53_add_tunedgemm_rows.sh` still has the change if it is ever wanted.

Consequence for the PR: the JSON's `M_LEQ_32` and `M_LEQ_64` entries are then
unreachable for this model. **Keep them anyway.** The aiter op benchmark in
Task 2 calls `gemm_a16w16` directly and bypasses `tuned_gemm`, so those two M
values are measured legitimately (35.2 → 5.3 and 13.7 → 5.6 us), and the
entries are correct for any caller that does reach the triton path there. Do
not claim an end-to-end effect for them.

---

## 6. Traps that already cost time

- **`nrun.sh` retries fire the command twice.** When the srun client hangs, the
  `docker exec -d` it already started keeps running, so a retry launches a
  second one. Two `bench_serving` clients against one server doubles the real
  concurrency and every number comes out worse — this produced a fake 47% TTFT
  regression once. Wrap any launch in `flock`; see
  `tools/run_bench_0928_full_guarded.sh`. The other session hit the same class
  of bug on 0914 and saw a fake −51% "win" from a leftover process.
- **`pkill -f <pattern>` matches its own `bash -c` command line.** Use a
  bracket: `pkill -f "glm53_bench_router[_]gemm"`.
- **Σ of all kernel times in a trace is not decode wall time.** It counts
  streams separately and includes a spin-waiting all-reduce that varies by
  milliseconds run to run. One capture showed Σ 16.8 ms vs 10.3 ms with the
  router GEMM essentially unchanged. Compare named kernels, and use e2e ITL for
  the bottom line.
- **Microbenchmarks under-report the cost of adding a kernel.** Back-to-back
  graph nodes pipeline the launch overhead in a way they do not in the model:
  the split-K pair measured 4.72 us in isolation but 8.68 us in the trace. The
  decode graph has a ~3.8 us per-kernel floor. Confirm config choices against
  traces.
- **`analysis_GLM5.3/` is root-owned** (created from inside the container). Do
  file writes there from inside the container, or write elsewhere.
- **Two `_gemm_a16_w16_kernel` entries in a decode trace are different GEMMs.**
  Use the launch count: 42/forward is the router (K=4096), 34/forward is the KDA
  projection (N=6144), 11/forward is the DSA one (K=1536). Misreading this once
  led to the wrong conclusion that conc64 used the triton router.

---

## 7. Corrections to earlier write-ups, so they are not repeated

- The root cause is **not** the tile (`BLOCK_SIZE_M=16` for a 4-row problem, 18
  workgroups on 256 CUs). The same tile reaches 5.76 us once `waves_per_eu` and
  `cache_modifier` are fixed. It is register spilling.
- It is **not** `num_warps=8` either. See the ablation table in section 4.
- AITER **PR5599 is unrelated** on three counts: not in this image (AITER is at
  #5414), it tunes the *expert* GEMM (`a8w8_blockscale` fmoe) not the router
  `gemm_a16w16`, and it targets FP8 while our routed experts are MXFP4
  (`mfma_moe1_silu_mul_afp4_wfp4`). Nothing in aiter reads
  `configs/model_configs/*.csv` at runtime; the runtime table is the merged
  `/tmp/aiter_configs/` copy.
- `wvSpltK` does support N=288 — an earlier report of it failing was an argument
  order mistake. Measured correctly it is 5.85–7.15 us at M=4, not better than
  the tuned Triton config, so it is not worth a PR.
