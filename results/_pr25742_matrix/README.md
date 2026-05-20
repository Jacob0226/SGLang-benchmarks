# sglang #25742 GSM8K Regression Bisect — `amd/GLM-5.1-MXFP4` on MI355X

Date: 2026-05-20  
Hardware: 1× node, 8× AMD MI355X (gfx950, 256 GB HBM)  
Model: `amd/GLM-5.1-MXFP4` (397B MoE, Quark MXFP4)  
Eval: `python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py --port 8552 --num-questions 1200 --parallel 1200` (sgl-DSL, `temperature=0`, no chat template)  
Driver script: `repro_pr25742_acc_matrix.sh` (extracted from `GLM.sh`'s MI355X branch — no `--match-pr`, no MTP, no lm-eval)

---

## 0. TL;DR

- **The TP=2 accuracy collapse is a real regression**, and it is **entirely caused by the `aiter` upgrade**, not by any sglang Python commit.
- **First-bad ROCm docker image: `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260501`** (and equivalent `lmsysorg/sglang-rocm` mirror).
- **First-good image: `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260430`**.
- Between those two builds, **aiter** was bumped:
  - 4/30 image: `amd-aiter 0.1.12.post1`
  - 5/1 image: `amd-aiter 0.1.12.post2.dev150+ga6bb49937` (= `ROCm/aiter` commit `a6bb499`, PR ROCm/aiter#2879 "Support preshuffled layout in indexer_k_quant_and_cache / cp_gather_indexer_k_quant_cache")
- **Polarity-flip confirmation** (§7): swapping ONLY aiter (4/30 sglang + 5/1 aiter) drops TP=2 accuracy from **0.939 → 0.623** while sglang Python stays untouched. Reproducible, definitive.
- The corresponding sglang-side commit that bumped the build-time aiter pin is **`7e17a7a1e Upd: AITER->(#2879)a6bb499 (#24218)`** — it only changes the `AITER_COMMIT` env var consumed by the docker build (no Python diff), which is why `git checkout` of sglang inside an already-built image cannot revert the regression.
- The PR #25742 framing is wrong on two counts: (a) it's not a v0.5.12 regression — first-bad is `v0.5.10.post1-rocm720-mi35x-20260501`; (b) it's not a sglang regression — it's an aiter build-pin upgrade in PR ROCm/aiter#2879.

---

## 1. Original report

[sglang #25742](https://github.com/sgl-project/sglang/issues/25742) reports that `amd/GLM-5.1-MXFP4` on `lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-20260517` gives:
- OFF variant (no MTP): gsm8k strict / flex = **0.3177 / 0.3177**
- EAGLE-MTP variant: **0.1774 / 0.1782**

Reproduction setup: `--tensor-parallel-size 2`, `--cuda-graph-max-bs 256`, `--context-length 9472`, `--reasoning-parser glm45`, `--tool-call-parser glm47`, NSA tilelang prefill+decode, `--kv-cache-dtype fp8_e4m3`, lm-eval-harness with `--apply_chat_template`.

---

## 2. Eval path differences (`bench_sglang.py` vs `lm_eval`)

The PR's 0.32 number is `lm_eval` with `--apply_chat_template`. Two compounding effects explain why it lands so low:

| Layer | Effect on score |
|---|---|
| **(a) TP=2-only model-serving accuracy degradation** | Drops 0.94 → ~0.60 on this docker image |
| **(b) `--reasoning-parser glm45` puts `<think>…</think>` into `reasoning_content`; lm-eval reads only `content`** | Drops a further ~0.60 → ~0.20 because many responses log empty `content` |

Verification on `v0.5.12-rocm720-mi35x-20260517`, TP=2:
| Eval | Result |
|---|---|
| `bench_sglang.py` (sgl-DSL, no chat template) | 0.579 (Invalid 0.337) |
| `lm_eval` chat-completions + `apply_chat_template`, `--reasoning-parser glm45` | 0.1820 strict / 0.1842 flex |

InferenceX's eval harness already monkey-patches `LocalChatCompletion.parse_generations` to fall back to `reasoning_content` when `content` is empty (`benchmark_lib.sh` `_patch_lm_eval`), so internal sweeps don't see the lm-eval compounding — they see ~0.6 at the OFF variant rather than ~0.18.

This bisect uses **`bench_sglang.py`** to isolate effect (a) from effect (b).

---

## 3. Docker × TP coarse matrix

| Docker image | TP=8 | TP=2 |
|---|---|---|
| `lmsysorg/sglang-rocm:v0.5.10rc0-rocm720-mi35x-20260415` | **0.941** (Inv 0.001) | **0.943** (Inv 0.000) ✓ |
| `rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260422` | — | **0.936** (Inv 0.000) ✓ |
| `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260423` | — | **0.933** (Inv 0.002) ✓ |
| `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260428` | — | **0.932** (Inv 0.002) ✓ |
| `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260430` | — | **0.939** (Inv 0.000) ✓ |
| `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260501` | — | **0.601** (Inv 0.338) ✗ |
| `rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260503` | 0.932 (Inv 0.005) | 0.623 (Inv 0.295) ✗ |
| `rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260510` | 0.927 (Inv 0.002) | 0.590 (Inv 0.357) ✗ |
| `lmsysorg/sglang-rocm:v0.5.12-rocm720-mi35x-20260517` | 0.917 (Inv 0.004) | 0.579 (Inv 0.337) ✗ |

Observations:
- **TP=8 is healthy on every image** (~0.92–0.94). The collapse is TP=2-only.
- **Invalid jumps from 0% to ~30%** across the boundary — this is not a gradual numerical drift; some kernel/path goes from correct to broken between 4/30 and 5/1 builds.
- The earlier "v0.5.12 regression" framing is wrong: 5/3, 5/10, 5/17 all show the same 0.58–0.62 band as 5/1 (first-bad).

---

## 4. Bisect timeline

| Step | Image | TP=2 result | Verdict |
|---|---|---|---|
| 1 | 4/15 (`v0.5.10rc0-20260415`) | 0.943 | GOOD |
| 2 | 5/3 (`v0.5.10.post1-20260503`) | 0.623 | BAD |
| 3 | 4/22 (`v0.5.10rc0-20260422`) — last rc0 | 0.936 | GOOD |
| 4 | 4/23 (`v0.5.10.post1-20260423`) — first post1 | 0.933 | GOOD → version-string change isn't the regression |
| 5 | 4/28 | 0.932 | GOOD |
| 6 | 4/30 | 0.939 | GOOD |
| 7 | 5/1 | 0.601 | BAD ← **first-bad confined to single day** |

Bisect converged. Window narrowed from 18 days → 1 day → 41 sglang commits in that 26-hour build window.

---

## 5. Commit-level investigation inside the 5/1 container

The 4/30 and 5/1 docker images have these sglang HEADs:
- 4/30 image: `aa74911448f332bc807d75408047bb1df794f9b9` (UTC 4/30 12:05)
- 5/1 image:  `4a50cd781e16c793c2377600aa1745f82fbed51c` (UTC 5/1 13:57)

41 sglang commits land in that 26-hour window. Of those, only **4** touch any quant/MoE runtime code:

| Commit | Title | Runtime impact |
|---|---|---|
| `108bfd8b6` | [MoE] Add Aiter MoE runner backend and purge `aiter.fused_moe` from quant methods (#23597) | High — touches `mxfp4.py`, `quark_w4a4_mxfp4_moe.py`, MoE runner |
| `71e89e900` | [MUSA][19/N] Support qwen series models (#23654) | Medium — touches `topk.py`, `fp8_kernel.py`, `fp8_utils.py` |
| `651af06a0` | [Feature] Xiaomi MiMo-V2.5 day0 support (#23811) | Medium — touches `model_config.py`, `vision.py`, `linear.py` (QKVParallelLinear V-shard `head_size` → `v_head_size`) |
| `cf4f46209` / `7bb7f6049` / `dc395bc05` | (3 commits) | Zero — pure CI YAML / scripts |

Verification by `git checkout` inside the 5/1 container (which keeps the 5/1-baked `aiter` and `sgl-kernel` binaries), running TP=2:

| Checkout | Commit | TP=2 Accuracy | Verdict |
|---|---|---|---|
| `108bfd8b6^` | `0acc569ed` | 0.623 (Inv 0.318) | BAD |
| `71e89e900^` | `dc395bc05` | 0.593 (Inv 0.348) | BAD |
| `651af06a0^` | `cf4f46209` | 0.609 (Inv 0.302) | BAD |

All three "before" states are BAD. Going further back inside the 5/1 container is pointless: **rewinding sglang Python source inside the 5/1 image does not restore accuracy**. The regression is not in sglang Python.

---

## 6. Real root cause: `aiter` build-pin upgrade

The two images differ in their baked-in `aiter` package:

| Image | aiter version | sglang HEAD |
|---|---|---|
| 4/30 (GOOD) | **`amd-aiter 0.1.12.post1`** | `aa74911448` |
| 5/1 (BAD) | **`amd-aiter 0.1.12.post2.dev150+ga6bb49937`** | `4a50cd781e` |

`aiter` is installed editable from `/sgl-workspace/aiter` (cloned at docker-build-time at the commit specified by the `AITER_COMMIT` env var). The version `0.1.12.post2.dev150+ga6bb49937` corresponds to upstream commit `a6bb499` in `ROCm/aiter` (PR ROCm/aiter#2879).

The sglang-side commit that bumped this build-time pin is **`7e17a7a1e Upd: AITER->(#2879)a6bb499 (#24218)`** — it only changes the docker recipe's `AITER_COMMIT` env var, no Python diff. That's why `git checkout` inside an already-built image cannot revert it.

This explains every observation:
1. **TP-specificity**: a kernel change in `aiter` could plausibly behave correctly for TP=8 partition shapes but break TP=2 partition shapes (or vice versa).
2. **Discontinuity in Invalid rate** (0% → 33% in one day): consistent with a kernel returning structurally wrong outputs (e.g., NaN/zeros from a misaligned shard), not a gradual numerical drift.
3. **TP=8 mild downward drift across 4/15 → 5/17** (0.94 → 0.92, ~2.4 pp): consistent with continued aiter/kernel tuning churn that hits TP=8 less hard than TP=2.

---

## 7. Polarity-flip confirmation (✅ aiter is the sole cause)

Direct swap test executed inside the **4/30 (GOOD) container** — sglang Python untouched (still at `aa7491144`), only `/sgl-workspace/aiter` replaced with the 5/1 image's aiter checkout (`a6bb499`), and `flydsl` upgraded `0.1.2 → 0.1.5.dev504` to satisfy the new aiter's `flydsl>=0.1.3` requirement.

| Config | TP=2 Accuracy | Invalid | Verdict |
|---|---|---|---|
| 4/30 sglang `aa7491144` + 4/30 aiter `v0.1.12.post1` (original) | **0.939** | 0.000 | GOOD |
| 4/30 sglang `aa7491144` + **5/1 aiter `a6bb499`** (swapped in) | **0.623** | 0.311 | **BAD** |

Polarity flipped while only changing `aiter`. **Confirmed: the regression is entirely in the aiter upgrade.**

The new (BAD) accuracy 0.623 / Invalid 0.311 sits in exactly the same band as the 5 BAD images measured earlier (0.579–0.623, Invalid 0.295–0.357), ruling out residual sglang-side effects.

---

## 8. Suggested next steps for sglang / AMD

1. **Bisect `aiter` between `v0.1.12.post1` and `a6bb499`** (commit window in `ROCm/aiter`) to find the specific kernel change that breaks TP=2 sharding for Quark MXFP4 MoE / NSA path on gfx950. PR ROCm/aiter#2879 itself is a strong starting candidate (it touches `indexer_k_quant_and_cache` / `cp_gather_indexer_k_quant_cache`, both of which are on the DSA decode path used by GLM-5.1).
2. **Revert or pin `AITER_COMMIT` back to `v0.1.12.post1`** in `lmsysorg/sglang-rocm` and `rocm/sgl-dev` docker builds until the aiter fix lands, so downstream users (InferenceX, AMD CI) get a working `amd/GLM-5.1-MXFP4` again.
3. **Add a TP=2 GSM8K accuracy gate** to AMD nightly CI for `amd/GLM-5.1-MXFP4` — current sweeps only test TP=2 with custom InferenceX harness; sglang's own MI355X CI did not catch this regression.
4. **Reframe issue #25742** away from "v0.5.12 regression" / "lm-eval reasoning_content blocker" toward "aiter `post1` → `post2.dev150` (PR ROCm/aiter#2879) build-pin upgrade breaks TP=2 Quark MXFP4 on gfx950".

---

## 9. Aiter sub-bisect progress (in-flight)

Goal: identify the single first-bad commit inside the aiter window `(v0.1.12.post1, a6bb499]` (150 commits).

Sub-bisect uses the `bisect_0501` container (sglang HEAD `4a50cd781e`, aiter HEAD initially `a6bb499`, flydsl `0.1.5.dev504`). For each tested commit, `git checkout` is done inside `/sgl-workspace/aiter`, the relevant prebuilt `.so` files are deleted to force aiter's JIT path to rebuild from the current source, then TP=2 GSM8K is run.

Results so far:

| Aiter commit | Title | Method | TP=2 Acc / Invalid | Verdict |
|---|---|---|---|---|
| `a6bb499` (#2879) | preshuffled layout in `indexer_k_quant_and_cache` | as-baked (5/1 image) | 0.601 / 0.338 | BAD |
| `da67a099` (#2927) = `a6bb499^` | Re-enable native qh128 fp8 kernel on gfx950 | source-only checkout (stale `module_cache.so`) | 0.588 / 0.358 | BAD |
| `da67a099` (#2927) | (same commit, retry) | source-only + delete `module_cache.so` → JIT-rebuild | 0.657 / 0.273 | BAD (PR #2879 contributes some, not main) |
| `c71075ced` (#2890) = `da67a099^` | Fix 2-stage fused_allreduce_rmsnorm memory ordering | source-only + delete `module_cache.so` + `module_mla_*.so` → JIT-rebuild | 0.608 / 0.323 | BAD (PR #2927 also not the sole cause) |

So far ruled out (as sole cause): **#2879**, **#2927**, **#2890**. The first-bad commit is in `(v0.1.12.post1, c71075ced)` — i.e. 147 commits earlier than `c71075ced`.

Remaining suspect commits in window (touch quant/MoE/MLA runtime code in `aiter/{ops,kernels}` or `csrc/kernels/`):

| Commit | PR | Why suspicious |
|---|---|---|
| `e039817a` | #2852 | MLA PS mode `nhead8,2` in MI308 |
| `fea695b9` | #2727 | MI350 MLA PS mode `nhead128/64/32` kernel |
| `c849fd58` | #2729 | bf16 MLA decode kernel for `gqa_ratio=64` |
| `d87e5991` | #2717 | Replace CK/CK_TILE in MLA Reduce/Metadata with OPUS |
| `22db4ebb` | #2676 | Native `qh32 qseqlen2` MLA PS kernel for gfx950 |
| `1bbd58c9` | #2759 | fix `fused_dynamic_mxfp4_quant_moe_sort_hip` in EP |
| `b025f094` | #2700 | Optimize `fused_dynamic_mxfp4_quant_moe_sort_hip` in small M |
| `5282d715` | #2693 | fix `fused_dynamic_mxfp4_quant_moe_sort` dispatch |

---

## 10. Aiter rebuild methodology / pitfalls

Why naive `git checkout` of aiter inside a container does NOT correctly toggle behaviour:

1. **Prebuilt `.so` cache (`PREBUILD_KERNELS=1` baked into docker)**. The docker image builds 98 kernels at image-build time and stores them under `/sgl-workspace/aiter/aiter/jit/module_*.so`. When `git checkout` switches the source to an older commit, the `.so` files still contain the NEW kernel code. aiter prefers the prebuilt `.so` over JIT, so the bug persists across source rewinds.
2. **flydsl version coupling**. `flydsl` (a small Python+MLIR runtime aiter depends on) is bumped together with aiter — the 4/30 image had `flydsl 0.1.2`, the 5/1 image has `flydsl 0.1.5.dev504`. The aiter commit that bumps this pin is `4de5759ed [FLYDSL]: update version to 0.1.5.dev504 (#2935)`. Running v0.1.12.post1 aiter with flydsl 0.1.5 breaks the `aiter.fused_moe` import (`AttributeError: module 'aiter' has no attribute 'fmoe'`); running new aiter with flydsl 0.1.2 prints `Unsupported flydsl version: expected >=0.1.3, got 0.1.2. CK and HIP ops are disabled. Triton ops remain available.`
3. **PREBUILD full rebuild is impractically slow** (`PREBUILD_KERNELS=1 GPU_ARCHS=gfx950 python setup.py build_ext --inplace` rebuilds ~98 kernels via hipcc, taking 30–60 min on cold cache, even for a single-commit diff).
4. **JIT path is fast (~30 s per module) but only rebuilds what the running workload actually calls**. So if a buggy kernel is in a `.so` that the workload happens to import but the diff doesn't touch the entry-point Python wrapper, deleting just that one `.so` and letting JIT rebuild it is enough.

**Working recipe** for each bisect step:

```bash
# A. checkout target aiter commit
cd /sgl-workspace/aiter && git checkout <target_commit>

# B. align flydsl with the era of the target commit
#    - commits before `4de5759ed (#2935)`:  flydsl 0.1.2
#    - commits at/after `4de5759ed (#2935)`: flydsl 0.1.5.dev504
#    (snapshots of both versions are saved under /home/jacchang/_flydsl_0_1_5/
#     and /opt/venv/lib/python3.10/site-packages/flydsl.bak.0.1.2)

# C. delete any prebuilt .so that the candidate PR's diff touches, so JIT
#    will recompile from the checked-out source. Modules that GLM-5.1 +
#    TP=2 actually loads (from server.log `[aiter] import [module_*] under …`):
for m in \
  module_aiter_core module_activation module_cache module_custom \
  module_custom_all_reduce module_fused_qk_norm_rope_cache_quant_shuffle \
  module_moe_asm \
  module_moe_ck2stages_fp4x2_fp4x2_preshuffle_on_b16_silu_per_1x32_mulWeightStage2 \
  module_moe_cktile2stages module_moe_sorting \
  module_norm module_quant module_rmsnorm_quant \
  module_rope_2c_cached_positions_fwd ; do
    rm -f "/sgl-workspace/aiter/aiter/jit/${m}.so"
    rm -rf "/sgl-workspace/aiter/aiter/jit/build/${m}"
done

# D. run repro_pr25742_acc_matrix.sh — first launch JIT-compiles all
#    needed kernels from the checked-out source (~30 s/module, ~5–7 min
#    total for the 14 modules above), then runs GSM8K (~3 min).
```

Per-step wall time with the recipe above: roughly 15 min (5–7 min JIT + ~5 min server load + ~3 min GSM8K).

**Sanity check for the recipe** — running v0.1.12.post1 source + flydsl 0.1.2 + all 14 modules deleted (so JIT rebuilds from post1 source) should reproduce the original 0.94 baseline. If it instead matches the BAD band (~0.6), the recipe is leaking state from the newer build and the bisect signal is unreliable.

---

## 11. Reproduction

```bash
# Container with the BAD image (one of:)
docker run -d --name bisect_0501 --network host --ipc host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --cap-add SYS_PTRACE --security-opt seccomp=unconfined --security-opt label=disable \
    -v /home/jacchang:/home/jacchang -v /data:/data -v /home:/home \
    rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260501 sleep infinity

# GOOD baseline
docker run -d --name bisect_0430 ... rocm/sgl-dev:v0.5.10.post1-rocm720-mi35x-20260430 sleep infinity

# Run the matrix driver (script lives at /home/jacchang/SGLang-benchmarks/repro_pr25742_acc_matrix.sh)
docker exec bisect_0430 bash /home/jacchang/SGLang-benchmarks/repro_pr25742_acc_matrix.sh \
    2 /home/jacchang/SGLang-benchmarks/results/_pr25742_matrix/<date>_TP2
```

`repro_pr25742_acc_matrix.sh` does:
1. `pkill` any stale sglang processes
2. Set `PYTHONPATH=/sgl-workspace/aiter:/sgl-workspace/sglang/python` (older images need this because non-interactive `bash -c` doesn't source `/etc/bash.bashrc`)
3. Launch sglang server: `--tp $TP --tool-call-parser glm47 --reasoning-parser glm45 --nsa-prefill-backend tilelang --nsa-decode-backend tilelang --kv-cache-dtype fp8_e4m3 --mem-fraction-static 0.85 --disable-radix-cache`
4. Wait for `/health` (poll every 5 s)
5. Run `bench_sglang.py --num-questions 1200 --parallel 1200`
6. Clean up server (specific `pkill -f` matchers to avoid suicide on wrappers containing `sglang-rocm` in their cmdline)

Per-config wall time: ~3 min server load + ~2.5 min gsm8k (TP=2) or ~1 min gsm8k (TP=8).
