# Reproducing approximately 97% GSM8K on the GLM-5.3-Flash ROCm PR stack

This is the exact setup that gives approximately 97% GSM8K on MI355X with our GLM-5.3-Flash enablement stack, at both TP8 and TP4. `cuda graph` is enabled throughout; nothing here disables it.

**Read this first.** What decides whether you get 97% or garbage is *whether the very first thing the server prefills is several long prompts at once*. It is not the AITER revision, not the source tree, and not the server flags. Send four concurrent requests of roughly a thousand tokens each to a freshly started server, before it has served anything else, and every request after that fails to terminate for the life of the process. Send the same four sequentially, or send anything at all beforehand, and nothing happens. A single long request is safe at any length tested, including 8K.

The mitigation is therefore one line — **have the server answer one short request before it takes real traffic** — but read the rest of this section before relying on it, because the boundaries are only partly mapped.

Measured 2026-09-14. GSM8K over all 1,319 examples, `--thinking`, temperature 1.0, top-p 0.95, 4,096-token cap, `--num-threads 1319`, TP4, decode CUDA graphs active:

| stack | AITER | warmup | score | truncated | tokens emitted |
| --- | --- | --- | --- | --- | --- |
| `b7dda1a7ba` on `v0.5.18-rocm720-mi35x` | `d9e5ef7c` | none | **97.04%** | 0.30% | 171,798 |
| `b7dda1a7ba` on `v0.5.18-rocm720-mi35x` | `d9e5ef7c` | 4 × ~1000 tok, concurrent | **20.92%** | 75.89% | 4,218,323 |
| `main` + 9 PRs on `v0.5.19-…-20260909` | `4ad99832` | none | **96.74%** | 0.38% | 185,579 |
| `main` + 9 PRs on `v0.5.19-…-20260909` | `4ad99832` | 4 × ~1000 tok, concurrent | **20.24%** | 74.98% | 4,205,029 |

Both AITER revisions pass without the trigger and fail with it. The AITER revision moves the score by 0.3 points; the warmup moves it by 76. **This reproduces on the reference configuration this whole document is built around**, so the 97% below is not a property of that stack — it is a property of never having issued concurrent long prefills. GSM8K prompts are about 100 tokens, which is why the reference workflow never trips it.

The reproducer needs no benchmark harness and no unusual endpoint: four `POST /v1/chat/completions` carrying an ordinary English prompt of ~1,000 tokens with `max_tokens: 16`, issued from four threads.

The failure is non-termination, not wrong answers. Generations run to the cap — 989 of 1,319 in the poisoned run against 3 in the clean one — and most of them render as nothing at all: 854 of those 989 emitted 4,096 tokens of empty text. The rest carry the repeated-token signature (`I?N?` 605 times in one, `- Query: -1` 274 times in another). Of the generations that still stop, about 80% are correct, which is what makes this look like a quality regression rather than a broken server.

### Where the trigger's edges are

Same stack, same harness, varying only the warmup, then grading all 1,319 examples:

**Grade at full concurrency or you will not see this.** How visible the damage is depends on how much load follows the trigger, so a small evaluation can pronounce a poisoned server healthy. Eleven runs of the identical 4 × ~990 recipe on a fresh server, differing only in the first workload sent afterwards:

| first post-trigger load | outcome |
| --- | --- |
| 1,319 concurrent | **poisoned 6 / 6** — 20.24%, 20.32%, 20.92%, 21.23%, 31.01%, 35.41% |
| 128 concurrent | poisoned 1, degraded 1 (74.22%), clean 2 |
| ≤ 64 concurrent | clean 4 / 4 — up to 100% on a 64-question probe |

At full scale it is completely reproducible; below about 128 concurrent it is intermittent, and at 32–64 it is invisible. A 64-concurrency probe on a server that had just been hit with the trigger scored 100%, and a 512-concurrency probe on that same server afterwards was still clean — so a small first workload appears not merely to hide the damage but to clear it.

Every boundary in the tables below was graded at 1,319 concurrency, which is the regime where the effect is deterministic. Do not re-derive any of them with a cheap probe.

Two conditions must hold together for the trigger itself. The batch has to contain **several sufficiently long prefills**, and it has to be the **first substantial traffic the server serves**. Either one alone was harmless.

First, with the warmup as the server's first traffic:

| first traffic | per request | concurrency | score |
| --- | --- | --- | --- |
| none | — | — | 96.74% – 97.50% |
| `bench_serving` random tokens | ~920 tok | 1 | 96.74% |
| natural text, `/v1/chat/completions` | 1,840 tok | 1 | 97.12% |
| natural text, `/v1/chat/completions` | **8,020 tok** | 1 | **96.29%** |
| `bench_serving` random tokens | 64 tok | 4 | 96.89% |
| `bench_serving` random tokens | ~460 tok | 4 | 97.04% |
| `bench_serving` random tokens | ~920 tok | 4 | **33.43%** |
| natural text, `/v1/chat/completions` | ~990 tok | 4 | **35.41%** |
| natural text, `/v1/chat/completions` | 1,001 tok | 4 | **20.24%** |

A single 8,020-token prefill is clean, so **a lone long request is safe however long it is** — up to 8K, at least. Four concurrent ~1,000-token prefills are not. Four concurrent ~460-token ones are. The per-batch threshold therefore sits between ~1,840 and ~3,680 prefill tokens, and the per-request floor between ~460 and ~920 is untested, as is the 2- and 3-concurrent case.

Second, and this is what makes it survivable in practice: **anything the server serves beforehand immunises it.** The same 4 × ~990 batch that poisons a fresh server is harmless once the server has answered something else first:

| sequence | score after the 4 × ~990 batch |
| --- | --- |
| fresh server → 4 × ~990 → grade | **35.41%** |
| fresh server → 16 short questions → 4 × ~990 → grade | clean |
| fresh server → 64-question GSM8K → 4 × ~990 → grade | 100.00% (64 q) |
| fresh server → 1,319-question GSM8K → 4 × ~990 → grade | **97.50%** |

So the practical mitigation is one line: **send a single short request to a newly started server before it takes real traffic.** All three inoculated runs were graded at 1,319 concurrency, the regime where an uninoculated server fails 6 out of 6, so this is a real result rather than an artifact of a weak probe — but it is one observation per workload, not a guarantee. How small the inoculating request can be is untested; the smallest tried is a batch of 16 short questions.

It is not `bench_serving`, not random token IDs, and not the `/generate` endpoint: ordinary chat traffic reproduces it. This model's `index_topk` is 2,048 — printed at startup as `Set dense attention kv len threshold to model index_topk=2048` — which falls inside the ~1,840-to-~3,680 window. That is suggestive and nothing more.

### It is not the ROCm enablement PRs

Andy's original combined enablement, [#36607](https://github.com/sgl-project/sglang/pull/36607) at its merge commit `aa8c950a3d` — the tree before the revert, on the image and AITER revision it was validated against — fails the same way. Same trigger, same 1,319-concurrency grading, three fresh servers:

| tree | image / AITER | trials | poisoned | scores |
| --- | --- | --- | --- | --- |
| `aa8c950a3d` — Andy's #36607, pre-revert | v0.5.18 / `d9e5ef7c` | 3 | **3 / 3** | 50.42%, 50.04%, 53.68% |
| `b7dda1a7ba` — the seven-PR re-land | v0.5.18 / `d9e5ef7c` | 1 | 1 / 1 | 20.92% |
| `main` + the current stack | v0.5.19 / `4ad99832` | 6 | **6 / 6** | 20.24% – 35.41% |

So the defect predates the re-split and is not something the Day-0 PRs introduced. **It should not be used to block them.**

It is also not in the enablement code at all. Comparing the two patches rather than the two trees — Andy's touches 14 non-test files, the current stack 17, with 11 in common — **neither functionally touches the linear-attention state path**. The current stack's only edit to `mem_cache/memory_pool.py` is three comment lines; Andy's patch does not touch that file. The state pool, its allocator, `schedule_batch` and `model_runner` are all untouched by both. That puts the defect in upstream sglang's hybrid mamba/DSA machinery, which is platform-neutral — so this plausibly affects GLM-5.3-Flash on CUDA as well. Untested here; there is no B200 on this box.

One asymmetry is worth recording rather than explaining away: Andy's tree lands at ~50% while both re-landed stacks land at 20–35%, and the two bands do not overlap across nine runs. Something in the re-split makes an upstream bug worse. The three files Andy patched that the current stack does not are the only candidates: `layers/communicator_mhc.py` (the fused mHC attention-to-FFN boundary), `layers/moe/moe_runner/triton_utils/fused_moe.py` (the HIP Triton SwiGLU clamp, still open as [#37769](https://github.com/sgl-project/sglang/pull/37769)), and `server_args.py` (the non-gfx95 DSA top-k fallback).

### Mechanism: what has been ruled out

The model is a hybrid — 34 of its 45 layers are KDA linear attention, which carry per-sequence recurrent state, and the server allocates a 76 GB Mamba state pool for them (`max_mamba_cache_size=2220`, one slot per request). GLM-5.2 is pure DSA and has no such pool, which is the obvious reason it never shows this. That made the state pool the leading suspect. Two experiments say otherwise:

| experiment | result | what it rules out |
| --- | --- | --- |
| `--mamba-ssm-dtype bfloat16` (slots 2,220 → 4,296, `ssm_state` 73.7 → 71.3 GB) | still poisoned, 21.23% | the state's size, layout and numeric precision |
| instrumented slot allocator: log duplicate slots per batch, slots reissued while live, and every `clear_slots` call | **0 aliases, 0 reissues, 32 `clear_slots` calls, 0 skipped** | slot aliasing and missed zeroing — allocation and clearing are correct |

So the mamba *bookkeeping* is right. What it does not establish is **ordering**, and that is where the evidence now points.

`clear_slots` carries the docstring *"Zero out mamba state at the given pool indices. Must run on forward stream"*, and its caller adds *"before the mamba layers read the pool, so the copies don't race the scheduler copy stream"* — the code already knows it is racing something. Two further facts line up: `free_mamba_cache` returns a slot to the free list purely on the CPU side, with no GPU synchronisation, so a slot can be reissued while the previous owner's state writes are still in flight; and `glm5_next` runs part of its forward on a second stream (`self.alt_stream = torch.cuda.Stream()`).

Inserting a global `torch.cuda.synchronize()` around the clear changes the failure rate, which is what makes this an ordering problem rather than a numerics one. All rows graded at 1,319 concurrency:

| barrier | trials | poisoned | scores when clean |
| --- | --- | --- | --- |
| none | 6 | **6 / 6** | — |
| immediately **after** `clear_slots` | 2 | 0 / 2, but both degraded | 74.53%, 76.42% |
| immediately **before** `clear_slots` | 5 | **1 / 5** | 96.59%, 96.97%, 97.12%, 97.42% |

Placing the barrier before the clear helps much more than placing it after, which rules out "the barrier merely perturbs timing" — a pure timing effect would be roughly symmetric. The required ordering is therefore *the clear must follow work that precedes it*, consistent with a slot being reissued before its previous owner's writes have retired.

**Neither placement is a fix.** A barrier before the clear still poisoned 1 of 5. So either the window is only narrowed rather than closed, or more than one unsynchronised interaction is involved. The precise racing party is not identified.

A note on method, since this cost real time: two conclusions in this investigation were drawn from n=2 and both were wrong — first "the trigger is probabilistic" (an artifact of comparing runs with different probe sizes), then "a barrier before the clear fixes it" (2/2 clean, which became 4/5 at n=5). At roughly 15 minutes per trial and effect sizes this noisy, **no barrier-placement claim here is worth making below n=5.** Continuing to bisect the race statistically is expensive; the cheaper path is direct instrumentation of what the mamba layer actually reads for a slot that was just marked for clearing.

A second candidate remains open and untested: a shape-keyed kernel-selection cache — AITER logs `not found tuned config ... will use default config` per shape — populated wrongly by whatever batch arrives first and then reused for later shapes.

**Any GSM8K number `GLM.sh` reports for this model is void.** Its `warmup()` is `bench_serving --random-input 1024 --max-concurrency 4 --num-prompt 4`, which is precisely the trigger, and it runs before the grading.

### How this document got it wrong twice

The original claim here was that AITER `4ad99832` broke the model and `d9e5ef7c` was the fix. The failing measurements were taken through `GLM.sh` and the passing ones through a hand-written launch that only ever sent short prompts, so the harness changed at the same time as AITER and AITER took the blame.

The 2026-09-14 revision correctly demoted that to "configuration-dependent, not broken", but named `--max-running-requests` as the leading hypothesis. That is also wrong. `GLM.sh`'s environment block, its extra server flags, and both together were each re-tested on the 9-PR stack at 1,319 threads and scored 96.82%, 97.27% and 96.74% — the entire server-flag space is clean. Adding the warmup to that same 96.74% configuration dropped it to 33.43%. Both earlier conclusions came from comparing two runs that differed in more than one thing.

### A separate, still-valid AITER warning

Independent of everything above, AITER `c16d44b93a528b2a4bfd6d8d3409116d465872a9`, which ships inside `rocm/sgl-dev:v0.5.18-rocm720-mi35x-20260901`, hard-aborts this model during decode:

```text
:0:rocdevice.cpp :3586: Callback: Queue aborting with error :
HSA_STATUS_ERROR_EXCEPTION: An HSAIL operation resulted in a hardware exception
Fatal Python error: Aborted
```

When it survives the abort it emits corrupted tokens instead — `" Paris!!!!!!!!"` for `"The capital of France is"`. AITER `d9e5ef7ce08ee7045d583aed768cff41aa9210fe` runs the identical source tree at 97.19%. To reproduce the numbers on this page, use the pinned image below; a newer one is not automatically unusable, but it is no longer the configuration these numbers describe.

The 17-19% GSM8K figures this document used to attribute to AITER `4ad99832` were all measured through `GLM.sh`, i.e. with the trigger. They are reproducible — 20.32% and 31.01% on two repeats — but they are measurements of the concurrent-prefill bug, not of that AITER revision. The same AITER scores 96.74% on the same image and tree when the warmup is removed, and `d9e5ef7c` scores 20.92% when it is added. Do not cite them as an AITER comparison.

Swapping AITER inside a given image is still constrained, which matters if you want to bisect the hard abort above. AITER's flydsl kernels are written against the `flydsl` package baked into the image at `/opt/venv/lib/python3.10/site-packages/flydsl`, and the MLIR op signatures it exposes move between images. Pointing `PYTHONPATH` at an AITER `d9e5ef7c` worktree inside the v0.5.19 image — with its matching composable_kernel revision `f33252ce` and a fresh JIT cache — gets as far as kernel compilation and then dies:

```text
File "/opt/venv/lib/python3.10/site-packages/flydsl/compiler/kernel_function.py", line 602, in _emit_kernel
File ".../aiter-d9e5ef7c/aiter/ops/flydsl/kernels/buffer_ops.py", line 678, in buffer_store
    rocdl.RawPtrBufferStoreOp(
TypeError: RawPtrBufferStoreOp.__init__() takes 5 positional arguments but 6 were given
```

That constraint is real but narrower than "you cannot change AITER without changing the image", which is what this document used to say. The image ships `/sgl-workspace/aiter` as a full git checkout — `d9e5ef7c` is reachable from `4ad99832`, 338 commits back — and what actually breaks is checking out a revision older than the flydsl API change. That change is `da03212a7`, ROCm/aiter#4436 *"update flydsl version and Adapt flydsl kernels to internal LLVM ROCDL API changes"*, dated 2026-08-17. It splits the range in two:

| range | commits | in the v0.5.19 image |
| --- | --- | --- |
| `d9e5ef7c..da03212a7` | 176 | will not load — this is the `RawPtrBufferStoreOp` failure above |
| `da03212a7..4ad99832` | 162 | checks out, rebuilds against a fresh JIT cache, and runs |

So AITER *is* bisectable inside one image, over the upper 162 commits. Anything below `da03212a7` needs a different image.

The hard abort above is a separate question and is established by a three-way comparison, each with its own empty cache and the same source tree: `d9e5ef7c` passes, `d9e5ef7c` without `SGLANG_OPT_USE_TOPK_V2=0` also passes, and `c16d44b9` with a brand-new cache still aborts. Why it aborts is not identified. Note that the suspect list this document used to carry — the gluon paged-MQA-logits changes ROCm/aiter#4774 and ROCm/aiter#4440, and the Triton MoE routing fix ROCm/aiter#4530 — was assembled to explain the non-termination, which turned out not to be an AITER problem at all. Those three are not evidence for anything now.

## Source

```bash
git clone https://github.com/Jacob0226/sglang.git
git -C sglang checkout jacob/glm53-rocm-stack-0903
```

```text
HEAD: b7dda1a7bae27295047e1e198a80caac3e5cc338
Base: dad2ed4421dcd0e999d0ed71bd5d14a15efebbca   (xinyuan/glm-5.3-flash-support)
```

The branch is the seven Day 0 PRs that existed when it was cut, applied in order. Three of Max's commits belong to one PR, which is why nine commits map to seven PRs. **The live stack is now eight** — #39317 joined it after this branch was measured.

**The original PR numbers are dead, and so are four of their replacements.** The first seven were closed and reopened against `main` as [#38541](https://github.com/sgl-project/sglang/pull/38541)–[#38547](https://github.com/sgl-project/sglang/pull/38547) under `Raiden-Makoto`. On 2026-09-13 Max closed #38541–#38544 so Jacob could carry his own four again; they are now [#39338](https://github.com/sgl-project/sglang/pull/39338)–[#39341](https://github.com/sgl-project/sglang/pull/39341), byte-identical to the #38541–#38544 heads and based on `main`. #38545–#38547 are unchanged. Four of the titles were rewritten on 2026-09-15 to say what the change enables rather than how; the table carries the current ones.

The commit column is the branch commit that produced the measurements on this page, not the reopened one. #39317 has no entry there because it post-dates the branch.

| PR | Was | Commit measured here | Title | Owner |
| --- | --- | --- | --- | --- |
| [#39338](https://github.com/sgl-project/sglang/pull/39338) (was #38541) | #37530 | `9752e69b32` | Enable zero-RoPE MHA prefill on ROCm | Jacob |
| [#39339](https://github.com/sgl-project/sglang/pull/39339) (was #38542) | #37563 | `5f5f51b1d7` | Build the fused DSA k-pool top-k JIT kernel on HIP | Jacob |
| [#39340](https://github.com/sgl-project/sglang/pull/39340) (was #38543) | #37573 | `f9c73a066a` | Support non-2048 top-k widths in the DSA page-table transform | Jacob |
| [#39341](https://github.com/sgl-project/sglang/pull/39341) (was #38544) | #37685 | `dac4e6c154` | Enable the k-pool DSA indexer on gfx950 | Jacob |
| [#38545](https://github.com/sgl-project/sglang/pull/38545) | #37626 | `323430c5b0` | Route mHC through AITER on gfx950 | Max |
| [#38546](https://github.com/sgl-project/sglang/pull/38546) | #37629 | `a6c7efa61f`, `f5da12ecac`, `05d78dac17` | Enable FP8 and Quark MXFP4 MoE on gfx950 | Max |
| [#38547](https://github.com/sgl-project/sglang/pull/38547) | #37653 | `b7dda1a7ba` | Enable zero-RoPE TileLang DSA on gfx950 | Max |
| [#39317](https://github.com/sgl-project/sglang/pull/39317) | #37673 | — | Honor fused and per-expert names in quark `exclude` | Arist12, recovered by Max |

[#37769](https://github.com/sgl-project/sglang/pull/37769) kept its number and is still open. It is not part of this stack and is not needed for these numbers: it unblocks `--moe-runner-backend triton` on HIP, which the AITER runner never enters.

Two things changed in the reopen beyond the numbering. #38544 gained a commit the old #37685 did not have, `825ab39b46` "[ROCm] Bound the k-pool indexer's AITER MQA logits at 2 GiB", so it is new code that the results on this page never exercised. And they are no longer a stack: each one is independently based on `main` and carries its own `Merge branch 'main'` tip, so there is no longer a branch to check out — see the next section for how to apply them.

## The MXFP4 checkpoint needs the loading work, which #38546 and #39317 now carry

**Superseded 2026-09-14.** This section was written when the loading work sat in #38998 and #38999. It no longer does. On 2026-09-13 Max folded Andy's commit `b1416885b6` "[AMD] Support mixed Quark MXFP4 and block-FP8 loading" into [#38546](https://github.com/sgl-project/sglang/pull/38546), and [#39317](https://github.com/sgl-project/sglang/pull/39317) — @Arist12's recovered #37673 — carries the `quark/utils.py` half. Between them they are functionally identical to #38998 + #38999; a file-by-file diff leaves only docstrings, comments and import formatting. **#38998 and #38999 are now redundant.** The stack is eight PRs, not nine: #39338–#39341 (the recovered Jacob PRs, replacing #38541–#38544) plus #38545, #38546, #38547 and #39317.

Verified on that eight-PR stack, all merging cleanly onto `main`, TP4 on MI355X: `amd/GLM-5.3-Flash-Quark-MXFP4` loads with no shape assertion and no missing or unexpected parameter, and grades **96.51%** GSM8K over all 1,319 examples (99.70% stop, 0.30% truncated, 0.00% error). `SGLang-benchmarks/tools/glm53_mxfp4_verify.sh` runs it; `tools/glm53_mxfp4_quant_resolution.py` checks the per-layer decisions below without a GPU.

The split between the two PRs is real and both are needed. #38546 fixes `layer_quant_config`; #39317 fixes the fused `exclude` match. Re-measured on the three trees:

| Queried layer | `main` | + #39338–41, #38545–47 | + #39317 |
| --- | --- | --- | --- |
| `model.layers.7.self_attn.q_a_proj` | `fp4` `per_group` | `fp8_e4m3` `per_block` [128,128] | unchanged |
| `model.layers.45.mlp.experts.7.up_proj` | `fp4` `per_group` | `fp8_e4m3` `per_block` [128,128] | unchanged |
| `visual.blocks.0.attn.qkv_proj` | not excluded | **still not excluded** | **excluded** |

The rest of this section explains *why* the checkpoint fails without that work, which is unchanged and still worth reading. Substitute "#38546 and #39317" wherever it says "#38998 and #38999".

The loading support existed once. [#36607](https://github.com/sgl-project/sglang/pull/36607) — the original combined gfx942/gfx950 enablement, which Andy validated on this very checkpoint at TP4 97.19% and TP8+EP8 97.12% — carried a follow-up commit for "mixed Quark MXFP4 and explicitly block-quantized FP8 checkpoint loading". That whole PR was then reverted by `c767511ea`, and the seven-PR re-split re-landed the kernel and MoE work but not the checkpoint-loading work. #38546 does own the MXFP4 MoE weight and runner path, so what is missing is narrower than it sounds: only the decision of *which quantization method each layer is given*.

Why it is silent. Quark exports this model as globally MXFP4 with a list of per-layer exceptions pinned to block FP8, and it names those layers the way the checkpoint does. `QuarkConfig.apply_weight_name_mapper` rewrote only `exclude_layers`, so `layer_quant_config` kept the checkpoint's names, every lookup missed, and `_find_matched_config` returned the *global* config on a miss because it had no way to say "no explicit entry". Measured on `main` plus the first seven PRs, asking with the runtime name that sglang actually uses:

| Queried layer | Without the loading work | With it |
| --- | --- | --- |
| `model.layers.7.self_attn.q_a_proj` | `fp4` `per_group` | `fp8_e4m3` `per_block` [128,128] |
| `model.layers.45.mlp.experts.7.up_proj` | `fp4` `per_group` | `fp8_e4m3` `per_block` [128,128] |
| `model.layers.3.mlp.experts.0.up_proj` | `fp4` `per_group` | `fp4` `per_group` |
| `visual.blocks.0.attn.qkv_proj` | not excluded | excluded |

What the checkpoint actually contains, since these numbers decide how much the bug costs:

| | Count | Note |
| --- | --- | --- |
| `layer_quant_config` entries | 924 | every one block FP8 `[128,128]`; all 924 missed |
| — attention projections | 48 | `q_a`/`q_b`/`kv_a_proj_with_mqa`/`o_proj` in layers 3,7,…,43,45 |
| — expert weights | 864 | all in layer 45, the MTP layer |
| `exclude` entries | 626 | 125 vision, 24 of them a pre-fused `attn.qkv` |
| `*.weight_scale` tensors | 37,338 | the runtime registers `weight_scale_inv` for block FP8 |

The 48 attention entries sit in the main model, so this bites even with speculative decoding off; the 864 expert entries only matter once the MTP layer is loaded.

The two PRs, both opened 2026-09-11 as drafts off `main` `480b14eda`. **Historical — do not apply these; #38546 and #39317 carry the same changes.** They are kept here because the file/behaviour split is the clearest description of what the loading work actually does:

| PR | Files | What it does | Branch |
| --- | --- | --- | --- |
| [#38998](https://github.com/sgl-project/sglang/pull/38998) | `quark/quark.py`, `quark/utils.py` | Route `layer_quant_config` and `kv_cache_group` through the mapper; split `_find_matched_layer_config` so a miss returns `None`; dispatch explicitly pinned layers to `Fp8LinearMethod` / `Fp8MoEMethod`; raise on partially specified fused shards; honor a direct `exclude` match on an already-fused module | `jacob/glm53-day0-quark-mixed-mxfp4-block-fp8` |
| [#38999](https://github.com/sgl-project/sglang/pull/38999) | `models/glm5_next.py` | Add `.attn.qkv` → `.attn.qkv_proj` to `hf_to_sglang_mapper`; map `.weight_scale` → `.weight_scale_inv` at the load sites, only where the runtime exposes it | `jacob/glm53-day0-quark-weight-name-mapping` |

Neither breaks anything without the other and they can land in either order, but MXFP4 needs both. #38998 is model-agnostic and applies to any Quark checkpoint that mixes precisions.

One thing shrank the second PR after this document's gap analysis was written: [#38621](https://github.com/sgl-project/sglang/pull/38621) (NVIDIA's ModelOpt NVFP4 loading, merged 2026-09-10) already added `hf_to_sglang_mapper` to `Glm5NextForConditionalGeneration`, using `orig_to_new_substr` and without the `attn.qkv` suffix rule. #38999 extends that object rather than introducing it, and NVFP4's exclusion globs are unaffected because `model.visual*` does not end in the suffix.

Three pieces of #36607 are still **not** re-landed and are not in the eight: shared-experts fusion on AITER gfx95 (`glm5_next.py` still refuses with "requires CUDA devices"), the fused mHC attention-to-FFN boundary (`communicator_mhc.py` plus `hc_attn_to_mlp`), and the HIP Triton SwiGLU clamp, which is the still-open [#37769](https://github.com/sgl-project/sglang/pull/37769). None of them is needed to load or grade MXFP4, but the second one means `GLM.sh`'s `check_mhc_markers` will report `fused attn->FFN boundary: 0/4 ranks` and print its "5.42x slower" warning. That warning is calibrated against the *pre-mHC* head, not against a stack that has #38545's pre/post kernels and only lacks the boundary fusion, so on the eight-PR stack it overstates the gap.

### MXFP4 accuracy, as measured when the loading work was #38998 + #38999

One variable — same image, same tree, same TP4, same sampling, same harness; only the checkpoint differs. GSM8K all 1,319 examples, `--thinking`, temperature 1.0, top-p 0.95, 4,096 max output tokens, `--num-threads 1319`, decode CUDA graphs active, one scoring pass each:

| Checkpoint | score | stop | truncated | error | wall clock |
| --- | --- | --- | --- | --- | --- |
| `zai-org/GLM-5.3-Flash`, block FP8 | 97.50% | 99.85% | 0.15% | 0.00% | 304.3 s |
| `amd/GLM-5.3-Flash-Quark-MXFP4` | **96.89%** | 99.39% | 0.61% | 0.00% | 305.5 s |

Read the 0.61-point gap as parity pending a repeat pass, not as a demonstrated regression, and do not quote it as either without one. It is at the edge of this model's run-to-run spread rather than inside it: Andy's six reference runs spanned 96.82% to 97.35%, and #36607 measured 97.19% for this same MXFP4 checkpoint at TP4, which is 0.30 above what is measured here. A 300-example pass on the same server read 98.00%, which is the small-sample noise you would expect at n=300 and is not evidence of anything.

## Applying the reopened PRs to a current main-based image

The branch above is the tree the numbers came from, and it is the only tree those numbers describe. If instead you are starting from a recent ROCm image whose in-tree sglang is a `main` snapshot, apply the reopened PRs on top of it. This is the path to use when you want the enablement on current `main`; it is **not** a reproduction of the 97% figure, because it moves both the sglang base and the AITER revision away from what was measured.

Why the stack is needed at all is worth stating concretely, because a stock `main` image does not fail with anything that names mHC. `glm5_next.py` calls `sglang/kernels/ops/layernorm/mhc.py`, whose default path (`SGLANG_OPT_DEEPGEMM_HC_PRENORM=1`) goes to deep_gemm, which is CUDA-only. The server loads weights fine and then dies during decode CUDA-graph capture with:

```text
File "python/sglang/srt/layers/deep_gemm_wrapper/entrypoint.py", line 289, in tf32_hc_prenorm_gemm
    deep_gemm.tf32_hc_prenorm_gemm(x, fn, out, sqrsum, num_splits=num_splits)
NameError: name 'deep_gemm' is not defined
```

#38545 is what routes that call to AITER instead. After it is applied,
`python3 -c 'from sglang.srt.models.glm5_next import _use_aiter_gfx95; print(_use_aiter_gfx95)'`
prints `True` on gfx950, which is the cheapest check that the stack took effect.

All eight are independently based on `main`, so merging the PR heads is the way to apply them. It is simpler than selecting commits out of them, and it keeps each author's own `Merge branch 'main'` resolution instead of making you redo it:

```bash
cd /sgl-workspace/sglang
git checkout -b glm53-day0-stack upstream/main
for n in 39338 39339 39340 39341 38545 38546 38547 39317; do
  git fetch -q upstream "pull/$n/head:pr/$n"
  git merge --no-ff -m "merge #$n" "pr/$n"
done
```

All eight merged with no conflict against `main` on 2026-09-14. **Do not also apply #38998 and #38999** — their content is already in #38546 and #39317, and applying both conflicts in `quark/utils.py`.

<details>
<summary>The cherry-pick recipe that produced the numbers on this page (historical — prefer the merge form above)</summary>

It fetches each PR head and cherry-picks its non-merge commits, in PR order, skipping the `Merge branch 'main'` commits because they carry unrelated main history. Note the PR numbers are the pre-2026-09-13 ones.

```bash
cd /sgl-workspace/sglang
for n in 38541 38542 38543 38544 38545 38546 38547; do
  git fetch -q origin "pull/$n/head:pr/$n"
done
git checkout -b glm53-day0-pr38541-38547

git cherry-pick -x 2dce4407c3                                                   # 38541
git cherry-pick -x 3aca7ab2f7 31878a1028                                        # 38542
git cherry-pick -x b354503857                                                   # 38543
git cherry-pick -x 71e1739c1b 846d708e45 825ab39b46                             # 38544
git cherry-pick -x ac7ce35d8a 1b66f0ff70 feb62aae99 95dce2acae                  # 38545
git cherry-pick -x bbfc962eae 93b57828fa 045c63a536 61d0c9f367 7ffbfcf704 73ed73c0d5   # 38546
git cherry-pick -x 7e122b0278 54624dd98f                                        # 38547
```

</details>

Note the remote: these are fetched from `upstream` (`sgl-project/sglang`). If you are working in `~/PR/sglang`, `origin` there is the `Jacob0226` fork and its `main` is a stale divergent branch — `quark.py` is 533 lines against upstream's 1,100, and it is not an ancestor of `upstream/main`. Basing anything on `origin/main` silently gives you a tree that looks like `main` and is not.

Nine of those are functional: `2dce4407c3`, `3aca7ab2f7`, `b354503857`, `71e1739c1b`, `825ab39b46`, `ac7ce35d8a`, `bbfc962eae`, `045c63a536`, `7e122b0278`. The rest are test registrations, and two of #38542's — `c184340d8c` and `c1bb21cb79` — are omitted from the list above because they collide with main's test-taxonomy rename (`test/registered/kernel/attention/` vs `test/registered/unit/kernels/`). Skipping them is safe and changes no runtime code; if you want a rule rather than a list, a conflicting commit whose every path is under `test/` can be dropped, and one that touches anything else cannot.

**Whether they apply cleanly depends on how far `main` has moved, and that is an artifact of cherry-picking rather than a property of the PRs.** All nineteen land with no conflict on the `main` snapshot inside `v0.5.19-rocm720-mi35x-20260909` (`ffe98a4279`). Two days later, against `480b14eda`, `71e1739c1b` conflicts in `dsa_indexer_kpool.py`: `main` moved `is_in_breakable_cuda_graph` into its own module while the commit adds `get_exec` to the neighbouring `runtime_context` import. Keep both sides —

```python
from sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph import (
    is_in_breakable_cuda_graph,
)
from sglang.srt.runtime_context import get_device, get_exec
```

The PR itself is not conflicted, and GitHub is right to say so. #38544's head `038ba83de` is a `Merge remote-tracking branch 'upstream/main'` commit, so the author already merged `main` in and resolved this there; merging the PR head into `main` is clean. Skipping the merge commits — which this section tells you to do, because they drag in unrelated main history — also skips that resolution, so the conflict is yours to redo. Expect one more of these each time `main` advances, and do not read them as the PRs going stale.

Note that these images build sglang as an editable install pointing straight at `/sgl-workspace/sglang/python`, so the cherry-picks are live with no reinstall. They also leave that tree dirty with the image's own ROCm build edits (`python/pyproject.toml`, `python/sglang/kernels/aot/pyproject.toml`, a deleted `pyproject_rocm.toml`, and a pile of untracked `.hip` files). Save them with `git diff > mods.patch` before any `git reset --hard`, and restore with `git apply --3way`. They are build configuration for an already-compiled `sgl_kernel`, so they do not affect serving, but losing them silently makes the container harder to reason about later.

The working tree must be clean at `b7dda1a7ba`. Verify the tree and the model:

```bash
export SGLANG_SRC=/absolute/path/to/sglang
export MODEL_PATH=/data/huggingface/hub/zai-org/GLM-5.3-Flash

cd "$SGLANG_SRC"
git rev-parse HEAD
git status --short

sha256sum \
  python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py \
  python/sglang/srt/layers/attention/dsa/kpool_fp8_index.py \
  python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py \
  python/sglang/kernels/ops/layernorm/mhc.py \
  python/sglang/srt/models/deepseek_common/attention_forward_methods/forward_mha_rocm.py

sha256sum "$MODEL_PATH/config.json" "$MODEL_PATH/model.safetensors.index.json"
```

Expected:

```text
57b70ec973871390d83d27ba4f9a0335db00a90ec564a8125dc7bcf2864de243  dsa_indexer_kpool.py
09d3681de29f5fe579a3ea812c2b0ad63bf50d5d5b725ef37da2625d5a243bf9  kpool_fp8_index.py
433b21f717deda5fe481378401abc27cb624ea1fead3d76e9da0f280371a2841  tilelang_kernel.py
1d562f4deb38a383398b7ffbeb753ce90b1979b990950154387bd366ea5dd05a  mhc.py
ce650fb76ef6cfd099c532b6e56995271f433e4560e103ae4c833771bcc7d3c3  forward_mha_rocm.py

bb8f01c42cb92a52ca72e65afb4d5bd8d11aef083cd210e8de25dfb904f23e9f  config.json
3c3f40366a53c3fd7974b4eab7881a365a98c2a4329150befebab99fe7c18b05  model.safetensors.index.json
```

## Environment

```text
Image:        lmsysorg/sglang:v0.5.18-rocm720-mi35x
Image digest: sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72
AITER:        d9e5ef7ce08ee7045d583aed768cff41aa9210fe   (shipped in that image)
ROCm:         7.2.0
Model:        zai-org/GLM-5.3-Flash, revision 3f1971b7b5f7a528c9c4ef6212c8785298a8c24a
sgl-eval:     ns_commit_sha 645cf567ff08c0ae9cc3fc8e1edbb975b3067816
```

`ns_commit_sha` is a field sgl-eval writes into `metrics.json`, not a git ref you can fetch — `git+https://github.com/sgl-project/sgl-eval.git@645cf567...` fails with `not our ref`. Installing sgl-eval at `a231b7a439b235090ff7baa30778fa2b514309ae` produces that same `ns_commit_sha` value, so use that.

```bash
docker run -d --name glm53_repro \
  --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video \
  --cap-add SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64g \
  -v /data:/data -v "$HOME:$HOME" \
  lmsysorg/sglang@sha256:6d68cd19206716cb3f1e31e2ad89cd0852d7ae614a792773c30a4277f8955c72 \
  sleep infinity

docker exec glm53_repro git -C /sgl-workspace/aiter rev-parse HEAD
# must print d9e5ef7ce08ee7045d583aed768cff41aa9210fe

docker exec glm53_repro python3 -m pip install \
  "git+https://github.com/sgl-project/sgl-eval.git@a231b7a439b235090ff7baa30778fa2b514309ae"
```

## Cache hygiene

To be clear up front: cache state is *not* what breaks accuracy here. Giving the broken AITER revision a brand-new empty cache still aborts, which is how it was ruled out. The rules below are hygiene that keeps a debugging session honest, not a fix for the failure this document is about.

Give every cold server its own empty cache directory, and never start a second server against a cache another server is still populating.

This matters because AITER's JIT products are keyed by module name alone — the cache holds `module_mhc.so`, `module_moe_fmoe_asm.so`, `module_gemm_a8w8_blockscale_bpreshuffle.so` and so on, with no AITER revision anywhere in the path.

Left alone, AITER builds into `/sgl-workspace/aiter/aiter/jit`, which lives inside the image, so simply switching containers already gives you a clean cache and no cross-version mixing. The exposure is created by the `AITER_JIT_DIR` override below, which moves that directory onto the host so a run is reproducible and isolated. Once it is on the host, two different AITER checkouts pointed at one directory will silently share binaries: the second server finds the module already built and loads an object compiled from the other source tree. So the rule is per *cache directory*, not per container — a new AITER revision, or a new image, needs a new directory.

`AITER_JIT_DIR` is the only one of the two that does anything. Earlier revisions of this document also exported `AITER_ROOT_DIR`, which is *not* read from the environment: `aiter/jit/core.py` computes it as `os.path.abspath(f"{this_dir}/../../")` from the module's own location. Exporting it is at best inert, and pointing it at an empty directory only hides the tuned-config CSVs that the same file resolves against it. Set `AITER_JIT_DIR` and leave `AITER_ROOT_DIR` alone.

Killing a server while it compiles is the other half of this. It leaves `build/lock_module_*` and half-written artifacts behind, and a later run that loads them produces run-to-run corruption that looks exactly like a kernel bug. That one bites inside a single container too, override or not.

```bash
export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=1

export RUN_CACHE=/absolute/path/to/new-empty-cache
rm -rf "$RUN_CACHE" && mkdir -p "$RUN_CACHE"

export AITER_JIT_DIR="$RUN_CACHE/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang"
export TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg"
export SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

python3 -c 'import inspect, sglang; print(inspect.getfile(sglang))'
# must resolve under $SGLANG_SRC/python/sglang
```

## Launch

TP8:

```bash
python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 8 \
  --trust-remote-code \
  --kv-cache-dtype bfloat16 \
  --context-length 131072 \
  --mem-fraction-static 0.85 \
  --disable-radix-cache \
  --dsa-prefill-backend tilelang \
  --dsa-decode-backend tilelang \
  --moe-runner-backend aiter \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --host 0.0.0.0 \
  --port 30000
```

For TP4, export `HIP_VISIBLE_DEVICES=0,1,2,3` and `ROCR_VISIBLE_DEVICES=0,1,2,3` and change `--tp 8` to `--tp 4`. Everything else is identical.

Cold startup compiles the AITER, TileLang and Triton modules: 9 minutes for TP8 and 6 minutes for TP4 here, on an empty cache. Wait for the health check before starting the evaluator:

```bash
curl --fail http://127.0.0.1:30000/health
```

## Evaluate

```bash
ulimit -n 65535

python3 -m sgl_eval.cli run gsm8k \
  --base-url http://127.0.0.1:30000/v1 \
  --model "$MODEL_PATH" \
  --num-examples 1319 \
  --num-threads 1200 \
  --max-tokens 4096 \
  --temperature 1.0 \
  --top-p 0.95 \
  --seed 0 \
  --thinking \
  --out-dir ./gsm8k_ourstack
```

Greedy decoding is not a substitute. At `temperature 0` this model loops on its own reasoning and scores near zero, which is what greedy decoding does to a thinking model rather than a defect in the stack.

## Results

| Parallelism | score | stop_rate | truncated_rate | error_rate | wall clock |
| --- | --- | --- | --- | --- | --- |
| TP8 | **0.971948** | 99.77% | 0.23% | 0.00% | 90.9 s |
| TP4 | **0.969674** | 99.70% | 0.30% | 0.00% | 316.1 s |

Both are single scoring passes over all 1,319 examples, and both ran with decode CUDA graphs active. Andy's six runs on `aa8c950a3d` spanned 96.82% to 97.35%, so a single pass anywhere in that band is indistinguishable from the reference.

These two rows describe exactly one configuration: branch `b7dda1a7ba` on image `v0.5.18-rocm720-mi35x` with AITER `d9e5ef7c`, **on a server that has only ever seen short prompts**. That last clause is the load-bearing one. The same configuration scores 20.92% after four concurrent thousand-token prefills, so these numbers describe a workload as much as they describe a stack. Re-grade before quoting a number from any other tree, and check first that nothing in your harness issues concurrent long prefills before the evaluator runs.

## If you get garbage instead

Check these in order before touching any kernel. Every one of them cost me time today.

| Check | Command | Expected |
| --- | --- | --- |
| **nothing sent concurrent long prompts first** | grep the server log for prefills before the evaluator's; check your harness for a `bench_serving` warmup | the server saw only short prompts |
| which sglang is imported | `python3 -c 'import inspect,sglang;print(inspect.getfile(sglang))'` | under `$SGLANG_SRC/python` |
| tree is clean | `git status --short` | empty |
| cache is fresh | `ls "$RUN_CACHE"` | did not exist before this server |
| sampling | evaluator args | `temperature 1.0`, `top_p 0.95`, `--thinking` |
| AITER revision | `git -C /sgl-workspace/aiter rev-parse HEAD` | `d9e5ef7c...` for the numbers on this page |

Start at the top of that table, not the bottom. A high truncated rate with ~80% of the *stopped* answers still correct is the concurrent-prefill bug, and no amount of AITER swapping will fix it — restart the server and grade it before it serves anything long. A wrong AITER revision reproduces instead as the HSA abort above. A shared or stale cache reproduces as run-to-run nondeterminism. None of the three is a kernel bug in the PR stack.

One more trap: this model is nondeterministic even in the reference configuration. Identical greedy requests return different logprobs, and the known-good tree behaves the same way — two identical `max_new_tokens=1` requests differed by 0.41 in top-5 logprobs on the reference. Nondeterminism on its own is not evidence of a defect here, and the only reliable signal is output quality measured by the evaluator.
