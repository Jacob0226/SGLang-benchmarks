# GLM-5.3-Flash MXFP4 TP4 — where MI355X decode loses to B200, and what to do about it

Scope: **i8k / conc4 decode only**, and **no multi-stream** (no `alt_stream`, no
dual-stream fork). Everything below is one decode forward.

Measurement basis: the 5-step captures of 2026-09-23,
`results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/prof-Fixed-MXFP4-TP4-PRstack-steps5-node172`
against
`results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-NVFP4-TP4-steps5`,
analysed with `trace_analysis/run_glm53_sglang_vs_sglang.sh`. Both sides use a
clean forward — the B200 2-step capture had a spin-waiting collective in *both*
of its forwards, which is why an earlier reading of this comparison put B200's
all-reduce at 5.4x MI355X's. It is not; at parity it is 0.60 vs 0.42 ms.

Two checkpoints, not one model twice: MI355X serves `amd/GLM-5.3-Flash-Quark-MXFP4`
(Quark MXFP4, attention/mHC/MTP-experts left block-FP8) with bf16 KV and TileLang
DSA; B200 serves `RadixArk/GLM-5.3-Flash-NVFP4` (ModelOpt W4A4) with fp8 KV and
trtllm DSA. Read the buckets as "where does each platform's time go".

## The gap, and how it splits

| | MI355X | B200 |
|---|---|---|
| kernel launches per forward | **1745** (1728 kernel + 17 memcpy) | **1613** (8066 over 5 forwards) |
| Σ kernel | 9.77 ms | 7.61 ms |
| GPU-busy (union) | 9.99 ms | 6.00 ms |
| wall per forward | **10.00 ms** | **6.15 ms** |
| overlap factor | **1.00x** (2 streams) | 1.29x (47 streams) |
| measured median ITL | 9.00 ms | 6.08 ms |

The 3.85 ms wall gap decomposes almost exactly:

* **2.16 ms — kernels are slower** (Σ 9.77 vs 7.61)
* **1.61 ms — nothing overlaps** (B200 compresses 7.61 ms of kernel into 6.00 ms
  of wall; MI355X saves 0.03 ms)

The second half is out of scope here by construction. This document is about the
2.16 ms, and about launch count, which is the other thing 1745 launches over 45
layers (38.8 per layer) buys you.

Per bucket, Σ ms, MI355X minus B200:

| bucket | MI355X | B200 | Δ |
|---|---|---|---|
| MoE routing/sort | 1.411 | 0.623 | **+0.79** |
| shared-expert / dense MLP | 1.207 | 0.700 | **+0.51** |
| DSA indexer + k-pool | 1.070 | 0.637 | **+0.43** |
| MLA projections | 0.501 | 0.167 | +0.33 |
| sparse-MLA attn | 0.480 | 0.277 | +0.20 |
| all-reduce/comm | 0.603 | 0.422 | +0.18 |
| MoE up/gate GEMM | 0.648 | 0.610 | +0.04 |
| mHC (4-stream residual) | 1.148 | 1.297 | −0.15 |
| KDA projections | 1.378 | 1.551 | −0.17 |
| KDA linear-attn core | 0.544 | 0.635 | −0.09 |
| MoE down GEMM | 0.340 | 0.448 | −0.11 |

MI355X is **faster** at the KDA half of the model and at mHC. It loses in the
MoE plumbing, the DSA indexer, and the MLA projections.

---

## Ranked opportunities

### 1. Shared-experts fusion — already measured, just not enabled

`glm5_next.py` still refuses it with *"Shared experts fusion currently requires
CUDA devices"*. The reverted #36607 widened that gate to
`_is_cuda or _use_aiter_gfx95`, and the Day-0 re-split never brought it back.

The shared expert currently runs as its own four-kernel chain, 0.775 ms of the
1.207 ms `shared-expert/dense MLP` bucket:

| kernel | Σ ms | n |
|---|---|---|
| `_gemm_afp4wfp4_kernel ... MergedColumnParallelLinear` | 0.208 | 42 |
| `_gemm_afp4wfp4_reduce_kernel ... ACTUAL_KSPLIT_4` | 0.191 | 42 |
| `_fused_clamp_silu_mul_kernel` | 0.189 | 42 |
| `_gemm_afp4wfp4_kernel ... RowParallelLinear` | 0.187 | 42 |

Note the **separate split-K reduce kernel**: at M=4 the gate-up GEMM is split 4
ways and needs a second kernel to reduce it. Fusing the shared expert into the
routed MoE deletes all four, 168 launches per forward.

Already A/B'd on the 0914 image, MXFP4 TP4, same tree and JIT cache, only
`--disable-shared-experts-fusion` differing: **TPOT −3.4% to −8.4%, output
throughput +3.7% to +8.5% across four cells, GSM8K 96.89% vs 97.04%** (inside
this checkpoint's run-to-run band). The checkpoint is safe for it — its shared
expert carries the same `fp4/per_group` scheme and the same per-expert shapes as
a routed one and is not in the quark `exclude` list
(`tools/glm53_probe_shared_experts.sh`).

Branch: `jacob/glm53-stack-0921-sharedfusion`, commit `c74aad2f11`, 3 lines.
Precedent upstream: #38878 loads fused shared experts for Qwen4-Exp / Qwen3.5 MTP
on AMD, and #36124 is the quark shared-experts gate recognising a trailing MTP
layer — the same machinery this needs.

**Effort: done. Highest confidence of anything here.**

### 2. MoE routing and sorting at M=4 — the biggest single deficit

0.79 ms behind B200, and it is not the expert GEMMs (those are at parity):

| kernel | Σ ms | n | what it is |
|---|---|---|---|
| `aiter::opus_moe_sorting_entry<MoeSortingKernel...>` | 0.464 | 42 | build expert batches |
| `_gemm_a16_w16_kernel ... MoEGate` | 0.353 | 42 | router GEMM, **BF16, unquantized** |
| `aiter::grouped_topk_kernel` | 0.247 | 42 | top-8 of 288 |
| `fused_mx_quant_moe_sort` ×2 | 0.346 | 84 | quant + sort |
| `vectorized_elementwise_kernel<8, CUDAFunctor_add>` | 0.202 | 42 | the residual add |

Two things stand out.

**The router GEMM is 8.4 µs per layer for a 4×4096 @ 4096×288 product.** That is
4.7 MFLOP in 8.4 µs — about 0.6 GFLOP/s. It is entirely launch and tail, and it
runs in BF16 (`a16_w16`) while everything around it is quantized. A skinny-GEMM
kernel of the kind already used elsewhere in this same trace
(`aiter::wv_splitk_small_fp16_bf16_kernel`) should do it in 1-2 µs.

**The sorting machinery costs 0.81 ms to place 32 (token, expert) pairs.** At
bs=4 with topk=8 there are 32 assignments across 288 experts; three kernels and
126 launches per forward is the full large-M path being run at M=4. A small-M
specialisation, or fusing gate → topk → sort, is the target.

Upstream has been here: **#38328 "[MoE][ROCm] Admit the unified Triton router on
ROCm, including single-group routing"**. GLM-5.3-Flash is exactly single-group
(`n_group=1`, `topk_group=1`, `scoring_func=sigmoid`, `noaux_tc`). But in the
current tree `topk.py` dispatches ROCm+AITER to `aiter_biased_grouped_topk` and
gates the unified router behind `elif _is_cuda:` — so this model does **not** take
it. Worth measuring `moe_fused_gate` against the aiter pair on gfx950.

**Effort: medium (a kernel or a dispatch change). Upside: up to ~0.8 ms of 10.**

### 3. The DSA indexer's glue — 20 kernels per layer, half of them framework noise

The indexer runs on 11 of 45 layers and costs 1.070 ms, but it does so in about
**20 distinct kernels per layer, ~220 launches per forward**, most of them 4 µs:

| kernel | Σ ms | n | |
|---|---|---|---|
| `kpool_topk_transform_kernel<512>` | 0.128 | 11 | real work |
| `_gluon_deepgemm_fp8_paged_mqa_logits_preshuffle` | 0.046 | 11 | real work |
| `fast_hadamard_transform_kernel` | 0.047 | 11 | real work |
| `act_quant_kernel` | 0.057 | 11 | real work |
| `CatArrayBatchedCopy` + `_contig` | 0.121 | 22 | **glue** |
| `bfloat16tofloat32` + `bfloat16_copy` + `vectorized_layer_norm` | 0.133 | 33 | **glue** — one LayerNorm in three kernels |
| `index_elementwise` / `elementwise_kernel_with_index` / `FillFunctor` / `BUnaryFunctor` | 0.206 | 44 | **glue** |
| `triton_poi_fused__to_copy_gemm_a16w16`, `triton_poi_fused_mul_unsqueeze_1` | 0.092 | 22 | torch.compile leftovers |

Roughly **0.55 ms of the 1.07 ms is copies, casts and fills**, not arithmetic.
This is the clearest "Day-0 got it working" surface in the whole forward, and it
is where an aiter fused kernel would pay: the model's own
`dsa-kpool-indexer.md` lists `_kpool_*` kernels that already exist as fused ops,
so the pattern is established.

One thing to check before writing anything: GLM.sh sets
`SGLANG_DSA_FUSE_HADAMARD_QUANT=1` (PR #30715, hadamard + fp8 quant fused into
one Triton kernel, gated on gfx950 and `head_dim == block_size == 128`), yet the
trace still shows `fast_hadamard_transform_kernel` and `act_quant_kernel` as
separate launches. Either the gate is not firing on this model or they are
different call sites. **Verify first — it may be a free win already paid for.**

**Effort: medium-high (aiter kernel work). Upside: ~0.3-0.5 ms.**

### 4. Activation quantisation as its own kernel — 84 launches

`_dynamic_mxfp4_quant_kernel`, 0.354 ms over 84 launches, quantises activations
to MXFP4 twice per MoE layer. The fused pattern already exists three rows above
it in the same trace: `_fused_clamp_silu_mul_kernel_..._QUANT_BLOCK_SIZE_128_SCALE_FMT_...`
folds the quant into the activation. Folding the other two quants into their
producers (the mHC post kernel, and the MoE down-proj epilogue) removes 84
launches.

This is the same class of win as #30519 (fp8 MLA absorbed bmm) — stop
materialising a BF16 tensor only to requantise it.

**Effort: medium. Upside: ~0.2-0.3 ms.**

### 5. MLA projections — 3.0x B200, and they are the only unquantized GEMMs left

0.501 vs 0.167 ms. The MLA q/kv projections on the 11 DSA layers run through
`ck::kernel_gemm_xdl_cshuffle_v3_multi_d_blockscale_b_preshuffle` (0.285 ms,
n=25) and Tensile `Cijk_*` kernels. `aiter::wv_splitk_small_fp16_bf16_kernel`
(0.491 ms, n=113) is **BF16** and is shared between the indexer's
`ReplicatedLinear` and KDA's projections.

PR #30519 did exactly this for GLM-5.2 (fp8 MLA absorbed bmm on gfx950), and
AiterPR4453 tuned `batched_gemm_a8w8` per-token-group for large M on the same
path. Neither is in effect here: this checkpoint keeps attention in block-FP8 by
`layer_quant_config`, and the absorbed bmm is not taking an fp8 path.

**Effort: medium, with a known template. Upside: ~0.2-0.3 ms.**

### 6. mHC — already fused at this concurrency, nothing to do at conc4

Four kernels per layer, 1.148 ms, 225 launches — and MI355X is already 0.15 ms
**ahead** of B200 here. The cross-layer boundary fusion (#39200's
`hc_ffn_post_pre`) **is active** at conc4: `mhc_fused_post_pre_gemm_sqrsum_kernel`
appears with n=45. It is capped at `_MHC_FUSED_BOUNDARY_MAX_TOKENS = 16`, so
conc64 does not take it — the comment there says the fusion wins to 16 tokens
and reaches parity at 24, with 17-23 unmeasured. Sweeping that cap is a conc64
question, not a conc4 one.

### 7. Not applicable at i8k: the indexer skip

PR #31324 skips the DSA decode indexer when `kv_len <= index_topk`. This model's
`index_topk` is 2048 and i8k decodes at kv_len ≈ 8192, so the skip cannot fire.
It is worth 1.07 ms at **i1k**, where GLM.sh already turns on
`SGLANG_DSA_DECODE_DUAL_GRAPH=1` to capture both graphs.

---

## Does fp8 KV instead of bf16 make decode faster?

Short answer: **a little, and not where you would expect. It is mainly a memory
win, and it is blocked by a one-line bug.**

What it changes, at i8k/conc4 decode:

* **Bytes moved.** DSA attention reads `index_topk`=2048 (+ up to 3 tail) entries
  × 576 dims on 11 of 45 layers. At bf16 that is ~2.36 MB per request per layer,
  ~104 MB per forward at bs=4. Halving it saves on the order of **10-20 µs** of a
  9.77 ms forward. Not the lever.
* **The dot product.** This is the real one. The TileLang and Triton sparse-MLA
  kernels take an fp8 path when the KV is fp8 (`USE_FP8_DOT` in
  `triton_sparse_mla_decode.py`), and MFMA fp8 throughput is 2x bf16. `main_kernel`
  is 0.269 ms at conc4 and **1.175 ms at conc64**, so the ceiling is ~0.13 ms at
  conc4 and ~0.6 ms at conc64.
* **Capacity.** The KV pool is 102.28 GB for 8.64M tokens today; fp8 halves it.
  The 34 KDA layers' state pool (76.17 GB ssm + 2.68 GB conv) is untouched —
  concurrency on this model is usually limited by that pool, not KV.
* **Cost.** Chunked-prefill continuations must dequantise fp8 → bf16
  (`_get_mla_kv_buffer_from_fp8_for_dsa`), so i70k TTFT can get worse.

Net expectation: **~1-2% at conc4 decode, ~4% at conc64 decode, a possible
prefill regression at i70k, and ~51 GB back.** Worth doing for conc64 and for
memory; do not expect it to move conc4.

**It is blocked today.** `forward_mha.py:641-644` unwraps `TboAttnBackend` but
not the `HybridLinearAttnBackend` this model actually uses, so every rank dies
with `AttributeError: 'HybridLinearAttnBackend' object has no attribute
'forward_metadata'` the first time a chunked-prefill continuation reads prefix
KV — which the i70k shape and high-concurrency GSM8K both do. The fix is one
line, `backend = getattr(backend, "full_attn_backend", backend)`, and
`HybridLinearAttnBackend` does expose `full_attn_backend`
(`hybrid_linear_attn_backend.py:1170`).

Precedent: the DSv4 AMD work already runs fp8 KV on gfx950 — #37413 (fp8 two-pool
unified_kv) and #38901 (DSpark with fp8 unified_kv).

---

## What the DSv4 AMD work suggests that is not on the list above

* **#39968 "pick kv_splits per index stream, not by occupancy alone"** — decode
  attention split-K chosen badly at small batch is exactly the shape of the
  `_gemm_afp4wfp4_reduce_kernel ... ACTUAL_KSPLIT_4` row in §1 and the
  `Cijk_..._PostGSU16/PostGSU8` rows in §3. Worth auditing split-K choices at M=4
  across this forward, not just in attention.
* **#39116 "reduce host bubble on DSV4"** — host-side bubbles do not show in
  Σ kernel but do show in wall. MI355X's decode wall (10.00) already tracks its
  union (9.99), so there is no bubble to reclaim here; noted so nobody re-measures
  it.
* **#37810 breakable CUDA graph prefill** — prefill only.

## Suggested order

1. **Shared-experts fusion.** Measured, 3 lines, −3.4 to −8.4% TPOT.
2. **Verify `SGLANG_DSA_FUSE_HADAMARD_QUANT` is firing.** Possibly free.
3. **Router: measure `moe_fused_gate` vs `aiter_biased_grouped_topk` on gfx950**,
   and replace the BF16 `a16_w16` router GEMM with a skinny-GEMM path.
4. **fp8 KV** with the one-line hybrid-backend fix — measure conc64 and i70k TTFT.
5. **Indexer glue**: fold the copies/casts/fills into fused kernels.
6. **MoE sorting small-M specialisation.**

Items 1-4 are configuration, verification or small patches. 5 and 6 are kernel
work.

## Caveats attached to every number here

* One forward per side. MI355X gives exactly one usable decode forward whatever
  `PROF_NUM_STEPS` says (HIP graph replay; see the comment in `GLM.sh`), so there
  is no within-capture spread on that side. B200's five agree to ±0.02 ms.
* Σ kernel counts streams separately. It equals wall on MI355X (1.00x) and
  overstates B200 by 1.29x. The per-bucket comparison is therefore slightly
  unfair *to MI355X* in the other direction: B200's real per-bucket wall
  contribution is smaller still.
* The `all-reduce/comm` row is transport only after the spin-waiting launches are
  excluded. Re-check with `tools/glm53_allreduce_histogram.sh` on any new capture.
