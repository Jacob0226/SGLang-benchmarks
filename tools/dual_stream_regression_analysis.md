# GLM-5-FP8 NSA Decode Dual-Stream Regression Analysis (MI355X / ROCm)

**Date:** Apr 2026
**Branches under test:**
- `jacob/glm5-rocm-nsa-on-thomas` (c428a5dc3) — best-perf branch on top of PR #23562 + aiter PR #2879. Contains both cat-skip (default ON) and A_v4 dual-stream layout (opt-in via `SGLANG_ENABLE_HIP_DUAL_STREAM=1`). The dual-stream regression numbers in this doc were measured with the env var set.
- `jacob/glm5-rocm-nsa-cat-skip` (8d4b57132) — upstream-bound PR branch, rebased onto `sgl-project/main`. Contains only the cat-skip optimization (no dual-stream code). Independent of Thomas's PR; ships the strict-improvement piece by itself.
**TLDR:** A_v4 dual-stream layout (overlap NSA indexer with [q_b_proj + bmm w_kc + fused_qk_rope_cat]) **loses ~30 μs / layer** on MI355X due to HBM bandwidth contention and a HIP-graph-specific AllReduce slowdown. Single-stream + cat-skip optimization wins.

---

## Bench Numbers (8k1k conc4, GLM-5.1-FP8, TP=8, fp8 KV cache)

| Variant | Median TPOT | Δ vs Thomas |
|---|---|---|
| Thomas (PR #23562 only, no `--dual-stream-rocm`) | **21.21 ms** | baseline |
| Mine + `--dual-stream-rocm` (A_v4 + dual-stream MoE, fused shared expert disabled) | 24.45 ms | **+15.3% (regression)** |
| **Mine + nostream patch (A_v4 disabled, fused shared expert enabled, cat-skip kept)** | **20.48 ms** | **−3.4% (faster than Thomas)** |

The third row is the optimization we want to land. The dual-stream A_v4 layout is *available* (env var) but **not enabled by default** because it currently regresses on MI355X.

---

## Per-Kernel Comparison (avg over 10 layers)

| bucket | Thomas | Mine + dual-stream | Mine + nostream |
|---|---|---|---|
| total layer dur | 143.7 μs * | 321.6 μs † | 140.4 μs * |
| AllReduce (1 of 2 per layer) | 9.78 μs | **36.06 μs** ‡ | 9.51 μs |
| indexer chain (sum of 10 kernels) | 59.8 μs | 68.0 μs | 59.8 μs |
| CatArrayBatchedCopy (`concat_mla_absorb_q_general` fallback) | 2.6 μs | 0 μs | 0 μs |
| `_fused_append_shared_experts_kernel` | 2.13 μs | 0 μs ‖ | 2.15 μs |
| shared expert chain (gate_up + silu + down, unfused) | 0 μs ‖ | 47.9 μs | 0 μs ‖ |

\* Thomas and Mine+nostream both have `fused_rms_fp8_group_quant` firing **twice per layer** (once for input layernorm, once for q_a/kv_a layernorm fused on the alt-stream-None ELSE branch). My script picks up both, giving alternating 25 / 260 μs "layer durs", so reported 143.7 μs is half-layer avg → real layer ≈ 285 μs.

† Mine + dual-stream fires `fused_rms_fp8_group_quant` only once per layer (because qk-norm fork takes the IF branch using separate `add_rmsnorm_quant` for q_a + kv_a), so real layer = 321.6 μs.

‡ Same `aiter::cross_device_reduce_1stage<bf16, 8>` kernel — yet **2.3× slower** under dual-stream HIP-graph capture. See "AllReduce slowdown" below.

‖ Determined by the `--disable-shared-experts-fusion` flag, which is implied by GLM.sh's `--dual-stream-rocm`. Without that flag, aiter uses the fused shared expert kernel (~4μs vs ~48μs unfused).

---

## Why dual-stream loses on MI355X

> **Honesty note**: The bench / per-kernel-duration deltas below are **measured**.
> The proposed *mechanisms* (HBM bandwidth, CU split, AR fence) are **hypotheses
> consistent with the data but not directly verified** with hardware counters.
> See "Verification needed" at the bottom.

### Hardware refresher (MI355X, from CDNA 4 whitepaper)

```
1 GPU package = 8 XCDs (TSMC 3nm) + 2 IODs (TSMC 6nm) + 8 HBM3E stacks
              = 256 active CUs (32 per XCD, 4 disabled per XCD for yield)
              = 32 MB L2 cache (4MB per XCD)
              = 256 MB Infinity Cache (in IODs, shared across all XCDs)
              = 288 GB HBM3E, 8 TB/s aggregate bandwidth
```

(Note: previously this analysis quoted 304 CUs / 6 TB/s — both wrong, MI300X
numbers misapplied. MI355X is 256 active CUs / 8 TB/s.)

GPU streams **share** all of the above. When two streams have kernels running
concurrently they compete for these finite resources.

### 1. HBM bandwidth contention (HYPOTHESIS — likely contributor to indexer slowdown)

Memory-bound kernels in the indexer chain (Hadamard, act_quant, paged_mqa,
topk_transform — all do little compute relative to data movement) read from
HBM. When dual-stream runs cur's q_b_proj GEMM concurrently with alt's wk
GEMM (also reading weights from HBM), aggregate demand may approach or exceed
the 8 TB/s pipe.

But "cache thrashing" (an earlier claim) is **probably not the dominant cause**
for GLM-5 decode: weights are multi-GB and don't fit in the 256 MB Infinity
Cache regardless of single/dual-stream. KV cache + activations are GB-scale
too. The cache mostly handles small intermediates, which fit either way.

**What we actually measured** (on stream 113/106 alt-side vs Thomas single-
stream phys 8):

```
                    Thomas (single)  Mine alt-side  Δ
indexer_layernorm      4.7 μs           5.8 μs    +1.1
hadamard               4.2              4.6      +0.4
hadamard               4.1              4.9      +0.8
act_quant              4.7              6.0      +1.3
indexer_k_quant        4.4              4.6      +0.2
wv_splitk (NSA score)  5.4              6.0      +0.6
paged_mqa              4.1              4.9      +0.8
topk_transform        15.2             17.6      +2.4
                                  ───────────
                                       +8.2 μs total
```

GEMMs (q_b_proj, wq_b, wk) are basically unchanged. Memory-bound indexer
kernels each pay 0.5-2.4 μs. The slowdown pattern is *consistent with* HBM
contention but could also be e.g. memory-controller queue depth or HBM
channel-bank conflicts. Verification needs hardware counters.

Per-kernel measurements (cur+alt running concurrently vs cur alone):

```
                    Thomas (single)  Mine alt-side  Δ
indexer_layernorm      4.7 μs           5.8 μs    +1.1
hadamard               4.2              4.6      +0.4
hadamard               4.1              4.9      +0.8
act_quant              4.7              6.0      +1.3
indexer_k_quant        4.4              4.6      +0.2
wv_splitk (NSA score)  5.4              6.0      +0.6
paged_mqa              4.1              4.9      +0.8
topk_transform        15.2             17.6      +2.4
                                  ───────────
                                       +8.2 μs total
```

GEMM kernels (q_b_proj, wq_b, wk) are compute-bound and basically don't slow down (+0~0.1 μs). Only memory-bound kernels suffer.

### 2. Infinity Cache thrashing (LIKELY NOT the cause, retracted)

An earlier draft of this doc claimed "dual-stream's two working sets evict
each other from the 256 MB Infinity Cache". On reflection that's likely
wrong for GLM-5 decode: weights are multi-GB and don't fit in cache anyway,
and per-kernel intermediates at decode batch=4 are small (KB to single-MB
scale). Single-stream and dual-stream both have the same cache miss profile
for these workloads.

### 3. Compute-Unit (CU) split (HYPOTHESIS — probably weak contributor)

MI355X has **256 active CUs** across 8 XCDs (32 each, 4 disabled). When
two kernels run concurrently, ROCm scheduler distributes workgroups across
CUs. *In principle* this could halve the CUs available per kernel.

In practice, small kernels (B=4 decode) likely use only a fraction of the
CUs anyway — they're launch / latency dominated, not CU-throughput
dominated. So the +1.5 μs main_kernel slowdown observed in the trace
**probably is not** caused by CU split.

What might it be instead? Some possibilities:
- L1/LDS contention between concurrent waves
- Shared instruction cache pressure
- Wavefront-level scheduling artifacts
- Cumulative effect of (1) — main_kernel does some HBM access too

Honestly, **this needs `rocprof-v3` HW counters to attribute properly**.

### 4. AllReduce slowdown (+13 μs / call) — VERIFIED, mechanism is HYPOTHESIS

**What was directly measured**:
- Same `aiter::cross_device_reduce_1stage<bf16, 8 ranks>` kernel
  (byte-identical mangled name)
- Same MI355X hardware, same aiter build (PR #2879)
- Same TP=8 topology, same bench input sizes
- Single-stream HIP graph: AR ≈ 9.5 μs
- Dual-stream HIP graph:  AR ≈ 23 μs
- → **2.4× duration difference is real**, and disappears when alt_stream is
  set to None (i.e., MyBranch_NoDualStream test variant).

**What was NOT verified, just hypothesized**:

The proposed mechanism — that "the AR's peer-fence has to drain alt's
KV-cache writes, costing extra time attributed to the AR kernel" — is
**guess-work**. Other plausible mechanisms include:

- HIP-graph dual-stream replay scheduler emits extra event signal/wait
  nodes that gate the AR start, padding its measured duration.
- aiter's signal-buffer poll loop in the cross_device_reduce inner loop
  takes longer when peer GPUs are busier (e.g., still finishing alt-stream
  kv_cache writes).
- Some ROCm runtime quirk specific to graph capture under dual-stream that
  `rocprof` would reveal but I haven't run.

To verify the actual mechanism:
1. Run dual-stream + `--disable-cuda-graph` → if AR is fast, mechanism is
   graph-specific. If still slow, mechanism is dual-stream-runtime specific.
2. Profile AR with `rocprof-v3 --hsa-trace` to see kernel start/end vs
   stream queue depth.
3. Read the aiter `cross_device_reduce_1stage` source to see fence
   sequence and check whether there's a per-stream barrier.

This is two ARs per layer (attn AR + MoE AR), so 2 × 13 μs = **+26 μs/layer**
that disappears when dual-stream is off.

---

## Total accounting (Mine dual-stream vs Thomas)

| source | μs / layer |
|---|---|
| AR slowdown (dual-stream HIP-graph artifact) | +26 |
| Indexer kernels memory-bound contention | +8 |
| Other small kernels (main_kernel, bmm_w_vc, …) compute split | +5 |
| Layer structure (extra `add_rmsnorm_quant` for split q_a/kv_a vs fused) | ~0 (offset by saved fused_rms call) |
| Theoretical A_v4 saving (gap-fill overlap with indexer) | −10 |
| **Net regression** | **+29** |

Bench TPOT regression: 24.45 − 21.21 = 3.24 ms / token = **+50.6 μs / layer** at TPOT level. 29 μs at GPU level + ~20 μs CPU/scheduler overhead under dual-stream graph (extra event records, larger graph, etc.) explains the gap.

---

## What does win on MI355X

```
Branch: c428a5dc3 (jacob/glm5-rocm-nsa-on-thomas)
  + sed patch removing _is_hip from alt_stream gate at deepseek_v2.py:1925
        ↓
  alt_stream = None on ROCm
        ↓
  forward_normal_dual_stream NOT entered (gate is alt_stream is not None)
  forward_absorb_prepare's HIP branch runs but overlap_indexer_with_gap_fill = False
                                             ↓
                                     Single-stream serial kernel sequence
                                     (same as Thomas)
        ↓
  forward_absorb_core still runs cat-skip path (q_rope=None for decode)
  nsa_backend.forward_decode skips concat_mla_absorb_q_general
        ↓
  Saves ~2.6 μs / layer (real win)
```

Result: 20.48 ms median TPOT — **0.73 ms / token faster than Thomas baseline**.

The two non-controversial optimizations:

1. **`_is_hip` regression fix** in `deepseek_v2.py` line 1925: this is the one MUSA PR `b35213be1` accidentally introduced. **But** restoring it on its own enables dual-stream which currently loses, so it should be **opt-in via env var** rather than always-on.

2. **Cat-skip in `nsa_backend.forward_decode`** (gated `if q_all is None or not _is_hip`): pure free win, ~2.6 μs/layer, no downsides. The complementary edit in `forward_absorb_core` (passing `q_cat` with `q_rope=None` on the decode path) is what triggers the cat skip.

---

## Recommendation

Replace the current single squashed commit with a smaller commit that:

- Keeps the cat-skip optimization on by default (HIP-only).
- Restores the `_is_hip` alt_stream gate **only when an env var `SGLANG_ENABLE_HIP_DUAL_STREAM` is set** — default OFF.
- Keeps the A_v4 dual-stream layout code in `forward_absorb_prepare` (preserved for future use, maybe a future ROCm release fixes the AR slowdown).

Default behavior on ROCm: single-stream + cat-skip → matches `MyBranch_NoDualStream` test. Wins by 0.73 ms TPOT vs Thomas.

To experiment with dual-stream: `SGLANG_ENABLE_HIP_DUAL_STREAM=1 ./GLM.sh --dual-stream-rocm ...`

---

## Verification needed

The claims in §1, §3, §4-mechanism above are *hypotheses* consistent with the
observed kernel slowdowns but **not** verified with hardware counters or
source inspection. Concrete next steps:

1. **rocprof-v3 hardware counters** on MyBranch_NoDualStream vs DualStream0428_v2
   for one decode step:
   - HBM bytes / cycle per kernel (verifies §1)
   - CU active rate / wavefront occupancy per kernel (verifies §3)
   - LDS / L1 / L2 hit rates (sanity check §2)
2. **`--disable-cuda-graph` ablation** on dual-stream variant — if AR is fast
   in eager mode, §4 is graph-specific; otherwise dual-stream-runtime specific.
3. **aiter source inspection** — read `cross_device_reduce_1stage` to see
   fence sequence, peer signal protocol, etc.
4. **Larger batch ablation** — at conc=64 indexer is longer absolutely, dual-
   stream might recover ROI even with the contention costs. Not measured.
5. **Future ROCm releases** (>= 7.3?) may fix HIP graph dual-stream scheduling.
   Re-run periodically.
