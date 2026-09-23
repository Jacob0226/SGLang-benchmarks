#!/usr/bin/env bash
# Usage:
# ./GLM.sh
# ./GLM.sh --mtp --prof
# ./GLM.sh --prof-combined           # profile without splitting prefill/decode
# ./GLM.sh --dual-stream-rocm        # disable shared-experts-fusion for dual stream on ROCm
# ./GLM.sh --model /data/huggingface/hub/zai-org/GLM-5-FP8
# ./GLM.sh --prof --dual-stream-rocm --tag DualStream
# ./GLM.sh --tp 4 --tag 0507_TP4    # tensor parallel size (auto: TP=4 for FP4 models, TP=8 for FP8)
# ./GLM.sh --docker rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412   # tag results dir with docker image
# ./GLM.sh --port 8600              # change server port (default 8552; also: PORT=8600 ./GLM.sh)
# ALLOW_LOCKED_CLOCKS=1 ./GLM.sh    # skip the GPU application-clock check (see check_gpu_clocks)
#
# GLM-5 / GLM-5.1 FP4 examples (auto-detects quant scheme from model name;
# mirrors InferenceX recipes — see SemiAnalysisAI/InferenceX benchmarks/
# single_node/glm5.1_fp4_mi355x.sh and glm5_fp4_b200.sh):
# ./GLM.sh --model amd/GLM-5.1-MXFP4               # MI355X (MXFP4 self-declares)
# ./GLM.sh --model nvidia/GLM-5-NVFP4              # B200   (NV official ModelOpt quant)
#
# GLM-5.3-Flash on ROCm/gfx950 (sgl-project/sglang#36607 recipe: AITER MoE +
# TileLang DSA + gfx950 mHC fast paths; needs SGLANG_USE_AITER=1, set automatically):
# ./GLM.sh --model /data/huggingface/hub/zai-org/GLM-5.3-Flash --tp 4 --tag TP4
#
# GLM-5.3-Flash NVFP4 on Blackwell (RadixArk/GLM-5.3-Flash-NVFP4, W4A4 ModelOpt).
# Everything resolves from the model name: TP4, modelopt_fp4, trtllm DSA, fp8_e4m3
# KV, flashinfer_trtllm MoE, and the decode cuda-graph cap this checkpoint needs:
# ./GLM.sh --model /data/huggingface/hub/RadixArk/GLM-5.3-Flash-NVFP4 --tp 4 --tag NVFP4-TP4
#
# ===================== Known issues behind the per-model flags =====================
# The per-model blocks further down stay short on purpose. Anything in them that
# looks arbitrary is explained here once.
#
# [radix] --disable-radix-cache, every model
#   Not cosmetic. bench_serving reuses one seed for every cell, so the prefix tree
#   carries prompts over and the sweep grades a warm cache from the second cell on.
#   Measured 2026-09-15, GLM-5.3-Flash TP4/B200, 1024-token input (too short for
#   chunked prefill to explain it): conc4 4.5% of prefill tokens cached (it runs
#   first, on an empty tree), then 47.7 / 47.8 / 46.4 / 45.5% at conc 8/16/32/64.
#   Half the prefill disappears and TTFT with it. DISABLE_RADIX_CACHE=0 to measure
#   the cache deliberately.
#
# [rocm-kv] GLM-5.3-Flash on ROCm pairs TileLang DSA with BF16 KV, not FP8
#   FP8 KV with TileLang DSA is a supported ROCm path on paper -- forward_mha_rocm.py
#   branches on exactly "fp8_e4m3 and DSA backend is not trtllm" -- but that branch
#   only runs when a prefill reads already-written prefix KV, i.e. a chunked-prefill
#   continuation. PR #36607 validated 1024-token inputs, which never chunk, so it
#   never entered the branch. This harness does: the i70k shape and high-concurrency
#   GSM8K both chunk, and then every rank dies at once:
#     forward_mha.py:500 _get_mla_kv_buffer_from_fp8_for_dsa
#     -> backend.forward_metadata.page_table_1_flattened
#     AttributeError: 'HybridLinearAttnBackend' has no attribute 'forward_metadata'
#   The helper unwraps TboAttnBackend but not the hybrid linear-attention wrapper
#   that owns the DSA backend. One-line upstream fix (backend = getattr(backend,
#   "full_attn_backend", backend)) is staged in tools/glm53_flash_setup_container.sh
#   as APPLY_HYBRID_PATCH. Until it lands, BF16 is the only KV dtype that survives.
#   The Blackwell side picks trtllm DSA, which makes the bug unreachable, so FP8 KV
#   is safe there -- the DSA backend and the KV dtype are one decision, not two.
#
# [nvfp4-graph] GLM-5.3-Flash NVFP4 caps --cuda-graph-max-bs at 128
#   Without a cap the sweep produces nothing: 11 consecutive attempts on 2026-09-15
#   finished 0 of 15 cells (tools/glm53_retry_sweep.sh, ~/glm53_nvfp4_retry.log).
#   Where it dies is not deterministic, so the cap is not a diagnosed root-cause fix.
#   Of those 11: six died during decode cuda-graph capture at bs=208/224/248/432/
#   496/512 on different attempts with >20 GB free (so not memory pressure); four
#   captured every graph, served, then died in process_batch_result_decode; the TP8
#   run died in triton's clear_cache at bs=1. Identical arguments each time.
#   What the two configurations that do survive share is a small captured decode
#   batch: RadixArk's own command turns NEXTN on and so only captures verify/draft
#   graphs (bs<=48), and this cap does the same for the non-speculative baseline the
#   sweep measures. With it, two full sweeps completed 15/15 plus GSM8K.
#   Caveat, and it is a big one: every failure was on dgx-027 and every success on
#   dgx-024, so the cap and the machine changed together. dgx-027 also killed the
#   FP8 checkpoint at TP8 with the same IMA -- a configuration with no NVFP4, no
#   cutlass and no large decode graph -- while FP8 at TP4 passed there, which is
#   what one bad GPU out of eight would look like. Untested either way:
#   CUDA_GRAPH_MAX_BS=512 on a healthy node is the experiment that would settle it.
#
# [nvfp4-moe] GLM-5.3-Flash NVFP4 uses flashinfer_trtllm, not the card's cutlass
#   flashinfer_cutlass is why NVFP4 first measured slower than the FP8 weights.
#   Measured 2026-09-16, TP4/B200, same image, otherwise identical args, output
#   token throughput vs results/zai-org_GLM-5.3-Flash/.../bench-Fixed-TP4-0915-
#   cookbook-ht-noradix:
#     cutlass  i1k -12%..+5%   i8k -10%..-1%   i70k -5%..-0.3%
#     trtllm   i1k  +5%..+10%  i8k  +6%..+18%  i70k +16%..+23%
#   trtllm wins all 15 cells and beats cutlass by up to +30% at low concurrency,
#   where cutlass's grouped-GEMM overhead dominates. It has its own NVFP4 path
#   (fused_experts_none_to_flashinfer_trtllm_fp4), so this is a supported pairing
#   rather than a fallback, and it lets the NVFP4 and FP8 sweeps share a MoE runner
#   so their delta is the weights alone. GSM8K 97.50% on it, vs 97.14% on the card's
#   own configuration.
#
# [glm52-deadlock] GLM-5.2 deliberately skips the GLM-5/5.1 tuning block
#   --attention-backend nsa + --enable-flashinfer-allreduce-fusion + --stream-interval
#   30 + 32K chunking + --cuda-graph-max-bs together triggered a reproducible
#   vocab-sized TP all-gather deadlock on the DSA path. NVIDIA's own GLM-5.2-NVFP4
#   command sets none of them, and letting sglang auto-pick ran stably.
#
# [pcg] GLM-5.3-Flash leaves the prefill cuda-graph backend alone
#   None of the cookbook's verified GLM-5.3-Flash cells select one, and the one
#   command that mentions prefill graphs passes --disable-prefill-cuda-graph.
#   Forcing tc_piecewise onto the KDA/DSA hybrid would benchmark an unvalidated
#   path. ENABLE_PIECEWISE_CUDA_GRAPH=1 to measure it deliberately.
#
set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' 2>/dev/null || echo "[warn] cannot disable numa_balancing (need root); continuing"

MTP_ENABLED="false"
PROF_ENABLED="false"
PROF_COMBINED="false"   # if true: single combined trace (no --profile-by-stage)
DUAL_STREAM_ROCM="false"
MTP_TAG=""
USER_TAG=""
# TP_SIZE="auto" means: pick from MODEL_NAME after --model is parsed.
# FP4 models (MXFP4 / NVFP4) default to TP=4 so MI355X and B200 profiles
# are directly comparable at the same TP. FP8 / unrecognized fall back
# to TP=8. Override anytime with --tp.
TP_SIZE="auto"
MODEL_PATH="/data/huggingface/hub/zai-org/GLM-5-FP8"
# DOCKER labels the results directory so different docker images don't clobber
# each other. Override with --docker <image>. Known-good images:
#   rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412   # MI355
#   lmsysorg/sglang:v0.5.9-cu130-runtime              # B200
DOCKER="untagged-docker"
CURRENT_DIR=$(pwd)
while [[ $# -gt 0 ]]; do
  case $1 in
    --mtp)
        MTP_ENABLED="true"
        MTP_TAG="-MTP"
        shift 1
        ;;
    --prof)
        PROF_ENABLED="true"
        shift 1
        ;;
    --prof-combined)
        PROF_ENABLED="true"
        PROF_COMBINED="true"
        shift 1
        ;;
    --dual-stream-rocm)
        DUAL_STREAM_ROCM="true"
        shift 1
        ;;
    --model)
        MODEL_PATH="$2"
        shift 2
        ;;
    --tp)
        TP_SIZE="$2"
        shift 2
        ;;
    --tag)
        USER_TAG="-$2"
        shift 2
        ;;
    --docker)
        DOCKER="$2"
        shift 2
        ;;
    --port)
        PORT="$2"
        shift 2
        ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done
# Use the last two path components joined with '_' so the org is kept, e.g.
#   /data/huggingface/hub/amd/GLM-5.1-MXFP4    -> amd_GLM-5.1-MXFP4
#   /data/huggingface/hub/nvidia/GLM-5-NVFP4   -> nvidia_GLM-5-NVFP4
#   /data/huggingface/hub/zai-org/GLM-5.1-FP8  -> zai-org_GLM-5.1-FP8
_MODEL_PATH_TRIMMED="${MODEL_PATH%/}"
_MODEL_LEAF=$(basename "${_MODEL_PATH_TRIMMED}")
_MODEL_ORG=$(basename "$(dirname "${_MODEL_PATH_TRIMMED}")")
MODEL_NAME="${_MODEL_ORG}_${_MODEL_LEAF}"

# ===================== Quantization auto-detection (matches InferenceX) =====================
# Pick --quantization and --mem-fraction-static based on the model name.
# Mirrors SemiAnalysisAI/InferenceX recipes in benchmarks/single_node/
# {glm5.1_fp4_mi355x.sh, glm5_fp4_b200.sh, glm5_fp8_b200.sh}.
#
#   *MXFP4*  -> AMD MXFP4 (e.g. amd/GLM-5.1-MXFP4): model files self-declare
#               quant, so no --quantization flag is passed. AMD's Quark recipe
#               quantizes the shared experts to MXFP4 too (model card:
#               "MOE-only (shared experts quantized), OCP MXFP4"), so fusion
#               works (no --disable-shared-experts-fusion needed). Matches
#               InferenceX glm5.1_fp4_mi355x.sh.
#   *NVFP4*  -> NVIDIA NVFP4 (e.g. nvidia/GLM-5-NVFP4): needs
#               --quantization modelopt_fp4 and --mem-fraction-static 0.9
#               (matches InferenceX glm5_fp4_b200.sh's KV-pool budget; NV's
#               HF card uses 0.80). No --disable-shared-experts-fusion (NV's
#               HF launch command and InferenceX glm5_fp4_b200.sh both omit
#               it — sglang's modelopt_fp4 path handles it correctly).
#   *GLM-5.3* -> GLM-5.3-Flash (glm5_next). The checkpoint self-declares FP8 in
#               config.json's quantization_config (with a long
#               modules_to_not_convert list covering the MLA/MQA attention,
#               hyper-connection and mHC tensors), so passing --quantization fp8
#               would flatten that mixed-precision layout, so no --quantization.
#   *FP8*    -> default GLM-5-FP8: --quantization fp8 on B200, none on ROCm.
# Defined up here rather than beside the other helpers because the model-detection
# block below has to branch on the platform: GLM-5.3-Flash needs a different DSA
# backend and KV dtype on ROCm than on Blackwell, and that block runs at load time,
# long before start_server().
# FORCE_PLATFORM=rocm|cuda overrides the probe. Its reason for existing is that the
# two GLM-5.3-Flash recipes diverge in backend and KV dtype, and whoever is holding
# an MI355X cannot otherwise see what the Blackwell command would come out as. Pair
# it with DRY_RUN=1 (see start_server) to print the other platform's command without
# loading a model:
#   FORCE_PLATFORM=cuda DRY_RUN=1 ./GLM.sh --model .../GLM-5.3-Flash --tp 4
is_rocm_gpu_env() {
    case "${FORCE_PLATFORM:-}" in
        rocm) return 0 ;;
        cuda) return 1 ;;
    esac
    [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1
}

# ===================== Per-model server configuration =====================
# One block per (platform, checkpoint). Platform first, then the checkpoint named
# in full -- no name globs that quietly catch a second model, no fallthrough to a
# shared arm. Flags repeat between blocks on purpose: reading one block should tell
# you everything a model is served with, without tracing what it inherited.
#
# Each block sets exactly three things:
#   TP_SIZE             default TP, skipped when --tp was passed
#   MEM_FRACTION_STATIC --mem-fraction-static
#   MODEL_SERVER_ARGS   every other model- or platform-specific launch flag
# Flags common to all models (model path, host/port, parsers, watchdog, weight
# loader, --disable-radix-cache) live in start_server(); so do the ones driven by
# CLI flags rather than by the model (--mtp, --prof, --dual-stream-rocm).
#
# Env overrides stay inline as ${VAR:-default} so a one-off rerun never needs an
# edit here. Tags in [brackets] point at "Known issues" at the top of the file.
MODEL_SERVER_ARGS=()
MEM_FRACTION_STATIC="0.85"
# See the --disable-radix-cache block in start_server().  [radix]
_radix_default=1
# Set to "true" by --dual-stream-rocm (see start_server()) — needed so MoE.forward
# takes forward_normal_dual_stream instead of the fused shared-expert path.
NEED_DISABLE_SHARED_FUSION="false"

if is_rocm_gpu_env; then
    case "${MODEL_NAME}" in
        amd_GLM-5.1-MXFP4)
            # InferenceX glm5.1_fp4_mi355x.sh. MXFP4 self-declares in the
            # checkpoint, so no --quantization; shared experts are MXFP4 too, so
            # the fused shared-expert path is fine. TP=4 to line up with the B200
            # NVFP4 sweep (InferenceX's own MXFP4 sweep is TP=2; pass --tp 2).
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.85"
            MODEL_SERVER_ARGS=(
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-triton}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-triton}"
                --tokenizer-worker-num $((TP_SIZE * 2))
            )
            # gfx950 fused allreduce(+residual+rmsnorm), ~12us/layer on MI355X.
            # Validated on this recipe and GLM-5.2-MXFP4 only. Asserts against
            # --enable-prefill-cp, hence the escape hatch.
            [ "${DISABLE_AITER_ALLREDUCE_FUSION:-0}" = "1" ] || \
                MODEL_SERVER_ARGS+=(--enable-aiter-allreduce-fusion)
            ;;
        amd_GLM-5.2-MXFP4)
            # Same recipe as GLM-5.1-MXFP4 above; GLM-5.2's DSA changes nothing
            # that this arg set cares about on ROCm.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.85"
            MODEL_SERVER_ARGS=(
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-triton}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-triton}"
                --tokenizer-worker-num $((TP_SIZE * 2))
            )
            [ "${DISABLE_AITER_ALLREDUCE_FUSION:-0}" = "1" ] || \
                MODEL_SERVER_ARGS+=(--enable-aiter-allreduce-fusion)
            ;;
        zai-org_GLM-5.3-Flash)
            # sgl-project/sglang#36607's gfx950 recipe. Hybrid KDA + DSA MoE
            # (45 layers: 34 linear-attention, 11 DSA) with mHC. The checkpoint
            # self-declares FP8 block quant with attention/mHC left BF16, so
            # --quantization would override and break that layout. AITER MoE is
            # the PR's measured runner; SGLANG_USE_AITER is exported further down.
            # BF16 KV is forced by a TileLang DSA bug, not by preference. [rocm-kv]
            # --context-length: checkpoint declares 1,048,576, the i70k shape needs
            # 70,300, so pin something in between rather than size a 1M-token pool.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.85"
            MODEL_SERVER_ARGS=(
                --kv-cache-dtype "${KV_CACHE_DTYPE:-bfloat16}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-tilelang}"
                --moe-runner-backend "${MOE_RUNNER_BACKEND:-aiter}"
                --context-length "${CONTEXT_LENGTH:-131072}"
                --tokenizer-worker-num $((TP_SIZE * 2))
            )
            ;;
        amd_GLM-5.3-Flash-Quark-MXFP4)
            # Same gfx950 GLM-5.3-Flash recipe as the FP8 checkpoint above; only
            # the weights differ, and Quark self-declares them. Must be its own
            # block rather than falling into a *MXFP4* arm: the generic MXFP4
            # recipe leaves DSA at triton, which rejects index_kpool > 1 at decode
            # graph capture with NotImplementedError.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.85"
            MODEL_SERVER_ARGS=(
                --kv-cache-dtype "${KV_CACHE_DTYPE:-bfloat16}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-tilelang}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-tilelang}"
                --moe-runner-backend "${MOE_RUNNER_BACKEND:-aiter}"
                --context-length "${CONTEXT_LENGTH:-131072}"
                --tokenizer-worker-num $((TP_SIZE * 2))
            )
            ;;
        zai-org_GLM-5-FP8|zai-org_GLM-5.1-FP8)
            # InferenceX glm5_fp8_mi355x.sh. No ROCm-specific tuning beyond the
            # backends; the B200 tuning block is deliberately not mirrored here.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=8
            MEM_FRACTION_STATIC="0.85"
            MODEL_SERVER_ARGS=(
                --quantization fp8
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-triton}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-triton}"
                --tokenizer-worker-num $((TP_SIZE * 2))
            )
            ;;
        *)
            echo "[warn] no ROCm recipe for '${MODEL_NAME}' -- serving it with" \
                 "sglang defaults plus triton DSA. Add a block above before" \
                 "trusting any number from this run." >&2
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=8
            MODEL_SERVER_ARGS=(
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-triton}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-triton}"
                --tokenizer-worker-num $((TP_SIZE * 2))
            )
            ;;
    esac
else
    case "${MODEL_NAME}" in
        zai-org_GLM-5-FP8|zai-org_GLM-5.1-FP8)
            # InferenceX glm5_fp8_b200.sh: trtllm NSA, flashinfer MoE, 32K prefill
            # chunking, allreduce fusion, fixed stream-interval.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=8
            MEM_FRACTION_STATIC="0.85"
            MODEL_SERVER_ARGS=(
                --quantization fp8
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --attention-backend nsa
                --nsa-prefill-backend trtllm
                --nsa-decode-backend trtllm
                --moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
                --chunked-prefill-size 32768
                --max-prefill-tokens 32768
                --enable-flashinfer-allreduce-fusion
                --stream-interval 30
                --tokenizer-worker-num 6
            )
            ;;
        nvidia_GLM-5-NVFP4|lukealonso_GLM-5.1-NVFP4)
            # InferenceX glm5_fp4_b200.sh: the FP8 tuning above plus NVFP4's own
            # cuda-graph cap and scheduler poll interval. No
            # --disable-shared-experts-fusion -- NV's command omits it and
            # sglang's modelopt_fp4 path handles the mixed-precision shared
            # expert correctly. TP=4 so it cross-compares with MI355X MXFP4 TP4.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.8"
            MODEL_SERVER_ARGS=(
                --quantization modelopt_fp4
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --attention-backend nsa
                --nsa-prefill-backend trtllm
                --nsa-decode-backend trtllm
                --moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
                --chunked-prefill-size 32768
                --max-prefill-tokens 32768
                --enable-flashinfer-allreduce-fusion
                --stream-interval 30
                --tokenizer-worker-num 6
                --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-256}"
                --scheduler-recv-interval 10
            )
            ;;
        nvidia_GLM-5.2-NVFP4)
            # NVIDIA's official HF command verbatim, which is deliberately bare:
            # TP/quant/parsers/mem-fraction plus 16K chunking and nothing else.
            # The GLM-5/5.1 tuning block above deadlocks this model. [glm52-deadlock]
            # NV's card launches TP=8; we run TP=4 to match the rest of results/.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.8"
            MODEL_SERVER_ARGS=(
                --quantization modelopt_fp4
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --chunked-prefill-size 16384
            )
            ;;
        zai-org_GLM-5.3-Flash)
            # sglang cookbook's verified Blackwell cell
            # (docs/cookbook/autoregressive/GLM/GLM-5.3-Flash.mdx). Hybrid KDA +
            # DSA MoE, so no global --attention-backend: it would be wrong for two
            # thirds of the 45 layers, and sglang resolves DSA from the config.
            # The checkpoint self-declares FP8 block quant, so no --quantization.
            # trtllm DSA is paired with FP8 KV on purpose. [rocm-kv]
            # No --ep-size: the Blackwell cells leave the MoE pure TP.
            # 320B total / 18B active, ~306GB FP8 on disk -> ~77GB/GPU at TP=4.
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.8"
            MODEL_SERVER_ARGS=(
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-trtllm}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-trtllm}"
                --moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
                --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}"
                --context-length "${CONTEXT_LENGTH:-131072}"
            )
            [ -n "${EP_SIZE:-}" ] && MODEL_SERVER_ARGS+=(--ep-size "$EP_SIZE")
            ;;
        RadixArk_GLM-5.3-Flash-NVFP4|nvidia_GLM-5.3-Flash-NVFP4)
            # Same cookbook cell as the FP8 checkpoint above, with three changes
            # the W4A4 weights need: --quantization modelopt_fp4, mem-fraction
            # 0.85 (half-size weights leave more for the pools), and a decode
            # cuda-graph cap without which the server never survives long enough
            # to produce a cell. [nvfp4-graph]
            # The MoE runner is trtllm rather than the card's cutlass. [nvfp4-moe]
            # For the card's BF16-KV accuracy-reference pairing instead:
            #   KV_CACHE_DTYPE=bfloat16 DSA_PREFILL_BACKEND=tilelang \
            #   DSA_DECODE_BACKEND=tilelang ./GLM.sh --model .../GLM-5.3-Flash-NVFP4
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=4
            MEM_FRACTION_STATIC="0.85"
            MODEL_SERVER_ARGS=(
                --quantization modelopt_fp4
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
                --dsa-prefill-backend "${DSA_PREFILL_BACKEND:-trtllm}"
                --dsa-decode-backend "${DSA_DECODE_BACKEND:-trtllm}"
                --moe-runner-backend "${MOE_RUNNER_BACKEND:-flashinfer_trtllm}"
                --chunked-prefill-size "${CHUNKED_PREFILL_SIZE:-16384}"
                --context-length "${CONTEXT_LENGTH:-131072}"
                --cuda-graph-max-bs "${CUDA_GRAPH_MAX_BS:-128}"
            )
            [ -n "${EP_SIZE:-}" ] && MODEL_SERVER_ARGS+=(--ep-size "$EP_SIZE")
            ;;
        *)
            echo "[warn] no Blackwell recipe for '${MODEL_NAME}' -- serving it" \
                 "with sglang defaults. Add a block above before trusting any" \
                 "number from this run." >&2
            [ "$TP_SIZE" = "auto" ] && TP_SIZE=8
            MODEL_SERVER_ARGS=(
                --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8_e4m3}"
            )
            ;;
    esac
fi

# ===================== Server and Benchmark Setting =====================
# InferenceMax tuning (from InferenceX/glm5_fp8_mi355x.sh)
export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
# INT4-quantized quick all-reduce. This is the single biggest prefill lever on
# MI355X: in the ATOM stack it alone cut i8k/conc64 TPOT ~10.7% (53.6->47.9ms) by
# shrinking the TP all-reduce payload 4x. SGLang's own quick_all_reduce.py reads
# ROCM_QUICK_REDUCE_QUANTIZATION; aiter's path reads AITER_QUICK_REDUCE_QUANTIZATION
# -- set both so whichever all-reduce path is active gets quantized. Valid regimes:
# NONE (off), FP, INT8, INT6, INT4. Measured GLM-5.2-MXFP4 i8k/conc64 on 6PR:
# INT4 vs the docker's default INT8 -> 53.49->49.83ms TPOT (-6.8%), 1138->1228
# tok/s (+7.9%). Use a dedicated knob (QUICK_REDUCE_QUANT) so we override the
# image's baked-in ROCM_QUICK_REDUCE_QUANTIZATION=INT8; set QUICK_REDUCE_QUANT=INT8
# (or NONE) to compare.
export ROCM_QUICK_REDUCE_QUANTIZATION="${QUICK_REDUCE_QUANT:-INT4}"
export AITER_QUICK_REDUCE_QUANTIZATION="${QUICK_REDUCE_QUANT:-INT4}"

# GLM-5.2 DSA decode PAGED top-k routes to the DeepSeek-V4 "topk_v2" kernel, which
# is JIT-compiled by hipcc at CUDA-graph capture from
#   python/sglang/jit_kernel/include/sgl_kernel/deepseek_v4/topk_impl.cuh
# That header #includes <cooperative_groups.h> -- a CUDA header ROCm 7.2 does not
# ship -> hipcc "ninja exited with status 1 / cooperative_groups.h not found" ->
# server dies during startup. Disable topk_v2 until the kernel is hipified.
# STILL REQUIRED on the 0714 docker (v0.5.15.post1-rocm720-mi35x-20260714):
# empirically re-confirmed 2026-07-16 -- even though `from sgl_kernel import
# fast_topk_v2` imports (that is only the dispatcher), the real kernel is still the
# JIT topk_impl.cuh and it fails at capture exactly as before. Do NOT remove.
# Auto default by platform (while still allowing manual override):
#   ROCm -> 0 (workaround for topk_v2 JIT compile failure)
#   CUDA -> 1
if [ -z "${SGLANG_OPT_USE_TOPK_V2+x}" ]; then
    if [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
        export SGLANG_OPT_USE_TOPK_V2=0
    else
        export SGLANG_OPT_USE_TOPK_V2=1
    fi
fi
# Dense-decode "Design A" dual-graph (dense-decode-konly feature): captures BOTH a
# dense k-only and a sparse decode cuda-graph and dispatches per step on
# max_kv_len vs index_topk. For short context (kv_len <= index_topk, e.g. i1k) it
# runs the dense k-only path -- skipping the sparse indexer+topk+gather -> ~5-6%
# lower TPOT (measured GLM-5.2-MXFP4 i1k conc4: 11.06 vs 11.70 ms, GSM8K 0.931).
# Correct for mixed lengths (long context still uses sparse). Default ON so it is
# never forgotten; override with SGLANG_DSA_DECODE_DUAL_GRAPH=0. NOTE: captures 2x
# decode graphs (more capture time + memory). Only effective on branches that have
# the dense-decode feature + DSA models (ignored otherwise).
export SGLANG_DSA_DECODE_DUAL_GRAPH="${SGLANG_DSA_DECODE_DUAL_GRAPH:-1}"
# DSA indexer query Hadamard + FP8 quant fused into one Triton kernel (PR #30715).
# Opt-in, shape-guarded (gfx950, head_dim==block_size==128); default OFF in-product,
# so enable it here to avoid silently benchmarking the two-pass path. Override with
# SGLANG_DSA_FUSE_HADAMARD_QUANT=0. Ignored on branches without the feature.
export SGLANG_DSA_FUSE_HADAMARD_QUANT="${SGLANG_DSA_FUSE_HADAMARD_QUANT:-1}"
# export AITER_ONLINE_TUNE=1

# ===================== GLM-5.3-Flash (ROCm/gfx950, PR #36607) =====================
# The whole point of the gfx950 enablement is the AITER mHC path: matrix-free
# hypercomplex pre/post kernels plus a fused attention->FFN boundary, gated on
# HIP + gfx95 + SGLANG_USE_AITER. Without this env var the model still runs but
# falls back to the unfused reference path, which the PR measures at 5.42x slower
# -- i.e. exactly the "2x+ off" failure this script now warns about. Set it here
# so it can never be forgotten, and verify it actually fired (check_mhc_markers).
# ROCm only: on Blackwell there is no AITER, and exporting it there just puts a
# misleading line in the run's provenance.
if [[ "${MODEL_NAME}" == *GLM-5.3-Flash* ]] && is_rocm_gpu_env; then
    export SGLANG_USE_AITER="${SGLANG_USE_AITER:-1}"
fi

# Scheduler watchdog timeout (s). The torch-profiler teardown for an eager
# (no-cuda-graph) high-concurrency trace disposes millions of ProfilerResult
# objects single-threaded and can block the scheduler thread >20min
# (ProfilerResult::~ProfilerResult -> _M_dispose), tripping the default 1200s
# watchdog and killing the server mid-teardown. So default HIGH in --prof mode,
# but keep the normal 1200s for bench/serving so real hangs still surface fast.
# Override either way with WATCHDOG_TIMEOUT=<seconds> ./GLM.sh ...
if [ -z "${WATCHDOG_TIMEOUT:-}" ]; then
    if [ "$PROF_ENABLED" == "true" ]; then
        WATCHDOG_TIMEOUT=7200
    else
        WATCHDOG_TIMEOUT=1200
    fi
fi
HOST="localhost"
# Override with `--port <n>` or `PORT=<n> ./GLM.sh`. Default 8552.
PORT="${PORT:-8234}"
DATASET="random"
in_out_tokens=("1024:1024" "8192:1024"  "70000:300")
random_range_ratio=0.8
concurrencies=(4 8 16 32 64) # 128 256
# Optional env overrides (space-separated), e.g. for a targeted single-config
# rerun without editing this file:
#   IN_OUT_OVERRIDE="1024:1024" CONC_OVERRIDE="8" ./GLM.sh ...
if [ -n "${IN_OUT_OVERRIDE:-}" ]; then read -ra in_out_tokens <<< "$IN_OUT_OVERRIDE"; fi
if [ -n "${CONC_OVERRIDE:-}" ]; then read -ra concurrencies <<< "$CONC_OVERRIDE"; fi
# in_out_tokens=("1024:1024")
# concurrencies=(32 64)
PROMPT_MULTIPLIER=5

# Per-shape server args. --max-running-requests is a startup-only arg, so a
# shape listed here gets its own server launch (+~4 min model load); every
# shape without an entry shares one server as before.
#
# 70000:300 caps the running batch because that is what lines MI355X up with
# B200's operating point. Uncapped, MI355X admits ~37 of these requests at once
# and median TPOT at conc64 is 444 ms vs B200's 98 ms; at cap 8 it is ~90 ms,
# paid for with ~12% output throughput. B200 is not choosing this — its KV pool
# only fits ~8 such requests — so matching the cap is what makes the two
# comparable. Short shapes must NOT be capped: at 1024:1024/conc64 a cap of 8
# just serialises the run.
#
# GLM-5.3-Flash is the exception and defaults to NO cap. The premise above --
# that a B200 KV pool only fits ~8 of these requests -- does not hold for its
# hybrid KDA/DSA layout: only 11 of 45 layers keep a paged KV cache, so at TP4
# the pool is 4,979,200 tokens (32.85 GB fp8), enough for 64 concurrent 70K
# requests with room to spare. Measured 2026-08-28: at cap 8 the conc64 run just
# queues -- median TTFT 121.7 s and output throughput flat at ~127 tok/s from
# conc8 up -- so it reports the queue, not the model. Cap it
# explicitly with I70K_MAX_RUNNING_REQUESTS=8 if you want the GLM-5.2-comparable
# operating point back.
#
#   I70K_MAX_RUNNING_REQUESTS=16 ./GLM.sh   # different cap
#   I70K_MAX_RUNNING_REQUESTS=0  ./GLM.sh   # off -> single shared server (pre-08/13 behaviour)
case "${MODEL_NAME}" in
    *GLM-5.3*) _i70k_cap_default=0 ;;
    *)         _i70k_cap_default=8 ;;
esac
declare -A SHAPE_SERVER_ARGS=()
if [ "${I70K_MAX_RUNNING_REQUESTS:-$_i70k_cap_default}" != "0" ]; then
    SHAPE_SERVER_ARGS["70000:300"]="--max-running-requests ${I70K_MAX_RUNNING_REQUESTS:-$_i70k_cap_default}"
fi
# 4 steps, not 2, because the first captured DECODE step is not usable on ROCm
# and the second one is disturbed. Measured 2026-09-22 on GLM-5.3-Flash MXFP4
# TP4, v0.5.19-rocm720-mi35x-20260914, --profile-by-stage:
#
#   The decode session's GPU-side tracing starts 19-70 ms after its CPU side.
#   Both conc4 and conc64 traces hold two step[DECODE bs=N] CPU annotations but
#   only ONE gpu_user_annotation, and exactly one forward's worth of kernels (91
#   collectives = 2 per layer x 45 + 1). The first step falls entirely inside
#   that blind window: conc4's first GPU kernel lands +69.95 ms after the first
#   annotation, 1.5 ms before the second one. Prefill does not lose a step --
#   its first kernel precedes its first annotation by 0.36 ms, and a 150-200 ms
#   forward would swallow the latency anyway. It is decode's 2.5-15 ms steps
#   that are shorter than the arming delay. B200/CUPTI keeps both steps.
#
#   The steps that do survive are also the most disturbed ones. In the same
#   B200 capture the two decode forwards sit 11.08 ms apart while that run's own
#   median ITL was 6.08 ms, and the gap is mostly one spin-waiting collective:
#   the FIRST all-reduce of each forward (launch index 0 and 91 of 182) ran 3265
#   and 2471 us against a 4.6 us median. Ranks do not arm or flush their trace
#   buffers in lockstep, so the first collective after a perturbation absorbs
#   the skew. More steps push that cost into a smaller share of the sample.
#
# Cost is trace size and profiler teardown time, both linear in steps. Override
# with PROF_NUM_STEPS=2 to get the old behaviour.
if [ "$PROF_COMBINED" == "true" ]; then
    PROF_CMD=(--profile --profile-num-steps "${PROF_NUM_STEPS:-4}")
    COMBINED_SUFFIX="_Combined"
else
    PROF_CMD=(--profile --profile-num-steps "${PROF_NUM_STEPS:-4}" --profile-by-stage)
    COMBINED_SUFFIX=""
fi
# Optional: pass --profile-stages to restrict which stages profile-by-stage
# captures, e.g. PROF_STAGES_OVERRIDE="decode".
# WARNING: the 0714 ROCm image (v0.5.15.post1) IGNORES --profile-stages -- its
# profiler_manager hardcodes profiler_target_prefill_ct = profiler_target_decode_ct
# = num_steps, so profile-by-stage ALWAYS captures BOTH prefill(EXTEND) and
# decode(DECODE) regardless of this flag. It is kept here only for builds that
# do honor it. The real safeguard against the eager (no-cuda-graph) profiler
# teardown blowing past the watchdog is WATCHDOG_TIMEOUT (defaults to 7200 in
# --prof mode above); the ~100MB eager prefill trace teardown
# (ProfilerResult::~ProfilerResult) is nondeterministic and was observed to run
# anywhere from ~85s to >1200s, so give it headroom rather than relying on this.
if [ -n "${PROF_STAGES_OVERRIDE:-}" ]; then
    read -ra _prof_stages <<< "$PROF_STAGES_OVERRIDE"
    PROF_CMD+=(--profile-stages "${_prof_stages[@]}")
fi

# Vendor tag inserted into profiler trace filenames (e.g. ..._p8-AMD-TP-0-...).
# Auto-detect: AMD on ROCm, NV otherwise. Override with: VENDOR_TAG=AMD ./GLM.sh ...
if [ -z "${VENDOR_TAG:-}" ]; then
    if [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; then
        VENDOR_TAG="AMD"
    else
        VENDOR_TAG="NV"
    fi
fi

# ===================== Argument  =====================
SPECIAL_TAG="-bench"
if [ "$PROF_ENABLED" == "true" ]; then
    SPECIAL_TAG="-prof"
    concurrencies=(4)
    PROMPT_MULTIPLIER=2 # Faster for no cuda graph profiling

    # Debug
    in_out_tokens=("1024:16" "8192:16")
    concurrencies=(4 64)
    # Optional prof-mode overrides (space-separated), e.g. i1k only:
    #   PROF_IN_OUT_OVERRIDE="1024:16" PROF_CONC_OVERRIDE="4 64" ./GLM.sh --prof ...
    if [ -n "${PROF_IN_OUT_OVERRIDE:-}" ]; then read -ra in_out_tokens <<< "$PROF_IN_OUT_OVERRIDE"; fi
    if [ -n "${PROF_CONC_OVERRIDE:-}" ]; then read -ra concurrencies <<< "$PROF_CONC_OVERRIDE"; fi
fi
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
# Layout: results/<model>/<docker-image>/<mode>-Fixed-<tags>
# The mode (bench or prof) leads, then the scenario family: this script drives
# the fixed-length shapes (i1k/i8k/i70k), so its leaf is always "Fixed" to keep
# it apart from the agentic replay runs.
# MTP stays in the path so an MTP run and its non-MTP twin can share a --tag
# without overwriting each other.
LEAF_TAG="${SPECIAL_TAG#-}-Fixed${MTP_TAG}${USER_TAG}"
LOG_DIR="$HOME/SGLang-benchmarks/results/${MODEL_NAME}/$DOCKER_FILENAME/${LEAF_TAG}"
FINISH_LOG="$LOG_DIR/Finish.log"
# Single continuous server log. Exported so log_command() can stamp a banner into
# it before each client command (warmup / GSM8K / per-config warmup / bench), so
# the interleaved server log can be sliced back to "which benchmark is this".
SERVER_LOG="${LOG_DIR}/server_${MODEL_NAME}.log"
mkdir -p "$LOG_DIR"
touch "$FINISH_LOG"
if [ "$PROF_ENABLED" == "true" ]; then
    export SGLANG_TORCH_PROFILER_DIR=$LOG_DIR
fi

log_command() {
    local logfile=$1
    shift 
    
    # Record command
    echo ">>>Executing command:" | tee -a "$logfile"
    echo "$*" | tee -a "$logfile"  
    echo "---" | tee -a "$logfile"

    # Also stamp a greppable banner into the (continuous) server log so its
    # interleaved output can be attributed to the client command that drove it.
    # Grep '##### CLIENT CMD' in server_*.log to find each phase boundary.
    if [ -n "${SERVER_LOG:-}" ] && [ -f "${SERVER_LOG}" ]; then
        {
            echo ""
            echo "##### CLIENT CMD @ $(date '+%F %T') #####"
            echo "$*"
            echo "##########################################"
        } >> "${SERVER_LOG}"
    fi

    # Execute command
    "$@" 2>&1 | tee -a "$logfile"
}

list_profiler_dirs() {
    find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+(\.[0-9]+)?$' || true
}

rename_profiler_artifacts() {
    local input_tokens=$1
    local output_tokens=$2
    local c=$3
    local num_prompts=$4
    local before_dirs=$5
    local after_dirs=$6
    local target_dir_name="prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}"
    local target_dir_path="${LOG_DIR}/${target_dir_name}"
    local new_dirs

    new_dirs=$(comm -13 <(printf '%s\n' "${before_dirs}" | sort) <(printf '%s\n' "${after_dirs}" | sort))
    if [ -z "${new_dirs}" ]; then
        echo "No new profiler directory found under ${LOG_DIR}"
        return 0
    fi

    local src_dir src_dir_path
    src_dir=$(printf '%s\n' "${new_dirs}" | tail -n 1)
    src_dir_path="${LOG_DIR}/${src_dir}"
    if [ "${src_dir}" != "${target_dir_name}" ]; then
        if [ -e "${target_dir_path}" ]; then
            target_dir_path="${LOG_DIR}/${target_dir_name}_$(date +%s)"
            echo "Target directory exists. Using ${target_dir_path}"
        fi
        mv "${src_dir_path}" "${target_dir_path}"
        echo "Renamed profiler dir: ${src_dir} -> $(basename "${target_dir_path}")"
    fi

    local trace_file filename tp_rank new_name
    for trace_file in "${target_dir_path}"/*-TP-*.trace.json.gz; do
        [ -f "${trace_file}" ] || continue
        filename=$(basename "${trace_file}")
        tp_rank=$(sed -E 's/^.*-TP-([0-9]+)\.trace\.json\.gz$/\1/' <<< "${filename}")
        new_name="in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}-${VENDOR_TAG}-TP-${tp_rank}${NOGRAPH_SUFFIX}.trace.json.gz"
        mv "${trace_file}" "${target_dir_path}/${new_name}"
        echo "Renamed trace: ${filename} -> ${new_name}"
    done
}

rename_profiler_artifacts_by_stage() {
    local input_tokens=$1
    local output_tokens=$2
    local c=$3
    local num_prompts=$4
    local before_dirs=$5
    local after_dirs=$6
    local target_dir_name="prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}"
    local target_dir_path="${LOG_DIR}/${target_dir_name}"
    local new_dirs

    new_dirs=$(comm -13 <(printf '%s\n' "${before_dirs}" | sort) <(printf '%s\n' "${after_dirs}" | sort))
    if [ -z "${new_dirs}" ]; then
        echo "No new profiler directory found under ${LOG_DIR}"
        return 0
    fi

    local src_dir src_dir_path
    src_dir=$(printf '%s\n' "${new_dirs}" | tail -n 1)
    src_dir_path="${LOG_DIR}/${src_dir}"
    if [ "${src_dir}" != "${target_dir_name}" ]; then
        if [ -e "${target_dir_path}" ]; then
            target_dir_path="${LOG_DIR}/${target_dir_name}_$(date +%s)"
            echo "Target directory exists. Using ${target_dir_path}"
        fi
        mv "${src_dir_path}" "${target_dir_path}"
        echo "Renamed profiler dir: ${src_dir} -> $(basename "${target_dir_path}")"
    fi

    local trace_file filename tp_rank stage new_name
    for trace_file in "${target_dir_path}"/*-TP-*.trace.json.gz; do
        [ -f "${trace_file}" ] || continue
        filename=$(basename "${trace_file}")
        tp_rank=$(sed -E 's/^.*-TP-([0-9]+)-(EXTEND|DECODE)\.trace\.json\.gz$/\1/' <<< "${filename}")
        stage=$(sed -E 's/^.*-TP-([0-9]+)-(EXTEND|DECODE)\.trace\.json\.gz$/\2/' <<< "${filename}")

        if [ "${tp_rank}" = "${filename}" ] || [ "${stage}" = "${filename}" ]; then
            echo "Skip unmatched trace name: ${filename}"
            continue
        fi

        new_name="in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}-${VENDOR_TAG}-TP-${tp_rank}-${stage}${NOGRAPH_SUFFIX}.trace.json.gz"
        mv "${trace_file}" "${target_dir_path}/${new_name}"
        echo "Renamed trace: ${filename} -> ${new_name}"
    done
}

prof_cmd_has_profile_by_stage() {
    local arg
    for arg in "${PROF_CMD[@]}"; do
        if [ "${arg}" = "--profile-by-stage" ]; then
            return 0
        fi
    done
    return 1
}

start_server() {
    local logfile="${SERVER_LOG}"
    echo ">>> Starting SGLang server" | tee "$logfile"

    local cmd=(
        python3 -m sglang.launch_server
            --model $MODEL_PATH
            --tp $TP_SIZE
            --host $HOST
            --port $PORT
            --trust-remote-code
            --tool-call-parser glm47
            --reasoning-parser glm45
            --watchdog-timeout "${WATCHDOG_TIMEOUT:-1200}"
            --mem-fraction-static "$MEM_FRACTION_STATIC"
            --model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": ${WEIGHT_LOAD_THREADS:-32}}"
    )
    # Prefix reuse off for every model so each (shape, concurrency) cell measures a
    # cold prefill. This is NOT the cosmetic difference from the cookbook's flag
    # list that it looks like: bench_serving reuses the same seed for every cell,
    # so the tree carries prompts over from the previous cell and the sweep grades
    # a warm cache from the second cell on. Measured 2026-09-15 on GLM-5.3-Flash
    # TP4/B200 with the tree left enabled, at 1024 input (short enough that
    # chunked prefill cannot account for it, so every hit is real cross-request
    # reuse): conc4 4.5% of prefill tokens cached -- it runs first, on an empty
    # tree -- then 47.7 / 47.8 / 46.4 / 45.5% at conc 8 / 16 / 32 / 64. Half the
    # prefill silently disappears and TTFT with it. Set DISABLE_RADIX_CACHE=0 only
    # if measuring the cache is the point.
    if [ "${DISABLE_RADIX_CACHE:-$_radix_default}" = "1" ]; then
        cmd+=(--disable-radix-cache)
    fi

    # Everything model- and platform-specific was resolved up front, in the
    # per-model configuration block. See it for why any particular flag is here.
    cmd+=("${MODEL_SERVER_ARGS[@]}")

    if is_rocm_gpu_env && [ "$DUAL_STREAM_ROCM" == "true" ]; then
        # Two independent toggles must both be set for full ROCm dual-stream:
        #   (a) --disable-shared-experts-fusion
        #         Forces num_fused_shared_experts=0 so DeepseekV2MoE.forward
        #         takes forward_normal_dual_stream (shared ∥ routed overlap)
        #         instead of forward_normal which would use the fused
        #         _fused_append_shared_experts_kernel.
        #   (b) SGLANG_ENABLE_HIP_DUAL_STREAM=1
        #         Required to actually create alt_stream on ROCm. Without
        #         it, alt_stream=None on HIP and *both* the NSA-decode A_v4
        #         layout and the MoE forward_normal_dual_stream are skipped.
        #         (Default OFF because the layout regresses on MI355X — see
        #         tools/dual_stream_regression_analysis.md for full analysis.)
        NEED_DISABLE_SHARED_FUSION="true"
        export SGLANG_ENABLE_HIP_DUAL_STREAM=1
    fi

    # ===== Piecewise CUDA Graph (PCG) prefill backend — NVIDIA/CUDA only =====
    # The dev-glm52-nvfp4 image launched prefill with tc_piecewise by default,
    # but the v0.5.15.post1 release auto-disables it for the GLM-5.2 DSA path
    # (prefill=PhaseConfig(backend='disabled')). That regressed short-input TTFT
    # badly (i1k conc4: 208ms vs 72ms; ~1.5-3x worse on 1024-token inputs) while
    # long inputs were unaffected. PCG is on-by-default upstream but GLM-5.2 hits
    # an auto-disable rule (model-arch blacklist); explicitly selecting the
    # prefill backend skips the whole auto-disable cascade (this is the current
    # replacement for the old --enforce-piecewise-cuda-graph).
    #
    # CUDA-only on purpose: tc_piecewise is unsupported on ROCm/NPU/CPU/MPS/XPU
    # (sglang's own is_hip()/is_npu()/... rules disable it), and older images may
    # not even expose --cuda-graph-backend-prefill. Gate to the non-ROCm path.
    # Toggle with ENABLE_PIECEWISE_CUDA_GRAPH=0/1.
    #
    # Default OFF for GLM-5.3-Flash: none of the cookbook's verified cells select
    # a prefill cuda-graph backend for it, and the one command that does mention
    # prefill graphs (the GB300 encoder-disaggregation recipe) passes
    # --disable-prefill-cuda-graph. Forcing tc_piecewise onto the KDA/DSA hybrid
    # would be benchmarking an unvalidated path. Set
    # ENABLE_PIECEWISE_CUDA_GRAPH=1 to measure it deliberately.
    case "${MODEL_NAME}" in
        *GLM-5.3*) _pcg_default=0 ;;
        *)         _pcg_default=1 ;;
    esac
    if ! is_rocm_gpu_env && [ "${ENABLE_PIECEWISE_CUDA_GRAPH:-$_pcg_default}" = "1" ]; then
        cmd+=(--cuda-graph-backend-prefill tc_piecewise)
    fi

    # Auto-add --disable-shared-experts-fusion only when ROCm dual-stream is
    # enabled (forces num_fused_shared_experts=0 so MoE.forward takes
    # forward_normal_dual_stream instead of the fused path).
    # NOT added for nvidia/GLM-5-NVFP4 or amd/GLM-5.1-MXFP4: NV's official
    # sglang command, InferenceX glm5_fp4_b200.sh, and glm5.1_fp4_mi355x.sh
    # all omit it (sglang's modelopt_fp4 path handles NVFP4's mixed-precision
    # shared expert correctly; AMD's MXFP4 quantizes shared experts too).
    if [ "$NEED_DISABLE_SHARED_FUSION" = "true" ]; then
        cmd+=(--disable-shared-experts-fusion)
    fi

    # Override chunked-prefill-size for the blocks that hardcode it (the
    # GLM-5.3-Flash blocks already read CHUNKED_PREFILL_SIZE inline; appending
    # here too is harmless because the later flag wins in argparse).
    #   CHUNKED_PREFILL_SIZE=131072 ./GLM.sh ...
    if [ -n "${CHUNKED_PREFILL_SIZE:-}" ]; then
        cmd+=(--chunked-prefill-size "$CHUNKED_PREFILL_SIZE")
    fi

    # Args for the shape group this server is being launched for (see
    # SHAPE_SERVER_ARGS). Placed before SERVER_EXTRA_ARGS so an explicit
    # SERVER_EXTRA_ARGS from the caller still wins in argparse.
    if [ -n "${SHAPE_EXTRA_SERVER_ARGS:-}" ]; then
        read -ra _shape_srv <<< "$SHAPE_EXTRA_SERVER_ARGS"
        cmd+=("${_shape_srv[@]}")
    fi

    # Generic passthrough for extra server args (appended last -> wins in argparse).
    # Space-separated. e.g. test prefill piecewise cuda graph:
    #   SERVER_EXTRA_ARGS="--cuda-graph-backend-prefill tc_piecewise --piecewise-cuda-graph-compiler eager" ./GLM.sh ...
    if [ -n "${SERVER_EXTRA_ARGS:-}" ]; then
        read -ra _extra_srv <<< "$SERVER_EXTRA_ARGS"
        cmd+=("${_extra_srv[@]}")
    fi

    if [ "$MTP_ENABLED" == "true" ]; then
        # The EAGLE chain is per-model. InferenceX glm5_fp8_mi355x_mtp.sh runs
        # (steps=3, topk=1, draft=4) and that stays the default, but every
        # GLM-5.3-Flash cell in the cookbook that turns MTP on -- the Low Latency
        # column on all six platforms, FP8 and NVFP4 alike -- runs (5, 1, 6). Taking
        # the GLM-5 chain on GLM-5.3-Flash verifies 4 draft tokens where the
        # published numbers verify 6, so its speedup is not the measured one.
        # SGLANG_ENABLE_SPEC_V2=1 enables sglang's new spec scheduler (also set by InferenceX).
        # On MI355X we keep --nsa-{prefill,decode}-backend tilelang from above; do NOT
        # override --attention-backend (InferenceX doesn't either, tilelang NSA + EAGLE works).
        # Override for tree spec (topk>1), e.g.:
        #   SPEC_TOPK=2 SPEC_NUM_STEPS=5 SPEC_DRAFT_TOKENS=6 \
        #   SERVER_EXTRA_ARGS="--attention-backend triton" ./GLM.sh --mtp --prof
        # topk>1 is rejected by the DSA backend, so tree runs need triton.
        echo ">>> Speculative Decoding (MTP) is ENABLED." | tee -a "$logfile"
        export SGLANG_ENABLE_SPEC_V2=1
        case "${MODEL_NAME}" in
            *GLM-5.3*) _spec_steps=5; _spec_draft=6 ;;
            *)         _spec_steps=3; _spec_draft=4 ;;
        esac
        cmd+=(
            --speculative-algorithm EAGLE
            --speculative-num-draft-tokens "${SPEC_DRAFT_TOKENS:-$_spec_draft}"
            --speculative-num-steps "${SPEC_NUM_STEPS:-$_spec_steps}"
            --speculative-eagle-topk "${SPEC_TOPK:-1}"
        )
    fi

    if [ ${#EXTRA_SERVER_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_SERVER_ARGS[@]}")
    fi

    # DRY_RUN=1 prints the resolved launch command and exits without loading the
    # model (~4 min) — for checking that a new model's args resolve as intended.
    if [ "${DRY_RUN:-0}" = "1" ]; then
        set +x
        echo ""
        echo ">>> DRY_RUN: resolved server command"
        printf '    %s\n' "${cmd[@]}"
        echo ""
        echo ">>> MODEL_NAME=${MODEL_NAME}  TP_SIZE=${TP_SIZE}  MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC}"
        echo ">>> SGLANG_USE_AITER=${SGLANG_USE_AITER:-<unset>}"
        echo ">>> model block args: ${MODEL_SERVER_ARGS[*]}"
        echo ">>> shapes=${in_out_tokens[*]}  concurrencies=${concurrencies[*]}"
        echo ">>> LOG_DIR=${LOG_DIR}"
        exit 0
    fi

    # Preflight: fail fast if the port is already taken (otherwise the model
    # loads for ~2 min and only then dies with "[Errno 98] Address already in
    # use"). Common cause: a stale/orphaned sglang server from a previous run,
    # or another server sharing this host. Stop it or pick another --port.
    if (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null; then
        exec 3>&- 3<&-
        echo "!!! ERROR: ${HOST}:${PORT} is already in use. Stop the existing server " \
             "(e.g. 'pkill -9 -f sglang.launch_server') or run with '--port <free-port>'." | tee -a "$logfile"
        exit 1
    fi

    # Start server in background
    echo ">>> Executing command:" | tee -a "$logfile"
    echo "${cmd[*]}" | tee -a "$logfile"
    echo "---" | tee -a "$logfile"
    "${cmd[@]}" 2>&1 | tee -a "$logfile" &

    echo ">>> Waiting for server to be ready (checking: '${logfile}')..." | tee -a "$logfile"
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:$PORT/health" 2>/dev/null)" = "200" ]; do
        # Detect server death during startup (port bind failure, OOM, GPU fault)
        # so we don't poll forever. pgrep is scoped to this server's port.
        if ! pgrep -f "sglang.launch_server.*--port $PORT" >/dev/null 2>&1; then
            echo "!!! ERROR: server process died during startup. See '${logfile}' " \
                 "(look for 'Address already in use', OOM, or 'Memory access fault')." | tee -a "$logfile"
            exit 1
        fi
        echo "Waiting for server to be ready at http://${HOST}:$PORT/health..."
        sleep 5
    done

    # Verify the arch-gated fast paths actually engaged before spending an hour
    # benchmarking. Warns; does not abort, so a deliberate fallback run still works.
    check_mhc_markers
}

# ===================== gfx950 fast-path check =====================
# On gfx950 the GLM-5.3-Flash mHC fast paths are env/arch gated (HIP + gfx95 +
# SGLANG_USE_AITER). When a gate silently misses -- SGLANG_USE_AITER unset, wrong
# GPU, an image whose aiter lacks aiter.ops.mhc -- the model still serves correct
# tokens, just on the unfused reference path that PR #36607 measures at 5.42x
# slower. Nothing in the benchmark output says "you measured the slow path", so
# state it explicitly once at startup. The evidence that the fast path engaged is
# that every TP rank logs the mHC pre/post line.
#
# Only the pre/post line is checked, and that is a correction. This function used
# to also grep for "Using fused AITER mHC attention-to-FFN boundary" and warn when
# it was missing, which it now always is -- so every GLM-5.3-Flash run printed the
# 5.42x warning while running the fast path. #36607's boundary fusion came back
# upstream in a different shape, as #39200's glm5_next.hc_ffn_post_pre: wired
# through MHCLayerCommunicator when is_cross_layer_mhc_fusion_enabled() (true on
# gfx95 + AITER), logging nothing, and deliberately capped at
# _MHC_FUSED_BOUNDARY_MAX_TOKENS=16 so a conc64 decode does not take it -- a rank
# count was never the right shape for that question anyway. Note #39200 is NOT in
# the 20260914 image's own tree; that run had it only because the Day-0 PR heads
# each carry a newer main. Verified 2026-09-21 on the ten-PR Day-0 stack.
check_mhc_markers() {
    case "${MODEL_NAME}" in *GLM-5.3-Flash*) ;; *) return 0 ;; esac
    is_rocm_gpu_env || return 0
    local log="${SERVER_LOG}" pre
    pre=$(grep -c 'Using AITER gfx950 mHC pre/post kernels' "$log" 2>/dev/null || true)
    echo ">>> [mHC] AITER gfx950 pre/post kernels: ${pre}/${TP_SIZE} ranks"
    if [ "${pre:-0}" -lt "$TP_SIZE" ]; then
        echo "!!! WARNING: the GLM-5.3-Flash gfx950 mHC pre/post kernels did NOT engage"
        echo "!!! on every rank (${pre:-0}/${TP_SIZE}). The unfused fallback is 5.42x slower"
        echo "!!! (PR #36607), so these numbers are not comparable to any published"
        echo "!!! GLM-5.3-Flash result."
        echo "!!! Check SGLANG_USE_AITER=${SGLANG_USE_AITER:-<unset>}, that the GPU is gfx950,"
        echo "!!! and that this image's aiter exposes aiter.ops.mhc (mhc_pre / mhc_post)."
    fi
}

# WARMUP_CONCURRENCY=1 is load-bearing on GLM-5.3-Flash, not a tuning choice.
# Several ~1000-token prefills arriving together as the FIRST traffic a fresh
# server sees put it into a state where later requests stop terminating, for the
# life of the process. This warmup at --max-concurrency 4 was exactly that, and
# it runs before accuracy_test, so every GSM8K number this script produced for
# the model was graded on an already-broken server: 20.32% and 31.01% on two
# runs against ~97% from a hand-written launch, with 65-75% of completions
# running to the token cap.
#
# Serialising it removes the trigger and keeps the point of the warmup, which is
# to JIT-compile the kernels for a 1024-token prefill -- the same shapes get
# compiled, just not concurrently. It also inoculates the server: once anything
# has been served, the same concurrent batch is harmless, which is why the bench
# loop's own in1024/conc4 config is safe afterwards.
#
# Shortening --random-input instead would not work: it would stop warming the
# 1024-token prefill path this exists to warm.
#
# Measured on rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260909, TP4, GSM8K 1,319 at
# 1319 threads: concurrency 4 -> poisoned 6/6 (20-35%); concurrency 1 -> 96.74%.
# The defect is upstream, not in the ROCm PRs: it reproduces on sgl-project/
# sglang#36607's own tree and neither that patch nor the Day-0 stack touches the
# linear-attention state path. Full write-up in GLM53-Flash-ROCm-PrStack_Repro.md.
warmup() {
    local warmup_log="${LOG_DIR}/warmup.log"
    local warmup_cmd=(
        python3 -m sglang.bench_serving 
        --host $HOST 
        --port "${PORT}" 
        --model "${MODEL_PATH}" 
        --dataset-name "${DATASET}" 
        --random-input 1024
        --random-output 16
        --random-range-ratio "${random_range_ratio}"
        --max-concurrency "${WARMUP_CONCURRENCY:-1}"
        --num-prompt 4 
        --output-file /dev/null
        --ready-check-timeout-sec "${READY_CHECK_TIMEOUT_SEC:-600}"
    )
    log_command "$warmup_log" "${warmup_cmd[@]}"
}

# Which GSM8K harness to use: sgl-eval for thinking models, the in-tree
# bench_sglang.py otherwise. Force either way with ACCURACY_HARNESS=sgl-eval|bench_sglang.
#
# The GLM-5.3 arm is deliberately platform-independent. sgl-eval is not a ROCm
# workaround -- it is the only harness that grades this model the way it answers
# (zero-shot chat template, thinking enabled, \boxed{} extraction). bench_sglang.py
# is 5-shot raw completion capped at 512 tokens and reads "the last number anywhere
# in the output", which costs the same server 6-25 points. So MI355X and B200 must
# both take this arm or their accuracy columns are not the same measurement.
accuracy_harness() {
    if [ -n "${ACCURACY_HARNESS:-}" ]; then
        echo "$ACCURACY_HARNESS"
        return 0
    fi
    case "${MODEL_NAME}" in
        *GLM-5.3*)
            if command -v sgl-eval >/dev/null 2>&1; then
                echo "sgl-eval"
            else
                echo "[warn] sgl-eval not on PATH; falling back to bench_sglang.py, which" \
                     "under-scores this model by 6-25 points. Install with:" \
                     "pip install 'git+https://github.com/sgl-project/sgl-eval'" >&2
                echo "bench_sglang"
            fi
            ;;
        *) echo "bench_sglang" ;;
    esac
}

# Surface the headline score in the run's own stdout; otherwise it is buried in the
# harness log (bench_sglang.py) or a nested metrics.json (sgl-eval).
report_gsm8k_score() {
    local log=$1
    if [ "$(accuracy_harness)" = "sgl-eval" ]; then
        local mj
        mj=$(find "${LOG_DIR}/sgl_eval_gsm8k" -name metrics.json 2>/dev/null \
             | xargs -r ls -t 2>/dev/null | head -1)
        if [ -n "$mj" ]; then
            python3 - "$mj" <<'PY' || true
import json, sys
d = json.load(open(sys.argv[1])); a = d["aggregate"]; n = d["num_examples"]
print(">>> [gsm8k] sgl-eval %.2f%% (%d/%d) | truncated %.2f%% | errors %.3f | wall %.0fs | %s threads"
      % (a["score"] * 100, round(a["score"] * n), n, a["truncated_rate"] * 100,
         a["error_rate"], d["latency_seconds"], d.get("num_threads", "?")))
PY
        else
            echo "[warn] sgl-eval produced no metrics.json under ${LOG_DIR}/sgl_eval_gsm8k"
        fi
    else
        local acc
        acc=$(grep -aoE "Accuracy: [0-9.]+" "$log" | tail -1 | awk '{print $2}')
        [ -n "$acc" ] && echo ">>> [gsm8k] bench_sglang.py accuracy ${acc}"
    fi
}

accuracy_test() {
    # Optional skip (e.g. quick perf-only runs): SKIP_GSM8K=1 ./GLM.sh ...
    if [ "${SKIP_GSM8K:-0}" = "1" ]; then
        echo ">>> SKIP_GSM8K=1 set — skipping GSM8K accuracy test."
        return 0
    fi
    # GSM8K
    gsm8k_logfile=$LOG_DIR/Accuracy_GSM8K.log
    if ! grep -q "$gsm8k_logfile" "$FINISH_LOG"; then
        echo ">>> Running Accuracy check (GSM8K)..."
        local gsm8k_cmd
        if [ "$(accuracy_harness)" = "sgl-eval" ]; then
            # ===== sgl-eval: the right grader for a thinking model =====
            # benchmark/gsm8k/bench_sglang.py measures something else entirely:
            # 5-shot raw completion (no chat template, so no thinking), 512-token
            # cap, and "the last number anywhere in the output" as the answer.
            # On GLM-5.3-Flash that scores 71-90% depending on unrelated server
            # tuning, while sgl-eval's zero-shot + thinking + \boxed{} extraction
            # scores 95.8% and is stable. Measured on this box, same server:
            #   bench_sglang.py 1319q  89.4-90.3% | under GLM.sh's env tuning 71.3%
            #   sgl-eval        1319q  95.83%
            #
            # Two independent wall-clock levers. The uncapped 64-thread reference
            # run took 602.7s at 776 tok/s, and both levers were needed:
            #
            # GSM8K_MAX_TOKENS -- the dominant one, because the tail is serial.
            # Token distribution over the 1319 questions: p50=84, p90=201,
            # p99=2135, max=32768. Ten requests exceed 4096 and the slowest single
            # request alone took 565.6s of the 602.7s wall -- no amount of
            # concurrency touches that, it is one request emitting tokens one at a
            # time. Capping at 4096 drops 260K of the 467K generated tokens (56%)
            # and loses exactly one correct answer: 95.83% -> 95.75%, -0.08 points.
            # Measured cost of tighter caps: 3072 also -0.08, 2048 -0.15,
            # 1024 -0.30. Set GSM8K_MAX_TOKENS=32768 for the uncapped reference.
            #
            # GSM8K_THREADS -- sgl-eval's --num-threads is documented as
            # "concurrent requests" and is the direct equivalent of
            # bench_serving's --parallel (runner uses a pool of
            # min(num_threads, num_samples) in-flight requests); there is no
            # separate --parallel flag. The default is 64, which starves an 8-GPU
            # box on this workload: 64 concurrent *short* answers only reached
            # 776 tok/s, whereas 64 concurrent *long* ones on the same hardware
            # sustained 3229 tok/s. So the deficit is idle GPU waiting on
            # round-trips, not compute, and raising the ceiling is nearly free.
            # 207K post-cap tokens at that rate is ~60-90s end to end.
            gsm8k_cmd=(
                sgl-eval run gsm8k
                    --base-url "http://${HOST}:${PORT}/v1"
                    --model "${SERVED_MODEL_NAME:-$MODEL_PATH}"
                    --num-threads "${GSM8K_THREADS:-1319}"
                    --max-tokens "${GSM8K_MAX_TOKENS:-4096}"
                    --temperature "${GSM8K_TEMPERATURE:-1.0}"
                    --top-p "${GSM8K_TOP_P:-0.95}"
                    --seed "${GSM8K_SEED:-0}"
                    --thinking
                    --out-dir "${LOG_DIR}/sgl_eval_gsm8k"
            )
            if [ -n "${GSM8K_NUM_EXAMPLES:-}" ]; then
                gsm8k_cmd+=(--num-examples "${GSM8K_NUM_EXAMPLES}")
            fi
        else
            # --parallel caps how many GSM8K requests run concurrently. A very high
            # value (e.g. 1200) floods the server into one giant batch (#running-req
            # ~1197) which can trip a GPU memory-access fault in the MXFP4/tilelang
            # NSA kernels on MI355X. Accuracy is unaffected by lowering it (same 1200
            # questions, fewer in flight). Override with GSM8K_PARALLEL=<n>.
            gsm8k_cmd=(
                python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py
                    --port "$PORT"
                    --num-questions "${GSM8K_NUM_QUESTIONS:-1319}"
                    --parallel "${GSM8K_PARALLEL:-1319}"
            )
        fi
        log_command "$gsm8k_logfile" "${gsm8k_cmd[@]}"
        echo "$gsm8k_logfile" >> "$FINISH_LOG"
        report_gsm8k_score "$gsm8k_logfile"
    else
        echo "Found Accuracy_GSM8K.log in ${FINISH_LOG}. Skipping."
    fi
}

run_benchmarks() {
    # Benchmark Loop
    for io_pair in "${in_out_tokens[@]}"; do
        IFS=":" read -r input_tokens output_tokens <<< "$io_pair"
        for c in "${concurrencies[@]}"; do
            local num_prompts=$((c * PROMPT_MULTIPLIER))
            local logfile="${LOG_DIR}/bench_in${input_tokens}_out${output_tokens}_conc${c}.log"

            local cmd=(
                python3 -m sglang.bench_serving
                --host "${HOST}"
                --port "${PORT}"
                --model "${MODEL_PATH}"
                --dataset-name "${DATASET}"
                --random-input "${input_tokens}"
                --random-output "${output_tokens}"
                --random-range-ratio "${random_range_ratio}"
                --max-concurrency "${c}"
                --num-prompt "${num_prompts}"
                --output-file /dev/null
                --ready-check-timeout-sec "${READY_CHECK_TIMEOUT_SEC:-600}"
            )
            
            # Add profiling args
            if [ "$PROF_ENABLED" == "true" ]; then
                cmd+=("${PROF_CMD[@]}")
            fi

            # Determine skip condition:
            # - Profiling mode: skip if the profile output directory already exists
            # - Benchmark mode: skip if logfile is recorded in Finish.log
            local skip="false"
            local prof_dir="${LOG_DIR}/prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}"
            if [ "$PROF_ENABLED" == "true" ]; then
                if [ -d "${prof_dir}" ]; then
                    echo "Found profile dir ${prof_dir}. Skipping."
                    skip="true"
                fi
            else
                if grep -q "$logfile" "$FINISH_LOG"; then
                    echo "Found $logfile in ${FINISH_LOG}. Skipping."
                    skip="true"
                fi
            fi

            if [ "$skip" == "false" ]; then
                echo "Running: $logfile"
                # Per-config JIT warmup (prof mode only): run the SAME
                # (input,output,conc) shape once WITHOUT --profile so every kernel
                # for this config is already JIT-compiled/cached. Otherwise a
                # cold-cache compile (torch.compile/dynamo/triton codegen) can land
                # INSIDE the profiled window and pollute the trace with millions of
                # ast/isinstance python_function events (observed: i1k conc64
                # no-graph EXTEND ballooned to 4.4M events / 64MB vs ~0.5M clean).
                # That event bloat also inflates the profiler teardown
                # (ProfilerResult dispose is O(events)) which is what tripped the
                # 1200s watchdog. Warm cache -> clean small trace + fast teardown.
                # Disable with PROF_WARMUP=0.
                local _warm_dur=0
                if [ "$PROF_ENABLED" == "true" ] && [ "${PROF_WARMUP:-1}" == "1" ]; then
                    local warmup_cfg_log="${LOG_DIR}/warmup_in${input_tokens}_out${output_tokens}_conc${c}.log"
                    local warmup_cfg_cmd=(
                        python3 -m sglang.bench_serving
                        --host "${HOST}" --port "${PORT}" --model "${MODEL_PATH}"
                        --dataset-name "${DATASET}"
                        --random-input "${input_tokens}" --random-output "${output_tokens}"
                        --random-range-ratio "${random_range_ratio}"
                        --max-concurrency "${c}" --num-prompt "${num_prompts}"
                        --output-file /dev/null
                        --ready-check-timeout-sec "${READY_CHECK_TIMEOUT_SEC:-600}"
                    )
                    echo ">>> [warmup] JIT warmup for in${input_tokens}_out${output_tokens}_conc${c} (no --profile)"
                    local _warm_t0=$(date +%s)
                    log_command "$warmup_cfg_log" "${warmup_cfg_cmd[@]}" \
                        || echo "[warn] per-config warmup failed; continuing to profiled run."
                    _warm_dur=$(( $(date +%s) - _warm_t0 ))
                fi
                local _prof_t0=$(date +%s)
                local profiler_dirs_before=""
                local profiler_dirs_after=""
                if [ "$PROF_ENABLED" == "true" ]; then
                    profiler_dirs_before=$(list_profiler_dirs) # Get the current folders under $LOG_DIR
                fi
                # Don't let a profiler-teardown crash (server segfault / NCCL
                # heartbeat / known ROCm torch-profiler teardown fault) abort the
                # whole run via `set -e`. The trace is typically already flushed
                # to disk before the crash, so we still want to fall through to
                # the rename step below and continue with the next iteration.
                if ! log_command "$logfile" "${cmd[@]}"; then
                    echo "[warn] command exited non-zero for ${logfile} (likely profiler teardown crash); traces may still be present — continuing to rename."
                fi
                echo "$logfile" >> "$FINISH_LOG"

                # Record per-config timing: warmup (JIT precompile) vs the profiled
                # run itself (includes capture + profiler teardown). Written to
                # profile_timing.log next to the traces.
                if [ "$PROF_ENABLED" == "true" ]; then
                    local _prof_dur=$(( $(date +%s) - _prof_t0 ))
                    printf '%-40s warmup=%4ds  profile+teardown=%5ds  total=%5ds\n' \
                        "in${input_tokens}_out${output_tokens}_conc${c}${NOGRAPH_SUFFIX}" \
                        "${_warm_dur}" "${_prof_dur}" "$(( _warm_dur + _prof_dur ))" \
                        | tee -a "${LOG_DIR}/profile_timing.log"
                fi

                if [ "$PROF_ENABLED" == "true" ]; then
                    profiler_dirs_after=$(list_profiler_dirs) # Get the current folders under $LOG_DIR. This time will have another torch profiler folder
                    echo ">>> Processing profiler traces..."
                    if prof_cmd_has_profile_by_stage; then
                        rename_profiler_artifacts_by_stage "${input_tokens}" "${output_tokens}" "${c}" "${num_prompts}" "${profiler_dirs_before}" "${profiler_dirs_after}" \
                            || echo "[warn] rename_profiler_artifacts_by_stage failed for ${logfile}; raw trace dir left in place — continuing."
                    else
                        rename_profiler_artifacts "${input_tokens}" "${output_tokens}" "${c}" "${num_prompts}" "${profiler_dirs_before}" "${profiler_dirs_after}" \
                            || echo "[warn] rename_profiler_artifacts failed for ${logfile}; raw trace dir left in place — continuing."
                    fi
                fi
            fi
        done
    done
}


# ===================== Package Setup =====================
if ! is_rocm_gpu_env; then
    export SGL_ENABLE_JIT_DEEPGEMM=1
    # NV image ships a PEP 668 "externally managed" Python; install the
    # missing 'distro' dep (imported by sglang.bench_serving) with the
    # override flag. ROCm image already has it, so skip there.
    if ! python3 -c "import distro" >/dev/null 2>&1; then
        python3 -m pip install --user --break-system-packages distro
    fi
    # GLM-5.2 (nvidia/GLM-5.2-NVFP4) uses the glm_moe_dsa architecture, which the
    # transformers pinned in lmsysorg/sglang images (==5.8.1) is too old to load
    # ("layer_types entries must be in ... deepseek_sparse_attention"). Per the NV
    # model card (https://huggingface.co/nvidia/GLM-5.2-NVFP4), upgrade transformers
    # before launching. (--break-system-packages: same PEP 668 override as above.)
    # Gated to GLM-5.2 only so other models (GLM-5 / GLM-5.1) keep the pinned version.
    #
    # IMPORTANT: only upgrade if transformers is actually too old (< 5.3.0). Do NOT
    # blindly `pip install -U`: the purpose-built dev-glm52 image already ships a
    # compatible transformers (e.g. 5.12.1), and forcing it to the newest (5.13.0)
    # pulls in a version whose built-in `qwen3_asr` config collides with sglang's
    # own qwen3_asr registration ("'qwen3_asr' is already used by a Transformers
    # config"), which kills server startup at import time.
    #
    # GLM-5.3-Flash (glm5_next) declares transformers_version 5.16.0 in its
    # config.json, so it needs a newer floor than GLM-5.2.
    _min_transformers=""
    case "${MODEL_NAME}" in
        *GLM-5.3*) _min_transformers="5.16.0" ;;
        *GLM-5.2*) _min_transformers="5.3.0" ;;
    esac
    # Compare on base_version so a pre-release satisfies its own floor: the
    # glm-5.3-flash image ships transformers 5.16.0.dev0, and a plain
    # Version() compare rates that BELOW 5.16.0 and would pip-install over the
    # purpose-built build — the exact clobbering this gate exists to prevent.
    if [ -n "${_min_transformers}" ]; then
        if python3 -c "import transformers, sys; from packaging.version import Version; sys.exit(0 if Version(Version(transformers.__version__).base_version) >= Version('${_min_transformers}') else 1)" 2>/dev/null; then
            echo "[info] transformers $(python3 -c 'import transformers; print(transformers.__version__)') already satisfies >=${_min_transformers}; skipping upgrade."
        else
            python3 -m pip install --break-system-packages "transformers>=${_min_transformers}"
        fi
    fi
fi

# ===================== sgl-eval (GSM8K harness for thinking models) =====================
# accuracy_harness() selects sgl-eval for GLM-5.3-Flash and quietly falls back to
# benchmark/gsm8k/bench_sglang.py when it is missing. That fallback scores the SAME
# server 6-25 points lower (71-90% vs 95.8%) because it is 5-shot raw completion with
# no thinking and a 512-token cap, so a run that takes it produces an accuracy number
# that compares to nothing. Install the harness here instead of relying on whoever
# built the container.
#
# Pinned to a231b7a4 because that is the revision behind the ~97% reference runs: it
# writes ns_commit_sha 645cf567ff08c0ae9cc3fc8e1edbb975b3067816 into metrics.json,
# which is the value the reference metrics.json carries. Do not try to install
# 645cf567 itself -- it is a field sgl-eval computes, not a fetchable ref, and pip
# fails on it with "not our ref".
#
#   SGL_EVAL_REF=pypi ./GLM.sh          # take the released wheel instead
#   SGL_EVAL_REF=<git-sha> ./GLM.sh     # some other revision
#   SKIP_SGL_EVAL_INSTALL=1 ./GLM.sh    # leave the environment alone
install_sgl_eval() {
    if [ "${SKIP_SGL_EVAL_INSTALL:-0}" = "1" ] || [ "${SKIP_GSM8K:-0}" = "1" ]; then
        return 0
    fi
    # Only models whose harness is sgl-eval; ACCURACY_HARNESS forces either way.
    # Keep this arm identical to accuracy_harness()'s -- if the two disagree, one
    # platform installs the harness and the other silently grades with the wrong one.
    if [ "${ACCURACY_HARNESS:-}" != "sgl-eval" ]; then
        if [ -n "${ACCURACY_HARNESS:-}" ]; then
            return 0
        fi
        case "${MODEL_NAME}" in
            *GLM-5.3*) ;;
            *) return 0 ;;
        esac
    fi
    if command -v sgl-eval >/dev/null 2>&1; then
        echo "[info] sgl-eval already present at $(command -v sgl-eval); skipping install."
        return 0
    fi

    local ref="${SGL_EVAL_REF:-a231b7a439b235090ff7baa30778fa2b514309ae}"
    local spec="git+https://github.com/sgl-project/sgl-eval.git@${ref}"
    [ "$ref" = "pypi" ] && spec="sgl-eval"
    echo ">>> Installing GSM8K harness: ${spec}"
    # Widen only as far as each image needs: plain works on the venv-based ROCm
    # images, --break-system-packages is for the NV images' PEP 668 "externally
    # managed" Python, --user for a container that is not running as root.
    if ! python3 -m pip install "$spec" \
        && ! python3 -m pip install --break-system-packages "$spec" \
        && ! python3 -m pip install --user --break-system-packages "$spec"; then
        echo "[warn] sgl-eval install failed. accuracy_harness() will fall back to" \
             "bench_sglang.py, whose score is NOT comparable to the published" \
             "GLM-5.3-Flash numbers. Re-run with SKIP_GSM8K=1 to skip accuracy entirely."
        return 0
    fi
    # --user drops the console script in ~/.local/bin, which is not on PATH in
    # these images -- without this the harness would still silently fall back.
    if ! command -v sgl-eval >/dev/null 2>&1 && [ -x "${HOME}/.local/bin/sgl-eval" ]; then
        export PATH="${HOME}/.local/bin:${PATH}"
    fi
    if command -v sgl-eval >/dev/null 2>&1; then
        echo "[info] sgl-eval installed at $(command -v sgl-eval)."
    else
        echo "[warn] pip reported success but 'sgl-eval' is not on PATH."
    fi
}
install_sgl_eval


# ===================== GPU clock sanity check (NVIDIA only) =====================
# A locked application clock silently invalidates every number this script
# produces, and nothing else in the logs reveals it. Measured on dgx-029
# 2026-07-31: all 8 B200 pinned to 1005 MHz vs a 1965 MHz driver default cost
# 30-60% (GLM-5.2-NVFP4 TP4 i8k/conc64 1586 -> 1082 output tok/s) while drawing
# only ~575W of a 1000W budget at 45-52C -- so SW Power Cap / HW Slowdown /
# Thermal all read "Not Active" and the run looks like a software regression.
# Memory clock stays at default, which is why decode (bandwidth-bound) loses
# ~30% but prefill (compute-bound) loses ~60%.
# Abort rather than emit bad data. Reset the clocks with `nvidia-smi -rac`
# (needs root / privileged container); note that reset does NOT persist across
# driver reload or reboot. Bypass this check with ALLOW_LOCKED_CLOCKS=1.
check_gpu_clocks() {
    is_rocm_gpu_env && return 0
    command -v nvidia-smi >/dev/null 2>&1 || return 0

    local gpu_list locked
    gpu_list=$(seq -s, 0 $((TP_SIZE - 1)))
    # Compare per-GPU application clock against the driver default. Skip rows
    # where either value is non-numeric ("[N/A]" on GPUs that don't expose it).
    locked=$(nvidia-smi --query-gpu=index,clocks.applications.graphics,clocks.default_applications.graphics \
        --format=csv,noheader,nounits -i "$gpu_list" 2>/dev/null \
        | awk -F', ' '$2 ~ /^[0-9]+$/ && $3 ~ /^[0-9]+$/ && $2 < $3 {print $1" "$2" "$3}') || true
    [ -z "$locked" ] && return 0

    echo "!!! ERROR: GPU application clocks are locked below the driver default:"
    echo "$locked" | awk '{printf "      GPU %s: %s MHz  (default %s MHz)\n", $1, $2, $3}'
    echo "    Throughput would come out 30-60% low with no throttle flag set."
    echo "    Fix: nvidia-smi -rac    (root / privileged container)"
    echo "    Override: ALLOW_LOCKED_CLOCKS=1 ./GLM.sh ..."
    return 1
}
if [ "${ALLOW_LOCKED_CLOCKS:-0}" != "1" ]; then
    check_gpu_clocks || exit 1
fi

shapes_complete() {
    # True (0) if every (shape, concurrency) in the CURRENT in_out_tokens already
    # has output on disk, for the current mode (LOG_DIR / PROMPT_MULTIPLIER /
    # FINISH_LOG already set). Lets a re-run skip a whole server launch — warmup
    # and GSM8K included — and jump straight to the work that is still missing.
    local io input_tokens output_tokens c num_prompts
    for io in "${in_out_tokens[@]}"; do
        IFS=":" read -r input_tokens output_tokens <<< "$io"
        for c in "${concurrencies[@]}"; do
            num_prompts=$((c * PROMPT_MULTIPLIER))
            if [ "$PROF_ENABLED" == "true" ]; then
                [ -d "${LOG_DIR}/prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}${COMBINED_SUFFIX}" ] || return 1
            else
                grep -q "${LOG_DIR}/bench_in${input_tokens}_out${output_tokens}_conc${c}.log" "$FINISH_LOG" 2>/dev/null || return 1
            fi
        done
    done
    return 0
}

prof_mode_complete() {
    # Whole-mode version of the above; only meaningful in --prof mode (e.g.
    # cuda-graph already profiled -> go directly to no-cuda-graph).
    [ "$PROF_ENABLED" == "true" ] || return 1
    shapes_complete
}

build_shape_groups() {
    # Partition in_out_tokens into groups of shapes that can share one server,
    # keyed by the server args they need (see SHAPE_SERVER_ARGS). Groups run in
    # first-seen order and unlisted shapes all land in one group, so the extra
    # model load only happens when a listed shape is actually in the sweep — but
    # shapes can be reordered relative to in_out_tokens to keep a group together.
    # Sets _group_args[] (args string) and _group_shapes[] (space-separated shapes).
    _group_args=()
    _group_shapes=()
    local io args i found
    for io in "${in_out_tokens[@]}"; do
        args="${SHAPE_SERVER_ARGS[$io]:-}"
        found=-1
        for i in "${!_group_args[@]}"; do
            if [ "${_group_args[$i]}" = "$args" ]; then found=$i; break; fi
        done
        if [ "$found" -lt 0 ]; then
            _group_args+=("$args")
            _group_shapes+=("$io")
        else
            _group_shapes[$found]="${_group_shapes[$found]} $io"
        fi
    done
}

stop_server() {
    # Graceful stop (SIGTERM). Do NOT pkill -9 GPU server procs on ROCm: a hard
    # kill can trigger 100-200GB gpucore dumps that fill the shared disk.
    pkill -TERM -f sglang.launch_server || true
    for _i in $(seq 1 40); do pgrep -f sglang.launch_server >/dev/null 2>&1 || break; sleep 3; done
    pkill -TERM -f "sglang::" || true
    sleep 8
    # The listening socket outlives the launcher (it is held by a tokenizer
    # worker), and start_server's preflight aborts the run if the port is still
    # bound — so when another server is about to start, wait on the port itself.
    for _i in $(seq 1 30); do
        (exec 3<>"/dev/tcp/${HOST}/${PORT}") 2>/dev/null || break
        sleep 2
    done
}

# ------------------- Start -----------------
if [ "$PROF_ENABLED" == "true" ]; then
    PROF_SERVER_MODES=("default" "no-cuda-graph")
else
    PROF_SERVER_MODES=("default")
fi
# Space-separated override, e.g. to skip the eager pass when only the captured
# traces are wanted (the no-cuda-graph i70k/conc64 cell is the most expensive one
# in the sweep and the likeliest to OOM):
#   PROF_SERVER_MODES_OVERRIDE="default" ./GLM.sh --prof ...
if [ -n "${PROF_SERVER_MODES_OVERRIDE:-}" ]; then
    read -ra PROF_SERVER_MODES <<< "$PROF_SERVER_MODES_OVERRIDE"
fi

BASE_LOG_DIR="$LOG_DIR"

for PROF_MODE in "${PROF_SERVER_MODES[@]}"; do
    EXTRA_SERVER_ARGS=()
    NOGRAPH_SUFFIX=""
    if [ "$PROF_MODE" == "no-cuda-graph" ]; then
        EXTRA_SERVER_ARGS=(--disable-cuda-graph)
        # Eager (no-graph) forward allocates transient activation/workspace
        # memory that the captured-graph path doesn't; at large batch (conc64)
        # this can OOM/IMA -> segfault. Give ~0.1 more headroom by lowering
        # mem-fraction-static for the no-graph server only. Later --mem-fraction-static
        # wins in argparse. Override the delta with NOGRAPH_MEM_FRACTION_DELTA.
        _nograph_mfs=$(awk "BEGIN{v=$MEM_FRACTION_STATIC-${NOGRAPH_MEM_FRACTION_DELTA:-0.1}; if(v<0.1)v=0.1; printf \"%.2f\", v}")
        EXTRA_SERVER_ARGS+=(--mem-fraction-static "$_nograph_mfs")
        NOGRAPH_SUFFIX="-NoGraph"
        PROMPT_MULTIPLIER=1
        LOG_DIR="${BASE_LOG_DIR}/no-cuda-graph"
        mkdir -p "$LOG_DIR"
        FINISH_LOG="$LOG_DIR/Finish.log"
        touch "$FINISH_LOG"
        export SGLANG_TORCH_PROFILER_DIR=$LOG_DIR
    else
        LOG_DIR="${BASE_LOG_DIR}"
    fi

    # Smart skip: if this mode's profile dirs already exist, don't launch the
    # server / warmup / GSM8K at all — jump to the next mode. (e.g. cuda-graph
    # profiling already done -> go straight to no-cuda-graph.)
    if prof_mode_complete; then
        echo ">>> [${PROF_MODE}] all profile dirs already present under '${LOG_DIR}' — skipping server launch / warmup / gsm8k for this mode."
        continue
    fi

    # Run each profiling mode inside a subshell so any fatal error — server
    # death (start_server's `exit 1`), profiler teardown crash, etc. — only
    # aborts THIS mode rather than the whole script. The next mode (e.g.
    # no-cuda-graph) still runs, and the cleanup below always executes.
    (
        build_shape_groups
        for _gi in "${!_group_args[@]}"; do
            # run_benchmarks / shapes_complete read in_out_tokens, so scope it
            # down to this group's shapes for the lifetime of this server.
            read -ra in_out_tokens <<< "${_group_shapes[$_gi]}"
            SHAPE_EXTRA_SERVER_ARGS="${_group_args[$_gi]}"
            _tag="${PROF_MODE}"
            if [ "${#_group_args[@]}" -gt 1 ]; then
                _tag="${PROF_MODE} server $((_gi + 1))/${#_group_args[@]}"
                echo ">>> [${_tag}] shapes: ${in_out_tokens[*]} | extra server args: ${SHAPE_EXTRA_SERVER_ARGS:-<none>}"
            fi

            if shapes_complete; then
                echo ">>> [${_tag}] all results already present under '${LOG_DIR}' — skipping server launch / warmup / gsm8k."
                continue
            fi

            # Each group runs in its own subshell so a fatal error (start_server's
            # `exit 1`, a profiler teardown crash) only aborts THIS group; the
            # stop_server below still runs, so the next group gets a free port.
            (
                echo ">>> [${_tag}] Starting server and benchmarks..."
                start_server
                warmup
                # Accuracy is shape-independent, so only the first server runs it.
                if [ "$PROF_MODE" == "default" ] && [ "$_gi" -eq 0 ]; then
                    accuracy_test
                fi
                run_benchmarks
            ) || echo "[warn] shape group '${in_out_tokens[*]}' aborted (exit $?); continuing."
            stop_server
        done
    ) || echo "[warn] profiling mode '${PROF_MODE}' aborted (exit $?); continuing to cleanup and next mode."

    echo "[${PROF_SERVER_MODES[@]}], now is the end of ${PROF_MODE}"
    stop_server
done

set +x
echo ">>> Results: ${BASE_LOG_DIR}"














# GLM-4.7 is not a VLM.
# MMMU test
# TOKEN_LIST=(512 1024 2048 4096 8192 16384 131072)
# TOKEN_LIST=(16384 131072)
# ITERATIONS=3
# mkdir -p "$LOG_DIR/MMMU"
# for iter in $(seq 1 $((ITERATIONS))); do
#     for token in "${TOKEN_LIST[@]}"; do
#         mmmu_logfile="$LOG_DIR/MMMU/MMMU_Token${token}_Iter${iter}.log"
#         mmmu_cmd=(
#             python3 /sgl-workspace/sglang/benchmark/mmmu/bench_sglang.py  
#                 --port "$PORT" 
#                 --concurrency 900 
#                 --parallel 900 
#                 --temperature 0
#                 --max-new-tokens "$token" 
#                 --result-file "$LOG_DIR/MMMU/MMMU_Token${token}_ResultFile_Iter${iter}.jsonl"
#                 --raw-result-file "$LOG_DIR/MMMU/MMMU_Token${token}_RawResultFile_Iter${iter}.jsonl"
#         )
        
#         if ! grep -q "$mmmu_logfile" "$FINISH_LOG" 2>/dev/null; then
#             echo ">>> Running MMMU: Token $token, Iteration $iter"

#             start_time=$(date +%s)
#             log_command "$mmmu_logfile" "${mmmu_cmd[@]}"
#             end_time=$(date +%s)
#             elapsed=$((end_time - start_time))
#             formatted_time=$(date -u -d "@$elapsed" +"%H:%M:%S")
#             echo "-------------------------------------------" >> "$mmmu_logfile"
#             echo "Execution Time $formatted_time (HH:MM:SS)" >> "$mmmu_logfile"

#             echo "$mmmu_logfile" >> "$FINISH_LOG"
#             mv $CURRENT_DIR/answer_sglang.json $LOG_DIR/MMMU/MMMU_Token${token}_answer_sglang_Iter${iter}.json
#         else
#             echo "Found $mmmu_logfile in Finish.log. Skipping."
#         fi
#     done
# done
# # stop_server
# echo ">>> All configurations completed. Logs: ${LOG_DIR}. Stop server..."
# pkill -9 python || true
# sleep 10
