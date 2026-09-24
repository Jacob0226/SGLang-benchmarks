#!/usr/bin/env bash
# Baseline benchmark of GLM-5.3-Flash Quark-MXFP4 on the 0923 ROCm 10 image,
# i8k + i70k at conc 4 and 64, with the current GLM.sh and nothing overridden.
#
# This is the first full baseline on this image, and it needs to exist before any
# optimisation number here means anything: every earlier MI355X figure was taken
# on the 0914 image with AITER 4ad99832, and this one ships acf8fdf93.
#
# Uses the image's own in-tree sglang -- all ten GLM-5.3-Flash Day-0 PRs are
# merged into abef3efb64 (tools/glm53_check_image_prs.sh), so there is no
# PYTHONPATH tree and no PR stack to apply.
#
# GLM.sh is taken as-is, which now means: no SGLANG_DSA_FUSE_HADAMARD_QUANT (the
# variable does not exist in this sglang) and no forced SGLANG_OPT_USE_TOPK_V2
# (sglang's own model hook defaults it False for DSA on HIP).
#
# All four shapes share one server: GLM-5.3-Flash defaults I70K_MAX_RUNNING_REQUESTS
# to 0, so i70k needs no separate --max-running-requests launch.
#
# Output: SGLang-benchmarks/tmp/logs/bench_0923_routercfg.log
set -uo pipefail
LOGS=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$LOGS"
exec > "$LOGS/bench_0923_routercfg.log" 2>&1

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SGLANG_USE_AITER=1

# Same cache the knob A/Bs used, so the i8k shapes are already compiled and only
# i70k pays JIT. Kept deliberately: a cold cache would put compile time inside
# the first cell's TTFT.
RUN_CACHE=/home/jacchang/SGLang-benchmarks/tmp/cache-glm53-0923-rocm10
mkdir -p "$RUN_CACHE"
export AITER_JIT_DIR="$RUN_CACHE/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang"
export TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg"
export SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

export IN_OUT_OVERRIDE="${IN_OUT_OVERRIDE:-8192:1024}"
export CONC_OVERRIDE="${CONC_OVERRIDE:-4 64}"

cd /home/jacchang/SGLang-benchmarks/BenchFixedLength
./GLM.sh \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260923 \
    --tag "${TAG:-MXFP4-TP4-0923-routercfg}"
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="

echo
echo "================ summary ================"
R=/home/jacchang/SGLang-benchmarks/results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.20-rocm10-mi35x-20260923/bench-Fixed-${TAG:-MXFP4-TP4-0923-routercfg}
printf '%-24s %10s %10s %9s %9s %11s %12s\n' cell TTFT_ms P99TTFT TPOT_ms ITL_ms out_tok_s total_tok_s
for f in "$R"/bench_*.log; do
    [ -f "$f" ] || continue
    printf '%-24s %10s %10s %9s %9s %11s %12s\n' \
        "$(basename "$f" .log | sed 's/^bench_//')" \
        "$(grep -a 'Median TTFT' "$f" | awk '{print $4}')" \
        "$(grep -a 'P99 TTFT'    "$f" | awk '{print $4}')" \
        "$(grep -a 'Median TPOT' "$f" | awk '{print $4}')" \
        "$(grep -a 'Median ITL'  "$f" | awk '{print $4}')" \
        "$(grep -a 'Output token throughput' "$f" | awk '{print $5}')" \
        "$(grep -a 'Total token throughput'  "$f" | awk '{print $5}')"
done
echo
grep -ah '\[gsm8k\]' "$R"/../bench-Fixed-*/Accuracy_GSM8K.log 2>/dev/null | tail -1
grep -ah '\[gsm8k\]' "$LOGS/bench_0923_routercfg.log" | tail -1
