#!/usr/bin/env bash
# Profile i8k/conc4 decode on the 0923 image, to be diffed against the 0914
# capture kernel by kernel.
#
# The question: i8k/conc4 ITL went 9.04 -> 9.75 ms (+7.9%) between the 0914 and
# 0923 images while conc64 improved, and the same shape moved the same way on the
# 0922 image, so it is a persistent image effect rather than noise. A per-kernel
# delta says whether a kernel got slower or whether a different kernel was
# selected -- the two have completely different fixes.
#
# Graph-ON only, and one shape: ROCm yields exactly one usable decode forward per
# capture whatever PROF_NUM_STEPS says (HIP graph replay, see GLM.sh), so the
# eager pass would add trace bulk without adding decode samples, and it measures
# a different thing anyway (no replay, launch-bound).
#
# Output: SGLang-benchmarks/tmp/logs/prof_0923_routercfg.log
set -uo pipefail
LOGS=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$LOGS"
exec > "$LOGS/prof_0923_routercfg.log" 2>&1

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SGLANG_USE_AITER=1
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

# Match the 0914 steps5 capture exactly: same shape, same concurrency, same
# out=16, same number of requested steps.
export PROF_IN_OUT_OVERRIDE="8192:16"
export PROF_CONC_OVERRIDE="4"
export PROF_SERVER_MODES_OVERRIDE="default"
export PROF_NUM_STEPS=5
export SKIP_GSM8K=1

cd /home/jacchang/SGLang-benchmarks/BenchFixedLength
./GLM.sh --prof \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260923 \
    --tag "${TAG:-MXFP4-TP4-0923-routercfg-prof}"
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="
