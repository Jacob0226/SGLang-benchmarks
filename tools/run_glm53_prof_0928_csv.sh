#!/usr/bin/env bash
# Stock 0928 image profile at conc4 AND conc64.
#
# Two purposes:
#  1. Re-baseline on 20260928. Its AITER is still acf8fdf93 (#5414) and still
#     has no GEMM-A16W16-N=288-K=4096.json, so the router-GEMM fallback should
#     reproduce, but only sglang moved (0318a8d0af) and that needs confirming.
#  2. Answer whether the router GEMM even uses the triton _gemm_a16_w16_kernel
#     at M=64. tuned_gemm reports "torch solution:0" for N=288 at M=24..256 and
#     never mentions small M, which suggests the triton path is taken only for
#     small M -- if so, tuning M_LEQ_32/64 would do nothing for conc64.
set -uo pipefail
exec 200>/tmp/glm53_bench.lock
flock -n 200 || { echo "another run holds the lock; refusing"; exit 0; }

LOGS=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$LOGS"
exec > "$LOGS/prof_0928_csv.log" 2>&1

export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 SGLANG_USE_AITER=1
RUN_CACHE=/home/jacchang/SGLang-benchmarks/tmp/cache-glm53-0928-rocm10
mkdir -p "$RUN_CACHE"
export AITER_JIT_DIR="$RUN_CACHE/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang"
export TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg"
export SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

export PROF_IN_OUT_OVERRIDE="8192:16"
export PROF_CONC_OVERRIDE="4 64"
export PROF_SERVER_MODES_OVERRIDE="default"
export PROF_NUM_STEPS=5
export SKIP_GSM8K=1

cd /home/jacchang/SGLang-benchmarks/BenchFixedLength
./GLM.sh --prof \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928 \
    --tag "${TAG:-MXFP4-TP4-0928-csv-prof}"
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="
