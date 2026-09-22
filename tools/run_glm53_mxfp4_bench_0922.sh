#!/usr/bin/env bash
# Baseline bench of the Day-0 PR stack on the 0922 ROCm 10 image, to check the
# image jump for regressions before changing anything else.
#
# Deliberately one variable against yesterday's 0914 baseline arm
# (results/.../rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/
#  bench-Fixed-MXFP4-TP4-SharedFusion-baseline): same checkpoint (config.json
# verified byte-identical), same shapes, same concurrencies, same harness, and
# shared-experts fusion off on both -- this tree is the plain ten-PR stack, whose
# CUDA-only gate keeps the fusion disabled without needing the server flag.
#
# What moved: ROCm 7.2 -> 10.0.0, torch -> 2.11.0+rocm10, triton 3.7 -> 3.8.0,
# sglang 242d8a70c0 -> 50ec9702d0 (the image's own main snapshot). AITER did NOT
# move (4ad99832 in both images).
set -uo pipefail

SGLANG_SRC="${SGLANG_SRC:-/home/jacchang/PR/glm53-day0-stack-0922}"
export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=1

# New image, new AITER build products -> its own cache directory. Sharing the
# 0914 one would load .so files built against ROCm 7.2.
RUN_CACHE="${RUN_CACHE:-/home/jacchang/SGLang-benchmarks/tmp/cache-glm53-day0-stack-0922}"
mkdir -p "$RUN_CACHE"
export AITER_JIT_DIR="$RUN_CACHE/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang"
export TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg"
export SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

export IN_OUT_OVERRIDE="${IN_OUT_OVERRIDE:-1024:1024 8192:1024}"
export CONC_OVERRIDE="${CONC_OVERRIDE:-4 64}"
export SKIP_GSM8K="${SKIP_GSM8K:-0}"

echo "=== tree   : $(git -C "$SGLANG_SRC" rev-parse --short HEAD 2>/dev/null || echo '?') ==="
echo "=== sglang : $(python3 -c 'import inspect,sglang;print(inspect.getfile(sglang))') ==="
echo "=== aiter  : $(git -C /sgl-workspace/aiter rev-parse --short HEAD 2>/dev/null || echo '?') ==="

cd "$HOME/SGLang-benchmarks/BenchFixedLength"
./GLM.sh \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260922 \
    --tag "${TAG:-MXFP4-TP4-PRstack}"
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="
