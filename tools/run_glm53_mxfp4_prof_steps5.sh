#!/usr/bin/env bash
# Re-profile GLM-5.3-Flash Quark-MXFP4 (no MTP) with 5 captured steps, to pair
# with the B200 steps=5 capture.
#
# Why 5 and not the old 2: on ROCm the decode session's GPU-side tracing starts
# 19-70 ms after its CPU side, and a decode forward is 2.5-15 ms, so the first
# captured step has no GPU work at all -- every earlier MI355X decode trace held
# two step[DECODE bs=N] CPU annotations but one gpu_user_annotation and one
# forward's worth of kernels (91 collectives = 2 per layer x 45 + 1). At 2 steps
# that is half the sample. The steps that do survive are also the most disturbed
# ones, because ranks do not arm or flush their trace buffers in lockstep and
# the first collective after the perturbation absorbs the skew.
#
# Runs on the 0914 image (ROCm 7.2, in-tree sglang 242d8a70c0), which is what
# the ten-PR stack in /home/jacchang/PR/glm53-day0-stack is based on, so this is
# directly comparable to results/.../rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914
# -- but on node crsuse2-m2m-172 rather than the expired 032, hence the tag.
# The node costs ~1% at i1k/conc4 (9.26 vs 9.18 ms TPOT, measured 2026-09-23).
#
# No MTP: GLM.sh only enables it with --mtp, which is not passed.
set -uo pipefail

SGLANG_SRC="${SGLANG_SRC:-/home/jacchang/PR/glm53-day0-stack}"
export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=1

RUN_CACHE="${RUN_CACHE:-/home/jacchang/SGLang-benchmarks/tmp/cache-glm53-0914-on-node172}"
mkdir -p "$RUN_CACHE"
export AITER_JIT_DIR="$RUN_CACHE/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang"
export TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg"
export SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

export PROF_NUM_STEPS="${PROF_NUM_STEPS:-5}"
# Same shapes and concurrencies as the capture this one replaces, so the
# analysis flow needs no path edits.
export PROF_IN_OUT_OVERRIDE="${PROF_IN_OUT_OVERRIDE:-8192:16 70000:16}"
export PROF_CONC_OVERRIDE="${PROF_CONC_OVERRIDE:-4 64}"
export PROF_SERVER_MODES_OVERRIDE="${PROF_SERVER_MODES_OVERRIDE:-default no-cuda-graph}"
export SKIP_GSM8K="${SKIP_GSM8K:-1}"

echo "=== tree   : $(git -C "$SGLANG_SRC" rev-parse --short HEAD 2>/dev/null || echo '?') ==="
echo "=== sglang : $(python3 -c 'import inspect,sglang;print(inspect.getfile(sglang))') ==="
echo "=== aiter  : $(git -C /sgl-workspace/aiter rev-parse --short HEAD 2>/dev/null || echo '?') ==="
echo "=== rocm   : $(cat /opt/rocm/.info/version 2>/dev/null)  triton $(python3 -c 'import triton;print(triton.__version__)' 2>/dev/null) ==="
echo "=== steps  : ${PROF_NUM_STEPS} ==="

cd "$HOME/SGLang-benchmarks/BenchFixedLength"
./GLM.sh --prof \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260914 \
    --tag "${TAG:-MXFP4-TP4-PRstack-steps5-node172}"
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="
