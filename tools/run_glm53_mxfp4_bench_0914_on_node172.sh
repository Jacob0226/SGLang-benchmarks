#!/usr/bin/env bash
# Same-machine control: the 0914 ROCm 7.2 image on node crsuse2-m2m-172, the box
# the 0922 numbers came from.
#
# Why this run exists. The 0914 numbers were taken on crsuse2-m2m-032 (job
# 147020, since expired) and the 0922 numbers on crsuse2-m2m-172, so the ~9%
# conc4 decode gap between them is image AND machine confounded and the earlier
# node cannot be revisited. Running the old image here separates the two:
#   ~9.2 ms TPOT at i1k/conc4  -> the gap is the image (sglang 0.5.20 / triton 3.8)
#   ~10.0 ms                   -> the gap is the box, and 0922 is not a regression
#
# Everything except the image matches the 0922 control: same checkpoint (config
# .json verified byte-identical), same ten-PR stack but based on THIS image's own
# main snapshot (242d8a70c0), same shapes, same concurrencies, fusion off.
#
# Its own JIT cache: same AITER revision as 0922 (4ad99832) but a different ROCm,
# so the built objects are not interchangeable.
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

export IN_OUT_OVERRIDE="${IN_OUT_OVERRIDE:-1024:1024 8192:1024}"
export CONC_OVERRIDE="${CONC_OVERRIDE:-4 64}"
export SKIP_GSM8K="${SKIP_GSM8K:-1}"

echo "=== tree   : $(git -C "$SGLANG_SRC" rev-parse --short HEAD 2>/dev/null || echo '?') ==="
echo "=== sglang : $(python3 -c 'import inspect,sglang;print(inspect.getfile(sglang))') ==="
echo "=== aiter  : $(git -C /sgl-workspace/aiter rev-parse --short HEAD 2>/dev/null || echo '?') ==="
echo "=== rocm   : $(cat /opt/rocm/.info/version 2>/dev/null) ==="
echo "=== triton : $(python3 -c 'import triton;print(triton.__version__)' 2>/dev/null) ==="

cd "$HOME/SGLang-benchmarks/BenchFixedLength"
./GLM.sh \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260914-onNode172 \
    --tag "${TAG:-MXFP4-TP4-node172}"
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="
