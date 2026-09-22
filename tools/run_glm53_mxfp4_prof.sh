#!/usr/bin/env bash
# GLM-5.3-Flash Quark-MXFP4 profile run: i8k + i70k at concurrency 4 and 64,
# captured-graph pass only (PROF_SERVER_MODES_OVERRIDE=default).
#
# The image's in-tree sglang cannot load this checkpoint -- it dies during weight
# load with a bare `assert self.data.shape == loaded_weight.shape` in
# load_row_parallel_weight, which is the failure GLM53-Flash-ROCm-PrStack_Repro.md
# describes: the Quark checkpoint mixes MXFP4 with block-FP8 and the loading work
# lives in #38546 + #39317. So serve from the Day-0 PR stack via PYTHONPATH and
# leave the image's editable install alone, the same way glm53_mxfp4_verify.sh does.
#
# SGLANG_SRC = main @ 242d8a70c0 (the image's own snapshot) + the ten Day-0 PRs:
#   #39338 #39339 #39340 #39341 #38545 #38546 #38547 #39317 #39778 #39779
# Verified before launch with tools/glm53_check_stack.sh (ALL PASS).
#
# Launched detached inside the jacchang_GLM53-Flash container; log lands in
# ~/glm53_mxfp4_prof.log.
set -uo pipefail

SGLANG_SRC="${SGLANG_SRC:-/home/jacchang/PR/glm53-day0-stack}"
export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=1

# One JIT cache directory per (tree, image). AITER keys its JIT products by
# module name alone, so a directory shared with another tree silently loads
# binaries built from the other source. Kept across server launches within this
# run -- a cold cache costs ~6 min of AITER/TileLang/Triton compilation.
RUN_CACHE="${RUN_CACHE:-/home/jacchang/SGLang-benchmarks/tmp/cache-glm53-day0-stack-0921}"
mkdir -p "$RUN_CACHE"
export AITER_JIT_DIR="$RUN_CACHE/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang"
export TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg"
export SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

echo "=== tree   : $(git -C "$SGLANG_SRC" rev-parse --short HEAD 2>/dev/null || echo '?') ==="
echo "=== sglang : $(python3 -c 'import inspect,sglang;print(inspect.getfile(sglang))') ==="
echo "=== aiter  : $(git -C /sgl-workspace/aiter rev-parse HEAD) ==="

cd "$HOME/SGLang-benchmarks/BenchFixedLength"

export PROF_IN_OUT_OVERRIDE="8192:16 70000:16"
export PROF_CONC_OVERRIDE="4 64"
# "default" = captured cuda graph, "no-cuda-graph" = eager layer-structure traces.
# Override from the caller to run the other pass into the same results dir.
export PROF_SERVER_MODES_OVERRIDE="${PROF_SERVER_MODES_OVERRIDE:-default}"
export SKIP_GSM8K=1

./GLM.sh --prof \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260914 \
    --tag MXFP4-TP4-PRstack
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="
