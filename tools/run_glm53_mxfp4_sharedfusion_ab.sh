#!/usr/bin/env bash
# A/B the one speedup from the reverted #36607 that the Day-0 re-split never
# brought back: shared-experts fusion on AITER gfx95.
#
#   glm5_next.py still refuses it with "Shared experts fusion currently requires
#   CUDA devices"; #36607 widened that gate to `_is_cuda or _use_aiter_gfx95`.
#   The Quark MXFP4 checkpoint is safe for it -- its shared expert carries the
#   same fp4/per_group scheme and the same per-expert shapes as a routed one,
#   and it is not in the quark `exclude` list (tools/glm53_probe_shared_experts.sh).
#
# Both arms run the SAME patched tree and the SAME JIT cache, so the only
# difference is the server flag:
#   baseline = --disable-shared-experts-fusion  (identical to today's behaviour)
#   fusion   = the gate's own answer            (fused after the patch)
#
#   ARM=baseline|fusion run_glm53_mxfp4_sharedfusion_ab.sh
set -uo pipefail

ARM="${ARM:?set ARM=baseline|fusion}"
SGLANG_SRC="${SGLANG_SRC:-/home/jacchang/PR/glm53-day0-stack}"

export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=1

# Same cache as the profile run: same tree, same image, so the AITER/TileLang
# modules are already built and neither arm pays the cold-build cost.
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

if [ "$ARM" = "baseline" ]; then
    export SERVER_EXTRA_ARGS="--disable-shared-experts-fusion"
    TAG="MXFP4-TP4-SharedFusion-baseline"
else
    TAG="MXFP4-TP4-SharedFusion-on"
fi

# Decode-dominated shapes: the fusion removes a separate shared-expert GEMM plus
# its add from every MoE layer, so it should show up in TPOT rather than TTFT.
export IN_OUT_OVERRIDE="${IN_OUT_OVERRIDE:-1024:1024 8192:1024}"
export CONC_OVERRIDE="${CONC_OVERRIDE:-4 64}"
# Graded on both arms: fusion rewrites which tensor a shared expert's weights
# land in, so a wrong answer here is a loading bug, not a perf result.
export SKIP_GSM8K="${SKIP_GSM8K:-0}"

echo "=== arm    : $ARM ==="
echo "=== tree   : $(git -C "$SGLANG_SRC" rev-parse --short HEAD 2>/dev/null || echo '?') ==="
echo "=== sglang : $(python3 -c 'import inspect,sglang;print(inspect.getfile(sglang))') ==="
echo "=== extra  : ${SERVER_EXTRA_ARGS:-<none>} ==="

cd "$HOME/SGLang-benchmarks/BenchFixedLength"
./GLM.sh \
    --model /data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4 \
    --tp 4 \
    --docker rocm/sgl-dev:v0.5.19-rocm720-mi35x-20260914 \
    --tag "$TAG"
echo "=== GLM.sh exited with $? at $(date '+%F %T') ==="
