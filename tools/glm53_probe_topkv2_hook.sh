#!/usr/bin/env bash
# What does sglang itself decide about SGLANG_OPT_USE_TOPK_V2 for this model, and
# does GLM.sh's blanket export stop it from deciding?
#
# model_hook.py reads `not envs.SGLANG_OPT_USE_TOPK_V2.is_set()` before
# overriding, so an export -- even export=0 -- takes the decision away from the
# code that knows the model. Worth reading all four sites before changing GLM.sh.
#
# Output: SGLang-benchmarks/tmp/logs/topkv2_hook.txt
OUT=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$OUT"
exec > "$OUT/topkv2_hook.txt" 2>&1
SRC=${SRC:-/sgl-workspace/sglang}

echo "=== the four sites that set it, with context ==="
for ln in 321 403 450 469; do
    echo "--- around line $ln"
    sed -n "$((ln-18)),$((ln+6))p" "$SRC/python/sglang/srt/arg_groups/model_hook.py"
    echo
done

echo "=== is the ROCm topk_v2 kernel CUDA-only in disguise? ==="
h="$SRC/python/sglang/kernels/jit/include/sgl_kernel/deepseek_v4/topk_impl.cuh"
echo "--- how cooperative_groups / clusters are guarded:"
grep -n -B3 -A3 "cooperative_groups\|this_cluster\|__HIP\|USE_ROCM\|HIP_PLATFORM" "$h" | head -40
