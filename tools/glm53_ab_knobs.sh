#!/usr/bin/env bash
# One-variable A/B of decode knobs on GLM-5.3-Flash MXFP4 TP4, i8k/conc4 only.
#
# Each arm is a fresh server because these knobs are read at import. Arms run
# back to back in one process so they share a JIT cache and the same GPU state;
# arm 0 is a throwaway that pays the cold-cache compile, and `base` is repeated
# at the end so drift over the run is visible rather than assumed.
#
# Uses the image's OWN sglang: every one of the ten Day-0 PRs is merged into
# rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260923 (tools/glm53_check_image_prs.sh),
# so there is no PYTHONPATH tree to maintain any more.
#
# Candidates, and why each one:
#   topkv2      GLM.sh forces SGLANG_OPT_USE_TOPK_V2=0 on ROCm because the
#               topk_v2 JIT kernel used to fail to build on ROCm 7.2 (missing
#               cooperative_groups.h). This image is ROCm 10 and the build
#               default is True, so the workaround may be obsolete.
#   jitgtopk    SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK is False by default and
#               targets aiter::grouped_topk_kernel, 0.247 ms over 42 launches in
#               the 0914 decode trace.
#
# Knobs deliberately NOT tested: fused gate+topk, the DSA indexer fusion and the
# k-pool metadata fusion are all already on by default in this build, so the
# glue kernels seen in the trace are what survives them, not something a flag
# turns off.
#
# Output: SGLang-benchmarks/tmp/logs/ab_knobs.log, results under
# results/<model>/<docker>/bench-Fixed-<ARM>/
set -uo pipefail
LOGS=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$LOGS"
exec > "$LOGS/ab_knobs.log" 2>&1

export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=1

# New image, new AITER revision (acf8fdf93, was 4ad99832) -> its own cache.
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

export IN_OUT_OVERRIDE="8192:1024"
export CONC_OVERRIDE="4"
export SKIP_GSM8K=1

DOCKER=rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260923
MODEL=/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4

run_arm() {                     # run_arm <tag> [VAR=VAL ...]
    local tag=$1; shift
    echo
    echo "################################################################"
    echo "######## arm: $tag   extra env: $*"
    echo "################################################################"
    (
        for kv in "$@"; do export "$kv"; done
        cd /home/jacchang/SGLang-benchmarks/BenchFixedLength
        ./GLM.sh --model "$MODEL" --tp 4 --docker "$DOCKER" --tag "$tag"
    )
    echo "######## arm $tag exited $? at $(date '+%F %T')"
}

# Arm 0 pays the cold compile; its numbers are not comparable to the rest.
run_arm prime
run_arm base
run_arm topkv2   SGLANG_OPT_USE_TOPK_V2=1
run_arm jitgtopk SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=1
# Repeat the baseline last: if base and base2 disagree, the run drifted and no
# single-arm delta from the middle is trustworthy.
run_arm base2

echo
echo "================ summary ================"
R=/home/jacchang/SGLang-benchmarks/results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.20-rocm10-mi35x-20260923
printf '%-12s %9s %9s %9s %11s\n' arm TPOT_ms ITL_ms TTFT_ms out_tok_s
for tag in prime base topkv2 jitgtopk base2; do
    f=$R/bench-Fixed-$tag/bench_in8192_out1024_conc4.log
    [ -f "$f" ] || { printf '%-12s %9s\n' "$tag" "(missing)"; continue; }
    printf '%-12s %9s %9s %9s %11s\n' "$tag" \
        "$(grep -a 'Median TPOT' "$f" | awk '{print $4}')" \
        "$(grep -a 'Median ITL'  "$f" | awk '{print $4}')" \
        "$(grep -a 'Median TTFT' "$f" | awk '{print $4}')" \
        "$(grep -a 'Output token throughput' "$f" | awk '{print $5}')"
done
