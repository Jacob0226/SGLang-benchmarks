#!/usr/bin/env bash
# Launch GLM-5.2-MXFP4 (TP4, MI355X) for DSA attention-backend comparison
# (tilelang "main_kernel" vs triton sparse-MLA from PR #30575). Runs from an
# arbitrary host-side sglang checkout via PYTHONPATH (no reinstall).
#
# Usage (inside container):
#   SGL_REPO=/home/jacchang/PR/sglang-triton DSA_BACKEND=tilelang PORT=8236 tools/dsa_backend_bench_server.sh
#   SGL_REPO=/home/jacchang/PR/sglang-triton DSA_BACKEND=triton   PORT=8236 tools/dsa_backend_bench_server.sh
#
# Env: SGL_REPO (repo root, uses $SGL_REPO/python), DSA_BACKEND (tilelang|triton),
#      PORT (8236), TP (4), MODEL, PROFILE_DIR (torch profiler output dir).
set -euo pipefail
set -x

SGL_REPO="${SGL_REPO:-/home/jacchang/PR/sglang}"
export PYTHONPATH="${SGL_REPO}/python"
export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
export SGLANG_OPT_USE_TOPK_V2=0

DSA_BACKEND="${DSA_BACKEND:-tilelang}"
PORT="${PORT:-8236}"
TP="${TP:-4}"
MODEL="${MODEL:-/data/huggingface/hub/amd/GLM-5.2-MXFP4}"
LOGDIR="${LOGDIR:-/home/jacchang/SGLang-benchmarks/tmp}"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/dsa_bench_${DSA_BACKEND}_port${PORT}.log"
if [ -n "${PROFILE_DIR:-}" ]; then
  mkdir -p "$PROFILE_DIR"
  export SGLANG_TORCH_PROFILER_DIR="$PROFILE_DIR"
fi

cmd=(
  python3 -m sglang.launch_server
    --model "$MODEL"
    --tp "$TP"
    --host localhost
    --port "$PORT"
    --trust-remote-code
    --tool-call-parser glm47
    --reasoning-parser glm45
    --watchdog-timeout 1200
    --mem-fraction-static 0.85
    --kv-cache-dtype fp8_e4m3
    --disable-radix-cache
    --dsa-prefill-backend "$DSA_BACKEND"
    --dsa-decode-backend "$DSA_BACKEND"
    --enable-aiter-allreduce-fusion
    --chunked-prefill-size 16384
)

echo ">>> DSA backend=$DSA_BACKEND PORT=$PORT REPO=$SGL_REPO" | tee "$LOGFILE"
echo "${cmd[*]}" | tee -a "$LOGFILE"
exec "${cmd[@]}" 2>&1 | tee -a "$LOGFILE"
