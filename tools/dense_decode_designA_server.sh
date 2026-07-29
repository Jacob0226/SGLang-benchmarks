#!/usr/bin/env bash
# Launch GLM-5.2-MXFP4 (TP4, MI355X/tilelang) for Design A dense-decode
# dual-graph validation. Runs the SGLang server from the HOST-mounted fork
# checkout (~/PR/sglang, branch jacob/dense-decode-konly) via PYTHONPATH so
# host-side edits take effect with no reinstall.
#
# Usage (inside the container):
#   DUAL=1 tools/dense_decode_designA_server.sh   # Design A dual dense/sparse graphs
#   DUAL=0 tools/dense_decode_designA_server.sh   # baseline (single sparse graph)
#   DENSE_STATIC=1 tools/dense_decode_designA_server.sh  # M2a static k-only graph
#
# Env knobs: PORT (default 8234), MODEL, TP (default 4), MAX_BS (cuda-graph-max-bs).
set -euo pipefail
set -x

export PYTHONPATH=/home/jacchang/PR/sglang/python
export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
# 0708 image: topk_v2 JIT needs cooperative_groups.h (absent in ROCm 7.2).
export SGLANG_OPT_USE_TOPK_V2=0

DUAL="${DUAL:-1}"
DENSE_STATIC="${DENSE_STATIC:-0}"
export SGLANG_DSA_DECODE_DUAL_GRAPH="${DUAL}"
export SGLANG_DSA_DECODE_DENSE_GRAPH="${DENSE_STATIC}"

PORT="${PORT:-8234}"
TP="${TP:-4}"
MODEL="${MODEL:-/data/huggingface/hub/amd/GLM-5.2-MXFP4}"
LOGDIR="${LOGDIR:-/home/jacchang/SGLang-benchmarks/tmp}"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/designA_server_dual${DUAL}_static${DENSE_STATIC}_port${PORT}.log"

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
    --dsa-prefill-backend tilelang
    --dsa-decode-backend tilelang
    --enable-aiter-allreduce-fusion
    --chunked-prefill-size 16384
)
if [ -n "${MAX_BS:-}" ]; then
  cmd+=(--cuda-graph-max-bs "$MAX_BS")
fi

echo ">>> Design A server: DUAL=$DUAL DENSE_STATIC=$DENSE_STATIC PORT=$PORT" | tee "$LOGFILE"
echo "${cmd[*]}" | tee -a "$LOGFILE"
exec "${cmd[@]}" 2>&1 | tee -a "$LOGFILE"
