#!/bin/bash
# Launch the router-GEMM microbenchmark inside the container (detached).
LOG=/home/jacchang/SGLang-benchmarks/tmp/logs/router_gemm_bench.log
mkdir -p "$(dirname "$LOG")"

export HIP_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/jacchang/SGLang-benchmarks/tmp/cache-router-gemm
export AITER_LOG_MORE=0
mkdir -p "$TRITON_CACHE_DIR"

cd /sgl-workspace
python -u /home/jacchang/SGLang-benchmarks/tools/glm53_bench_router_gemm.py > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
chmod -R a+rw /home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm 2>/dev/null
chmod a+rw "$LOG" 2>/dev/null
