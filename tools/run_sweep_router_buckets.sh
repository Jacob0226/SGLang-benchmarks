#!/bin/bash
exec 200>/tmp/glm53_bench.lock
flock -n 200 || { echo "another GPU run holds the lock; refusing"; exit 0; }
LOG=/home/jacchang/SGLang-benchmarks/tmp/logs/router_bucket_sweep.log
mkdir -p "$(dirname "$LOG")"
export HIP_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/jacchang/SGLang-benchmarks/tmp/cache-router-gemm-0928
mkdir -p "$TRITON_CACHE_DIR"
cd /sgl-workspace
python -u /home/jacchang/SGLang-benchmarks/tools/glm53_sweep_router_buckets.py > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
chmod -R a+rw /home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928 2>/dev/null
chmod a+rw "$LOG" 2>/dev/null
