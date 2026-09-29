#!/bin/bash
exec 200>/tmp/glm53_bench.lock
flock -n 200 || { echo "another GPU run holds the lock; refusing"; exit 0; }
LOG=/home/jacchang/SGLang-benchmarks/tmp/logs/default_knob_ablation.log
export HIP_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/jacchang/SGLang-benchmarks/tmp/cache-router-gemm-0928
cd /sgl-workspace
python -u /home/jacchang/SGLang-benchmarks/tools/glm53_ablate_default_knobs.py > "$LOG" 2>&1
echo "EXIT=$?" >> "$LOG"
chmod a+rw "$LOG" 2>/dev/null
