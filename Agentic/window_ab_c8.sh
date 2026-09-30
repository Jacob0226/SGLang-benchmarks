#!/usr/bin/env bash
# Empirical 1200s vs 3600s at one operating point, same config, same conc.
# window_sensitivity.py answers this from finished runs, but its truncation
# drops requests still in flight at the cutoff, which is a negative throughput
# bias a real short run only partly shares. This settles which part is real.
set -u
H=/home/jacchang/SGLang-benchmarks/Agentic
L=/home/jacchang/SGLang-benchmarks/_run_logs
COMMON="--conc 8 --gpus 0,1,2,3 --mem-fraction 0.75 --chunked-prefill 8192"

$H/ix_agentx_glm53flash_b200.sh $COMMON --duration 1200 --tag TP4-f75c8k-d1200 > "$L/glm53flash_d1200_c8.log" 2>&1
echo "$(date -Is) d1200 exit=$?"
pkill -9 -f 'sglang.launch_server' 2>/dev/null; pkill -9 -f 'sglang::' 2>/dev/null; sleep 30

exec $H/ix_agentx_glm53flash_b200.sh $COMMON --duration 3600 --tag TP4-f75c8k-d3600 > "$L/glm53flash_d3600_c8.log" 2>&1
