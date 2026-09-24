#!/usr/bin/env bash
# One-shot handoff: let the running TP4 sweep finish its conc=8 point, stop it
# before it can start conc=12, run TP8 conc=1 on all eight GPUs, then hand the
# GPUs back to the TP4 sweep for conc 12 and 16.
#
# TP8 needs every GPU on the node, so it cannot overlap the TP4 lane; this
# script exists only to make the cut at a point boundary instead of killing a
# 3600s window half way through.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGS=/home/jacchang/SGLang-benchmarks/_run_logs
TP4_ROOT=/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/bench-Agentic-TP4_EP1-mtp5
CONC8_JSON="$TP4_ROOT/glm5.3flash_tp4_conc8_kvnone_spec-mtp/glm5.3flash_tp4_conc8_kvnone_spec-mtp_fp4_sglang_tp4-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_conc8_local-b200.json"

say() { echo "$(date -Is) $*"; }

say "waiting for the TP4 conc=8 result to land at $CONC8_JSON"
while [ ! -f "$CONC8_JSON" ]; do sleep 10; done
say "conc=8 done"

# The driver writes the JSON, then logs the memory-pool lines, kills the server
# and sleeps 30s before the next point. Cut in during that window: killing the
# loop leader stops it from launching conc=12, and nothing is mid-measurement.
pkill -f 'ix_agentx_glm53flash_b200.sh' 2>/dev/null
sleep 5
pkill -9 -f 'sglang.launch_server' 2>/dev/null
pkill -9 -f 'sglang::' 2>/dev/null
say "TP4 loop stopped; waiting for all eight GPUs to drain"

for _ in $(seq 1 80); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${busy:-0}" -le 1024 ] && break
    sleep 15
done
say "GPUs free (max used=${busy:-?} MiB)"

# TP8: every GPU, so no --gpus filter. Same recipe and same knobs as the TP4
# lane so the only variable between the two curves is the parallelism.
say "starting TP8 conc=1"
"$HERE/ix_agentx_glm53flash_b200.sh" --tp 8 --conc 1 > "$LOGS/glm53flash_agentx_tp8_conc1.log" 2>&1
say "TP8 conc=1 exit=$?"

pkill -9 -f 'sglang.launch_server' 2>/dev/null
pkill -9 -f 'sglang::' 2>/dev/null
sleep 30

# Resume the TP4 curve. The driver skips any point whose result JSON already
# exists, so this fills in 12 and 16 only.
say "resuming TP4 sweep for conc 12 and 16"
exec "$HERE/ix_agentx_glm53flash_b200.sh" --conc "1 4 8 12 16" --gpus 0,1,2,3 \
    >> "$LOGS/glm53flash_agentx_tp4_sweep.log" 2>&1
