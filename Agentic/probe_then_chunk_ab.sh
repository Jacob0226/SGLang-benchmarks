#!/usr/bin/env bash
# Let the running conc=32 probe land, then stop the sweep before it starts
# conc=64 and run the chunk A/B instead.
#
# The probe already answered the question it was asked. At conc 32 the engine
# is prefill-bound, not memory-bound: KV pool 6% used, mamba pool 1%, and
# 5.7M tokens of prefill queued behind an 8192-token-per-step budget. Running
# conc 64 on that same config would spend an hour confirming it harder.
#
# chunked-prefill-size came down to 8192 to stop the DSA indexer OOM at
# mem-fraction 0.85. At 0.75 the headroom is 43 GB instead of 25 GB, so 16384
# should fit (worst observed indexer buffer at that chunk was 17.9 GiB) and
# doubles prefill throughput per step.
#
# Warmup grace goes to 3600s. At conc 32 the 1800s default was 77% gone with
# warmup still draining, which truncates it and starts the measurement on a
# half-filled cache.
set -uo pipefail

H=/home/jacchang/SGLang-benchmarks/Agentic
L=/home/jacchang/SGLang-benchmarks/_run_logs
B=/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130
C32=$B/bench-Agentic-TP4-f75c8k-probe/glm5.3flash_tp4_conc32_kvnone_spec-mtp/glm5.3flash_tp4_conc32_kvnone_spec-mtp_fp4_sglang_tp4-pp1-dcp1-pcp1-ep1-dpafalse_disagg-false_spec-mtp_conc32_local-b200.json

say() { echo "$(date -Is) $*"; }

say "waiting for the conc=32 baseline (chunk 8192) to land"
while [ ! -f "$C32" ]; do sleep 20; done
say "conc=32 baseline done; stopping the sweep before conc=64 starts"

pkill -f 'ix_agentx_glm53flash_b200.sh' 2>/dev/null
sleep 5
pkill -9 -f 'sglang.launch_server' 2>/dev/null
pkill -9 -f 'sglang::' 2>/dev/null
for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [ "${busy:-0}" -le 1024 ] && break
    sleep 15
done
say "GPUs free (max used=${busy:-?} MiB)"

# Same point, same window, only the prefill chunk differs.
say "starting conc=32 with chunk 16384"
exec "$H/ix_agentx_glm53flash_b200.sh" \
    --conc 32 --gpus 0,1,2,3 --mem-fraction 0.75 --chunked-prefill 16384 \
    --duration 1200 --tag TP4-f75c16k-probe \
    --env AGENTIC_WARMUP_GRACE_PERIOD=3600 \
    >> "$L/glm53flash_probe_chunk16k.log" 2>&1
