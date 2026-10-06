#!/usr/bin/env bash
# Wait for the running sweep to finish, then run the identical sweep again.
#
# Two things this measures that a single sweep cannot: run-to-run spread on a
# quiet machine, and how much of a point's number comes from state that
# survives between runs. The second is not controlled away, deliberately --
# run 2 inherits warm compiled-kernel caches (SGLANG_CACHE_DIR, FlashInfer
# autotune, Triton), a warm HF corpus cache, and model weights in the host
# page cache. The KV/radix cache is not inherited: the server is restarted per
# concurrency, so every point starts cold on that one.
#
#   TAG1=TP4-v0521 TAG2=TP4-v0521-rep2 ./chain_repeat.sh
set -uo pipefail

H=/home/jacchang/SGLang-benchmarks/Agentic
L=/home/jacchang/SGLang-benchmarks/_run_logs
TAG1="${TAG1:?set TAG1, the tag of the sweep to wait for}"
TAG2="${TAG2:?set TAG2, the tag for the repeat}"
CONC="${CONC:-1 4 8 16 32 64}"
DURATION="${DURATION:-3600}"
GPUS="${GPUS:-0,1,2,3}"

say() { echo "$(date -Is) $*"; }

say "waiting for sweep '$TAG1' to finish"
while pgrep -f "ix_agentx_glm53flash.sh.*--tag $TAG1" >/dev/null 2>&1; do sleep 60; done
say "'$TAG1' is gone"

# Teardown of the last point can still be settling.
for _ in $(seq 1 60); do
    busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | sort -rn | head -1)
    [ "${busy:-0}" -le 1024 ] && break
    sleep 15
done
say "GPUs free (max used=${busy:-?} MiB); starting repeat '$TAG2'"

exec "$H/ix_agentx_glm53flash.sh" \
    --conc "$CONC" --gpus "$GPUS" --duration "$DURATION" --tag "$TAG2" \
    --env AGENTIC_WARMUP_GRACE_PERIOD=3600 \
    > "$L/glm53flash_${TAG2}.log" 2>&1
