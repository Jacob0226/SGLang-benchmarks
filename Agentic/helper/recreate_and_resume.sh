#!/usr/bin/env bash
# Wait for the image pull, recreate the benchmark container, resume a sweep.
#
# The container has been removed from under a running sweep four times now,
# most recently along with the image itself, so this encodes the spec rather
# than leaving it to be remembered: same mounts, host networking (the
# benchmark containers rely on it), and enough shm for SGLang's workers.
#
# Resuming is safe because the driver skips any concurrency whose result JSON
# already exists, so an interrupted point is redone and finished ones are not.
set -uo pipefail

IMAGE="${IMAGE:-lmsysorg/sglang:v0.5.21-cu130}"
NAME="${NAME:-jacchang_GLM53-Flash-MTP}"
TAG="${TAG:?set TAG, the sweep tag to resume}"
CONC="${CONC:-1 4 8 16 32 64}"
GPUS="${GPUS:-0,1,2,3}"
DURATION="${DURATION:-3600}"
L=/home/jacchang/SGLang-benchmarks/_run_logs

say() { echo "$(date -Is) $*"; }

say "waiting for $IMAGE to finish pulling"
while ! docker image inspect "$IMAGE" >/dev/null 2>&1; do sleep 30; done
say "image present"

if docker inspect "$NAME" >/dev/null 2>&1; then
    say "removing stale container"
    docker rm -f "$NAME" >/dev/null 2>&1
fi

say "creating $NAME on $IMAGE"
docker run -d --name "$NAME" --gpus all --ipc=host --network host --shm-size 32g \
    -v /mnt/home/jacchang:/home/jacchang -v /mnt:/data -v /raid:/raid \
    -e USER=jacchang -e HOME=/home/jacchang -w /home/jacchang \
    "$IMAGE" sleep infinity >/dev/null || { say "docker run failed"; exit 1; }
sleep 5

docker exec "$NAME" bash -c 'ls -d /data/huggingface/hub/nvidia/GLM-5.3-Flash-NVFP4 >/dev/null' \
    || { say "checkpoint not visible in the container"; exit 1; }
say "container up, checkpoint visible; resuming sweep '$TAG'"

docker exec -d -w /home/jacchang/SGLang-benchmarks/Agentic "$NAME" bash -lc \
    "./ix_agentx_glm53flash.sh --conc \"$CONC\" --gpus $GPUS --duration $DURATION \
     --tag $TAG --env AGENTIC_WARMUP_GRACE_PERIOD=3600 >> $L/glm53flash_${TAG}.log 2>&1"
say "resumed; finished concurrencies will be skipped"
