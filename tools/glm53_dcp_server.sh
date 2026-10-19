#!/bin/bash
# Launch GLM-5.3-Flash MXFP4 (TP4, optional DCP) inside the container from a container-local
# copy of a source tree. Run on the node host.
# usage: glm53_dcp_server.sh <tag> <gpus> <port> <dcp_size> [extra server args...]
#   env: TREE (default ~/PR/wt-glm53-dcp), DST (container copy, default /root/ab_trees/dcp),
#        NAME (container), SYNC=0 to reuse the copied tree,
#        EXTRA_ENV="A=1 B=2" exported inside the container,
#        LOGD (log dir inside the container), CACHE (SGLANG_CACHE_DIR)
set -euo pipefail
TAG=$1 GPUS=$2 PORT=$3 DCP=$4
shift 4
NAME=${NAME:-jacchang_GLM53-Flash}
TREE=${TREE:-/home/jacchang/PR/wt-glm53-dcp}
DST=${DST:-/root/ab_trees/dcp}
LOGD=${LOGD:-/home/jacchang/SGLang-benchmarks/tmp/glm53_dcp}
CACHE=${CACHE:-/root/sglang_cache_dcp}
MODEL=/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4
docker exec $NAME mkdir -p $LOGD

if [ "${SYNC:-1}" = 1 ]; then
    docker exec $NAME bash -c "rm -rf $DST && mkdir -p $DST && cp -r $TREE/python $DST/ && find $DST -name __pycache__ -prune -exec rm -rf {} +"
fi

DCP_ARGS=()
[ "$DCP" -gt 1 ] && DCP_ARGS=(--dcp-size "$DCP")

docker exec -d $NAME bash -c "
export PYTHONPATH=$DST/python HIP_VISIBLE_DEVICES=$GPUS SGLANG_USE_AITER=1 SAFETENSORS_FAST_GPU=1 \
       SGLANG_CACHE_DIR=$CACHE ${EXTRA_ENV:-}
cd /root && exec python3 -m sglang.launch_server \
  --model-path $MODEL --trust-remote-code \
  --tp 4 ${DCP_ARGS[*]} \
  --kv-cache-dtype bfloat16 \
  --dsa-prefill-backend triton --dsa-decode-backend triton \
  --moe-runner-backend aiter \
  --mem-fraction-static 0.85 --chunked-prefill-size 16384 \
  --reasoning-parser glm45 --tool-call-parser glm47 \
  --host 127.0.0.1 --port $PORT $* > $LOGD/server_$TAG.log 2>&1"
echo "launched $TAG on GPUs $GPUS port $PORT (dcp=$DCP); log: $LOGD/server_$TAG.log"
