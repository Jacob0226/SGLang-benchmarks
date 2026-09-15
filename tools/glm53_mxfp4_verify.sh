#!/usr/bin/env bash
# Serve amd/GLM-5.3-Flash-Quark-MXFP4 from an arbitrary sglang tree and grade it.
#
# Runs inside the container. The tree is supplied through PYTHONPATH rather than
# checked out into /sgl-workspace/sglang, so the image's own editable install is
# left alone and several trees can be graded against one image.
#
#   SGLANG_SRC=<tree> GPUS=4,5,6,7 PORT=8555 TAG=<name> glm53_mxfp4_verify.sh [serve|eval|both]
set -euo pipefail

SGLANG_SRC="${SGLANG_SRC:?set SGLANG_SRC to the sglang tree to grade}"
MODEL="${MODEL:-/data/hf_home/hub/models--amd--GLM-5.3-Flash-Quark-MXFP4/snapshots/b5688f25491202978c19c4d036eef579f61bbe07}"
GPUS="${GPUS:-4,5,6,7}"
PORT="${PORT:-8555}"
TAG="${TAG:-mxfp4}"
NUM_EXAMPLES="${NUM_EXAMPLES:-1319}"
STAGE="${1:-both}"

OUT=/home/jacchang/SGLang-benchmarks/results/glm53-mxfp4-verify/$TAG
RUN_CACHE=$OUT/cache
mkdir -p "$OUT"

# Set exactly one of these. Both together filter twice: ROCR remaps 4,5,6,7 to
# 0,1,2,3 and HIP then looks for 4,5,6,7 among four devices and finds none.
export HIP_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$SGLANG_SRC/python:/sgl-workspace/aiter"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export SGLANG_USE_AITER=1
export SAFETENSORS_FAST_GPU=1

# One empty cache per run. AITER keys its JIT products by module name alone, so
# two trees sharing a directory silently load each other's binaries.
export AITER_JIT_DIR="$RUN_CACHE/aiter"
export FLYDSL_RUNTIME_CACHE_DIR="$RUN_CACHE/flydsl"
export TILELANG_CACHE_DIR="$RUN_CACHE/tilelang"
export TRITON_CACHE_DIR="$RUN_CACHE/triton"
export TORCH_EXTENSIONS_DIR="$RUN_CACHE/torch_extensions"
export TORCHINDUCTOR_CACHE_DIR="$RUN_CACHE/torchinductor"
export XDG_CACHE_HOME="$RUN_CACHE/xdg"
export SGLANG_JIT_CACHE_DIR="$RUN_CACHE/sglang_jit"

if [[ "$STAGE" == serve || "$STAGE" == both ]]; then
  rm -rf "$RUN_CACHE"; mkdir -p "$RUN_CACHE"

  echo "tree   : $(git -C "$SGLANG_SRC" rev-parse --short HEAD)"
  echo "sglang : $(python3 -c 'import inspect,sglang;print(inspect.getfile(sglang))')"
  echo "aiter  : $(git -C /sgl-workspace/aiter rev-parse HEAD)"
  echo "gpus   : $GPUS   port: $PORT   out: $OUT"

  nohup python3 -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp 4 \
    --trust-remote-code \
    --attention-backend dsa \
    --kv-cache-dtype bfloat16 \
    --context-length 65536 \
    --mem-fraction-static 0.85 \
    --disable-radix-cache \
    --dsa-prefill-backend tilelang \
    --dsa-decode-backend tilelang \
    --linear-attn-backend triton \
    --moe-runner-backend aiter \
    --disable-shared-experts-fusion \
    --max-running-requests 64 \
    --cuda-graph-backend-decode full \
    --cuda-graph-max-bs-decode 64 \
    --model-loader-extra-config '{"enable_multithread_load":true,"num_threads":8}' \
    --watchdog-timeout 1200 \
    --reasoning-parser glm45 \
    --tool-call-parser glm47 \
    --host 0.0.0.0 --port "$PORT" > "$OUT/server.log" 2>&1 &
  echo "server pid $! -> $OUT/server.log"
fi

if [[ "$STAGE" == eval || "$STAGE" == both ]]; then
  for _ in $(seq 1 180); do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null && break
    sleep 20
  done
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "server never came up"; tail -40 "$OUT/server.log"; exit 1; }

  ulimit -n 65535
  python3 -m sgl_eval.cli run gsm8k \
    --base-url "http://127.0.0.1:$PORT/v1" \
    --model "$MODEL" \
    --num-examples "$NUM_EXAMPLES" \
    --num-threads 1200 \
    --max-tokens 4096 \
    --temperature 1.0 \
    --top-p 0.95 \
    --seed 0 \
    --thinking \
    --out-dir "$OUT/gsm8k" 2>&1 | tee "$OUT/gsm8k.log"
fi
