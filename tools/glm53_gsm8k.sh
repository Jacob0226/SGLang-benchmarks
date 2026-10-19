#!/bin/bash
# sgl_eval GSM8K (thinking, same recipe as glm53_mxfp4_verify.sh) against a running server.
# Run inside the container. usage: glm53_gsm8k.sh <port> <out_dir> [num_examples]
set -euo pipefail
PORT=$1 OUT=$2 N=${3:-1319}
MODEL=/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4
mkdir -p "$OUT"
ulimit -n 65535
cd /root
python3 -m sgl_eval.cli run gsm8k \
  --base-url "http://127.0.0.1:$PORT/v1" \
  --model "$MODEL" \
  --num-examples "$N" \
  --num-threads 1200 \
  --max-tokens 4096 \
  --temperature 1.0 \
  --top-p 0.95 \
  --seed 0 \
  --thinking \
  --out-dir "$OUT/gsm8k" > "$OUT/gsm8k.log" 2>&1
grep -E "score|truncated_rate|error_rate" "$OUT/gsm8k.log"
