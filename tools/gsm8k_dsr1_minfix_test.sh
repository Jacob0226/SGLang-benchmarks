#!/usr/bin/env bash
# Single-shot GSM8K accuracy test for DSR1-0528 on MI355X.
#
# Tests the "minimal-fix" hypothesis vs the buggy 0.011 accuracy seen in
# cascade_dsr1.sh's GSM8K precheck:
#   - Drop --page-size 64        (no public ref uses page_size>1 for FP8 KV)
#   - Drop --reasoning-parser    (may eat bench_sglang.py's text output)
#   - Set SGLANG_USE_AITER=1     (AMD official + InferenceX both set this)
#   - Set ROCM_QUICK_REDUCE_QUANTIZATION=NONE  (Clint Greene flips from INT4)
#
# Everything else (TP, mem-fraction, kv-cache-dtype, chunked-prefill-size,
# max-prefill-tokens, cuda-graph-max-bs, context-length) matches the user's
# cascade_dsr1.sh config so the change is isolated.
#
# Run inside jacchang_HiCache container:
#   bash /home/jacchang/SGLang-benchmarks/tools/gsm8k_dsr1_minfix_test.sh

set -uo pipefail
ulimit -n 65535

MODEL_PATH=/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528
PORT=30000
HOST=localhost
LOG_DIR=/home/jacchang/SGLang-benchmarks/results/rocm_sgl-dev-v0.5.11-rocm720-mi35x-20260507/DSR1-0528-gsm8k-minfix-test
SERVER_LOG="$LOG_DIR/server.log"
GSM8K_LOG="$LOG_DIR/Accuracy_GSM8K.log"
mkdir -p "$LOG_DIR"

trap 'pkill -9 -f sglang.launch_server 2>/dev/null || true' EXIT

# 4 minimal-fix env / flag changes
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE   # was INT4
export SAFETENSORS_FAST_GPU=1
export SGLANG_USE_AITER=1                    # was unset

echo "=== env ==="
env | grep -E "ROCM_|SGLANG_|SAFETENSORS" | sort
echo

echo "=== launching SGLang ==="
SERVER_CMD=(
  python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --tp 8
    --host "$HOST" --port "$PORT"
    --mem-fraction-static 0.85
    --watchdog-timeout 1200
    --enable-metrics
    --enable-cache-report
    --trust-remote-code
    --kv-cache-dtype fp8_e4m3
    --context-length 65536
    --chunked-prefill-size 32768
    --max-prefill-tokens 32768
    --cuda-graph-max-bs 8
    --attention-backend aiter
)
echo "${SERVER_CMD[*]}" | tee "$SERVER_LOG"
"${SERVER_CMD[@]}" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "server pid=$SERVER_PID"

echo "=== waiting up to 600s for /health 200 ==="
deadline=$(( $(date +%s) + 600 ))
while :; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then break; fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ">>> SERVER PROCESS DIED before /health came up"
    tail -50 "$SERVER_LOG"
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo ">>> /health timeout"
    tail -50 "$SERVER_LOG"
    exit 1
  fi
  sleep 5
done
echo ">>> server ready"

echo "=== GSM8K precheck (1200 questions, parallel=1200) ==="
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
  --port "$PORT" \
  --num-questions 1200 \
  --parallel 1200 \
  2>&1 | tee "$GSM8K_LOG"

echo
echo "=== RESULT ==="
grep -E "^(Accuracy|Invalid|Latency|Output throughput):" "$GSM8K_LOG" || true

echo
echo ">>> stopping server"
pkill -9 -f sglang.launch_server 2>/dev/null || true
sleep 3
echo ">>> done. logs: $LOG_DIR"
