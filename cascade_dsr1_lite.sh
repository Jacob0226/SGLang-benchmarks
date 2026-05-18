#!/usr/bin/env bash
# cascade_dsr1_lite.sh — DSR1-0528 cascade benchmark on ROCm aiter.
# Slim version of cascade_dsr1.sh. No NSA/GQA/MXFP4/EAGLE/CUDA dispatch.

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TAG=""
DOCKER="untagged-docker"
TP_SIZE=8
HOST="localhost"
PORT=30000
CACHE_MODE="L3_file"
CACHE_MODES=""
HICACHE_SIZE=auto
HOST_HEADROOM_GB=400
NUM_CLIENTS=300
NUM_ROUNDS=10
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
CUDA_GRAPH_MAX_BS=0
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
MEM_FRACTION_STATIC=0.85
PAGE_SIZE=64
CONTEXT_LENGTH=65536
HICACHE_WRITE_POLICY="write_through"
GSM8K_PRECHECK="true"
GSM8K_NUM_QUESTIONS=1200
GSM8K_PARALLEL=1200
WAIT_FOR_SERVER_SEC=1500

ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)               MODEL_PATH="$2"; shift 2;;
    --tag)                 TAG="$2"; shift 2;;
    --docker)              DOCKER="$2"; shift 2;;
    --cache-mode)          CACHE_MODE="$2"; shift 2;;
    --cache-modes)         CACHE_MODES="$2"; shift 2;;
    --tp)                  TP_SIZE="$2"; shift 2;;
    --port)                PORT="$2"; shift 2;;
    --hicache-size)        HICACHE_SIZE="$2"; shift 2;;
    --host-headroom-gb)    HOST_HEADROOM_GB="$2"; shift 2;;
    --num-clients)         NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)          NUM_ROUNDS="$2"; shift 2;;
    --request-length)      REQUEST_LENGTH="$2"; shift 2;;
    --max-parallel)        MAX_PARALLEL="$2"; shift 2;;
    --request-rate)        REQUEST_RATE="$2"; shift 2;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2;;
    --page-size)           PAGE_SIZE="$2"; shift 2;;
    --gsm8k-num-questions) GSM8K_NUM_QUESTIONS="$2"; shift 2;;
    --no-gsm8k-precheck)   GSM8K_PRECHECK="false"; shift 1;;
    -h|--help)
      echo "Usage: $0 --tag TAG --docker DOCKER [--cache-mode MODE | --cache-modes 'MODE1 MODE2 ...']"
      echo "Modes: none | L1 | L2 | L3_file"
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[ -z "$TAG" ] && { echo "ERROR: --tag required" >&2; exit 1; }
[ ! -d "$MODEL_PATH" ] && { echo "ERROR: bad model path: $MODEL_PATH" >&2; exit 1; }
case "$CACHE_MODE" in
  none|L1|L2|L3_file) ;;
  *) echo "ERROR: bad --cache-mode: $CACHE_MODE (none|L1|L2|L3_file)" >&2; exit 1;;
esac
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# ============================== Chain dispatcher ==============================
# --cache-modes "none L1 L2 L3_file" → re-exec self per mode, pkill+sleep between.
if [ -n "$CACHE_MODES" ]; then
  FORWARD_ARGS=()
  i=0
  while [ "$i" -lt "${#ORIG_ARGS[@]}" ]; do
    case "${ORIG_ARGS[$i]}" in
      --cache-modes) i=$((i + 2)) ;;
      *) FORWARD_ARGS+=("${ORIG_ARGS[$i]}"); i=$((i + 1)) ;;
    esac
  done
  for MODE in $CACHE_MODES; do
    echo ""
    echo ">>> ============================================================"
    echo ">>> chain: starting cache_mode=${MODE}"
    echo ">>> ============================================================"
    "$0" --cache-mode "$MODE" "${FORWARD_ARGS[@]}"
    pkill -9 sglang 2>/dev/null || true
    sleep 10
  done
  echo ">>> chain done (${CACHE_MODES})"
  exit 0
fi

# ============================== Auto-size hicache ==============================
if [ "$CACHE_MODE" = "none" ] || [ "$CACHE_MODE" = "L1" ]; then
  HICACHE_SIZE=0
elif [ "$HICACHE_SIZE" = "auto" ]; then
  sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  # Wait for MemAvailable to settle (prior pinned-host pool may still be releasing).
  prev=0
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    cur=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
    if [ "$prev" -gt 0 ] && [ $((cur - prev)) -le 5 ] && [ $((prev - cur)) -le 5 ]; then
      echo ">>> MemAvail settled at ${cur} GB"
      break
    fi
    prev=$cur
    sleep 5
  done
  MEM_AVAIL_GB=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
  USABLE_GB=$(( MEM_AVAIL_GB - HOST_HEADROOM_GB ))
  [ "$USABLE_GB" -le 0 ] && { echo "ERROR: only ${MEM_AVAIL_GB} GB avail, need >${HOST_HEADROOM_GB} headroom" >&2; exit 1; }
  PER_RANK=$(( USABLE_GB / TP_SIZE / 32 * 32 ))
  [ "$PER_RANK" -lt 32 ] && PER_RANK=32
  [ "$PER_RANK" -gt 512 ] && PER_RANK=512
  HICACHE_SIZE="$PER_RANK"
  echo ">>> auto hicache-size: ${HICACHE_SIZE} GB/rank (total $(( HICACHE_SIZE * TP_SIZE )) GB)"
fi

# ============================== Output dir + meta ==============================
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-cascade-${TAG}"
case "$CACHE_MODE" in
  none|L1)    LOG_DIR="${BASE_LOG_DIR}/${CACHE_MODE}" ;;
  L2|L3_file) LOG_DIR="${BASE_LOG_DIR}/${CACHE_MODE}/size_${HICACHE_SIZE}" ;;
esac
mkdir -p "$LOG_DIR"

META_HICACHE=$([ "$CACHE_MODE" = "none" ] || [ "$CACHE_MODE" = "L1" ] && echo "null" || echo "$HICACHE_SIZE")
cat > "$LOG_DIR/bench_meta.json" <<EOF
{
  "cache_mode": "$CACHE_MODE",
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "page_size": $PAGE_SIZE,
  "tp_size": $TP_SIZE,
  "hicache_size_gb": $META_HICACHE,
  "hicache_write_policy": "$HICACHE_WRITE_POLICY",
  "mem_fraction_static": $MEM_FRACTION_STATIC,
  "num_clients": $NUM_CLIENTS,
  "num_rounds": $NUM_ROUNDS,
  "request_length": $REQUEST_LENGTH,
  "max_parallel": $MAX_PARALLEL,
  "request_rate": $REQUEST_RATE,
  "tag": "$TAG",
  "docker": "$DOCKER",
  "gsm8k_precheck_accuracy": null
}
EOF

# ============================== L3 file store + disk check ==============================
HICACHE_FILE_STORE_DIR=""
if [[ "$CACHE_MODE" == L3_* ]]; then
  HICACHE_FILE_STORE_DIR="/tmp/cascade_dsr1_l3_${TAG}_${HICACHE_SIZE}"
  rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null
  mkdir -p "$HICACHE_FILE_STORE_DIR"
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_FILE_STORE_DIR"

  L2_TOTAL_GB=$(( HICACHE_SIZE * TP_SIZE ))
  L3_FREE_GB=$(df -BG --output=avail "$HICACHE_FILE_STORE_DIR" | tail -1 | tr -dc '0-9')
  echo ">>> L3 disk: free=${L3_FREE_GB} GB, L2 total=${L2_TOTAL_GB} GB"
  if [ "${L3_FREE_GB:-0}" -le "$L2_TOTAL_GB" ]; then
    echo "ERROR: L3 free (${L3_FREE_GB} GB) <= L2 total (${L2_TOTAL_GB} GB)" >&2
    exit 1
  fi
fi

# ============================== Trap ==============================
trap '
  rm -rf "${HICACHE_FILE_STORE_DIR:-}" 2>/dev/null
  [ -n "${CACHE_MONITOR_PID:-}" ] && kill "${CACHE_MONITOR_PID}" 2>/dev/null
  pkill -9 sglang 2>/dev/null || true
  sleep 10
  true' EXIT

# ============================== Server cmd ==============================
# DSR1-0528 + ROCm aiter + page_size=64: must disable PR #18528 FP8 prefill kernel.
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1
export SGLANG_USE_AITER=1
export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
export SGLANG_AITER_FP8_PREFILL_ATTN=0

SERVER_CMD=(
  python3 -u -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --tp "$TP_SIZE"
    --host "$HOST" --port "$PORT"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --watchdog-timeout 2400
    --enable-metrics
    --enable-cache-report
    --trust-remote-code
    --kv-cache-dtype fp8_e4m3
    --page-size "$PAGE_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --attention-backend aiter
)
# Only pass --cuda-graph-max-bs if user explicitly overrode (>0).
[ "$CUDA_GRAPH_MAX_BS" -gt 0 ] && SERVER_CMD+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")

case "$CACHE_MODE" in
  none) SERVER_CMD+=(--disable-radix-cache);;
  L1)   :;;
  L2)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-io-backend kernel
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
    );;
  L3_file)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-io-backend kernel
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
      --hicache-storage-backend file
      --hicache-storage-prefetch-policy best_effort
    );;
esac

# ============================== Launch + pre-check + health ==============================
SERVER_LOG="$LOG_DIR/server.log"
echo ">>> launching SGLang cache_mode=${CACHE_MODE}"
echo "${SERVER_CMD[*]}" | tee "$SERVER_LOG"

if pgrep sglang >/dev/null 2>&1; then
  echo "ERROR: sglang already running. 'pkill -9 sglang && sleep 10' first." >&2
  pgrep -a sglang >&2
  exit 1
fi
if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
  echo "ERROR: port ${PORT} already serves /health 200" >&2
  exit 1
fi

"${SERVER_CMD[@]}" 2>&1 | tee -a "$SERVER_LOG" &
SERVER_BG_PID=$!

deadline=$(( $(date +%s) + WAIT_FOR_SERVER_SEC ))
echo ">>> wait /health up to ${WAIT_FOR_SERVER_SEC}s..."
while [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; do
  if ! kill -0 "$SERVER_BG_PID" 2>/dev/null; then
    echo "ERROR: server pipeline died. tail of $SERVER_LOG:" >&2
    tail -n 50 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERROR: /health timeout, see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
echo ">>> server ready (pid=$SERVER_BG_PID)"

# ============================== Warmup ==============================
echo ">>> warmup"
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --model "$MODEL_PATH" --dataset-name random \
  --random-input 1024 --random-output 128 --random-range-ratio 1.0 \
  --max-concurrency 4 --num-prompt 8 --output-file /dev/null \
  2>&1 | tee "$LOG_DIR/warmup.log"

# ============================== GSM8K precheck ==============================
if [ "$GSM8K_PRECHECK" = "true" ]; then
  GSM8K_SCRIPT=""
  for c in /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
           "$HOME/work-space/sglang/benchmark/gsm8k/bench_sglang.py"; do
    [ -f "$c" ] && GSM8K_SCRIPT="$c" && break
  done
  if [ -n "$GSM8K_SCRIPT" ]; then
    echo ">>> GSM8K precheck (${GSM8K_NUM_QUESTIONS} q, parallel=${GSM8K_PARALLEL})"
    ( cd "$LOG_DIR" && python3 "$GSM8K_SCRIPT" \
        --host "$HOST" --port "$PORT" \
        --num-questions "$GSM8K_NUM_QUESTIONS" \
        --parallel "$GSM8K_PARALLEL" \
        --result-file "$LOG_DIR/Accuracy_GSM8K.jsonl" \
        2>&1 | tee "$LOG_DIR/Accuracy_GSM8K.log" ) || true
    ACC=$(grep -oP '^Accuracy:\s+\K[0-9.]+' "$LOG_DIR/Accuracy_GSM8K.log" | tail -1 || true)
    echo ">>> GSM8K accuracy: ${ACC:-NA}"
    python3 - "$LOG_DIR/bench_meta.json" "${ACC:-null}" <<'PY'
import json, sys
p, acc = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["gsm8k_precheck_accuracy"] = float(acc) if acc != "null" else None
json.dump(d, open(p, "w"), indent=2)
PY
  else
    echo ">>> WARNING: bench_sglang.py not found, skipping GSM8K"
  fi
fi

# ============================== Bench multiturn ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
[ -z "$BENCH_SCRIPT" ] && { echo "ERROR: bench_multiturn.py not found" >&2; exit 1; }

# Flush radix tree + OS page cache so cascade starts cold.
curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
sleep 2
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
sleep 2

# Optional cache_monitor sidecar (per-tier hit rate per round → cache_tiers.csv).
CACHE_MONITOR_SCRIPT="$(dirname "$(readlink -f "$0")")/cache_monitor.py"
CACHE_MONITOR_PID=""
if [ -f "$CACHE_MONITOR_SCRIPT" ]; then
  python3 "$CACHE_MONITOR_SCRIPT" \
      --url "http://${HOST}:${PORT}/metrics" \
      --interval 10 \
      --num-clients "$NUM_CLIENTS" \
      --num-rounds "$NUM_ROUNDS" \
      --csv "$LOG_DIR/cache_tiers.csv" \
      > "$LOG_DIR/cache_monitor.log" 2>&1 &
  CACHE_MONITOR_PID=$!
fi

echo ">>> bench multiturn N=${NUM_CLIENTS} R=${REQUEST_LENGTH} rounds=${NUM_ROUNDS}"
python3 "$BENCH_SCRIPT" \
  --host "$HOST" --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --num-clients "$NUM_CLIENTS" \
  --num-rounds "$NUM_ROUNDS" \
  --request-length "$REQUEST_LENGTH" \
  --output-length "$OUTPUT_LENGTH" \
  --max-parallel "$MAX_PARALLEL" \
  --request-rate "$REQUEST_RATE" \
  --ready-queue-policy random \
  --log-file "$LOG_DIR/bench_multiturn.jsonl" \
  --tag "${MODEL_NAME}-${TAG}" \
  --disable-random-sample \
  --disable-auto-run \
  --enable-round-barrier \
  2>&1 | tee "$LOG_DIR/bench_multiturn.log"

if [ -n "$CACHE_MONITOR_PID" ] && kill -0 "$CACHE_MONITOR_PID" 2>/dev/null; then
  kill "$CACHE_MONITOR_PID" 2>/dev/null || true
  wait "$CACHE_MONITOR_PID" 2>/dev/null || true
fi

# ============================== Cleanup ==============================
echo ">>> stopping server"
pkill -9 sglang 2>/dev/null || true
sleep 10
[ -n "$HICACHE_FILE_STORE_DIR" ] && [ -d "$HICACHE_FILE_STORE_DIR" ] && rm -rf "$HICACHE_FILE_STORE_DIR"

echo ">>> done. results in: $LOG_DIR"
