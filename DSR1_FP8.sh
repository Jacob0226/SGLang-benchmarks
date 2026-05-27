#!/usr/bin/env bash
# DeepSeek-R1-0528 FP8 dense prefill benchmark.
#
# Goal: reproduce InferenceX-style random dense serving workloads for
# i1k/o1k and i8k/o1k, without prefix/radix cache by default, so MI355X vs B200
# differences are easier to attribute to dense prefill/decode kernels rather
# than HiCache behavior.
#
# Usage:
#   ./DSR1_FP8.sh --docker rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260526 --tag 0527
#   ./DSR1_FP8.sh --docker lmsysorg/sglang:v0.5.12.post1-cu130 --tag 0527
#   ./DSR1_FP8.sh --in-out "1024:1024 4096:1024 8192:1024 12288:1024"

set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' 2>/dev/null || true

MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TP_SIZE=8
DOCKER=${DOCKER:-untagged-docker}
USER_TAG=""
HOST="localhost"
PORT=8552

# InferenceX-like random dense workloads.
IN_OUT_TOKENS=("1024:1024" "8192:1024" "16384:1024")
CONCURRENCIES=(4) #  8 16 32 64
PROMPT_MULTIPLIER=5
RANDOM_RANGE_RATIO=1.0
DATASET="random"

MEM_FRACTION_STATIC=0.85
PAGE_SIZE=64
CONTEXT_LENGTH=65536
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
DISABLE_RADIX_CACHE=true
SKIP_WARMUP=false
EXTRA_SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2;;
    --tp) TP_SIZE="$2"; shift 2;;
    --docker) DOCKER="$2"; shift 2;;
    --tag) USER_TAG="-$2"; shift 2;;
    --host) HOST="$2"; shift 2;;
    --port) PORT="$2"; shift 2;;
    --in-out)
      read -r -a IN_OUT_TOKENS <<< "$2"
      shift 2
      ;;
    --concurrencies)
      read -r -a CONCURRENCIES <<< "$2"
      shift 2
      ;;
    --prompt-multiplier) PROMPT_MULTIPLIER="$2"; shift 2;;
    --random-range-ratio) RANDOM_RANGE_RATIO="$2"; shift 2;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2;;
    --page-size) PAGE_SIZE="$2"; shift 2;;
    --context-length) CONTEXT_LENGTH="$2"; shift 2;;
    --chunked-prefill-size) CHUNKED_PREFILL_SIZE="$2"; shift 2;;
    --max-prefill-tokens) MAX_PREFILL_TOKENS="$2"; shift 2;;
    --enable-radix-cache) DISABLE_RADIX_CACHE=false; shift 1;;
    --skip-warmup) SKIP_WARMUP=true; shift 1;;
    --extra-server-arg)
      EXTRA_SERVER_ARGS+=("$2")
      shift 2
      ;;
    -h|--help)
      cat <<EOF
Usage: $0 [opts]

Common opts:
  --docker IMAGE                 Label result folder by docker image
  --tag TAG                      Append tag to result folder
  --model PATH                   Default: $MODEL_PATH
  --tp N                         Default: $TP_SIZE
  --in-out "I:O ..."             Default: "1024:1024 8192:1024"
  --concurrencies "N ..."        Default: "4 8 16 32 64"
  --prompt-multiplier N          num_prompts = concurrency * N (default 5)
  --random-range-ratio R         Default: 1.0
  --mem-fraction-static F        Default: 0.85
  --enable-radix-cache           Keep prefix/radix cache on (default off)
  --skip-warmup                  Skip 2048/256 warmup
  --extra-server-arg ARG         Append one raw launch_server arg; repeatable
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

MODEL_NAME=$(basename "${MODEL_PATH%/}")
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
CACHE_TAG=$([ "$DISABLE_RADIX_CACHE" = true ] && echo "-NoRadix" || echo "-Radix")
LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-FP8-dense${CACHE_TAG}${USER_TAG}"
FINISH_LOG="$LOG_DIR/Finish.log"
mkdir -p "$LOG_DIR"
touch "$FINISH_LOG"

is_rocm_gpu_env() {
  [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1
}

log_command() {
  local logfile=$1
  shift
  echo ">>> Executing command:" | tee -a "$logfile"
  printf '%q ' "$@" | tee -a "$logfile"
  echo | tee -a "$logfile"
  echo "---" | tee -a "$logfile"
  "$@" 2>&1 | tee -a "$logfile"
}

cleanup_server() {
  pkill -9 sglang 2>/dev/null || true
  pkill -9 python 2>/dev/null || true
}

start_server() {
  local logfile="$LOG_DIR/server_${MODEL_NAME}.log"
  echo ">>> Starting SGLang server" | tee "$logfile"

  export PYTHONUNBUFFERED=1
  export SAFETENSORS_FAST_GPU=1

  local cmd=(
    python3 -u -m sglang.launch_server
      --model-path "$MODEL_PATH"
      --tp "$TP_SIZE"
      --host "$HOST"
      --port "$PORT"
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
  )

  if [ "$DISABLE_RADIX_CACHE" = true ]; then
    cmd+=(--disable-radix-cache)
  fi

  if is_rocm_gpu_env; then
    cmd+=(
      --attention-backend aiter
    )
    export SGLANG_USE_AITER=1
    export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
    export SGLANG_AITER_FP8_PREFILL_ATTN=0
  else
    cmd+=(
      --attention-backend trtllm_mla
      --moe-runner-backend flashinfer_trtllm
      --enable-flashinfer-allreduce-fusion
      --quantization fp8
    )
    export SGL_ENABLE_JIT_DEEPGEMM=1
  fi

  if [ "${#EXTRA_SERVER_ARGS[@]}" -gt 0 ]; then
    cmd+=("${EXTRA_SERVER_ARGS[@]}")
  fi

  if pgrep -f sglang.launch_server >/dev/null 2>&1; then
    echo "ERROR: sglang already running. Stop it first." >&2
    pgrep -af sglang >&2 || true
    exit 1
  fi
  if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
    echo "ERROR: port ${PORT} already serves /health 200" >&2
    exit 1
  fi

  echo ">>> Executing command:" | tee -a "$logfile"
  printf '%q ' "${cmd[@]}" | tee -a "$logfile"
  echo | tee -a "$logfile"
  echo "---" | tee -a "$logfile"
  "${cmd[@]}" 2>&1 | tee -a "$logfile" &

  echo ">>> Waiting for server to be ready at http://${HOST}:${PORT}/health" | tee -a "$logfile"
  until [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" = "200" ]; do
    sleep 5
  done
  echo ">>> server ready" | tee -a "$logfile"
}

warmup() {
  local logfile="$LOG_DIR/warmup.log"
  local cmd=(
    python3 -m sglang.bench_serving
      --backend sglang
      --host "$HOST"
      --port "$PORT"
      --model "$MODEL_PATH"
      --dataset-name "$DATASET"
      --random-input 2048
      --random-output 256
      --random-range-ratio "$RANDOM_RANGE_RATIO"
      --max-concurrency 4
      --num-prompt 8
      --output-file /dev/null
  )
  log_command "$logfile" "${cmd[@]}"
}

run_benchmarks() {
  for io_pair in "${IN_OUT_TOKENS[@]}"; do
    IFS=":" read -r input_tokens output_tokens <<< "$io_pair"
    for c in "${CONCURRENCIES[@]}"; do
      local num_prompts=$(( c * PROMPT_MULTIPLIER ))
      local logfile="$LOG_DIR/bench_in${input_tokens}_out${output_tokens}_conc${c}.log"
      if grep -q "$logfile" "$FINISH_LOG"; then
        echo "Found $logfile in ${FINISH_LOG}. Skipping."
        continue
      fi

      local cmd=(
        python3 -m sglang.bench_serving
          --backend sglang
          --host "$HOST"
          --port "$PORT"
          --model "$MODEL_PATH"
          --dataset-name "$DATASET"
          --random-input "$input_tokens"
          --random-output "$output_tokens"
          --random-range-ratio "$RANDOM_RANGE_RATIO"
          --max-concurrency "$c"
          --num-prompt "$num_prompts"
          --output-file /dev/null
      )
      log_command "$logfile" "${cmd[@]}"
      echo "$logfile" >> "$FINISH_LOG"
    done
  done
}

trap cleanup_server EXIT
start_server
if [ "$SKIP_WARMUP" != true ]; then
  warmup
fi
run_benchmarks

echo ">>> Done. Logs in $LOG_DIR"
