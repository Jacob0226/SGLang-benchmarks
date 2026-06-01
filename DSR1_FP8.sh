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
CONCURRENCIES=(4 32) #  8 16 32 64
PROMPT_MULTIPLIER=10
RANDOM_RANGE_RATIO=0.8
DATASET="random"

MEM_FRACTION_STATIC=""
MEM_FRACTION_EXPLICIT=false
PAGE_SIZE=""
PAGE_SIZE_EXPLICIT=false
CONTEXT_LENGTH=""
CONTEXT_LENGTH_EXPLICIT=false
CHUNKED_PREFILL_SIZE=""
CHUNKED_PREFILL_EXPLICIT=false
MAX_PREFILL_TOKENS=""
MAX_PREFILL_EXPLICIT=false
DISABLE_RADIX_CACHE=true
SKIP_WARMUP=false
GSM8K_PRECHECK=true
EXTRA_SERVER_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL_PATH="$2"; shift 2;;
    --tp) TP_SIZE="$2"; shift 2;;
    --docker)
      # Support `DOCKER=img ./DSR1_FP8.sh --docker "$DOCKER"` where "$DOCKER"
      # is expanded by the parent shell before the inline env assignment.
      [ -n "${2:-}" ] && DOCKER="$2"
      shift 2
      ;;
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
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; MEM_FRACTION_EXPLICIT=true; shift 2;;
    --page-size) PAGE_SIZE="$2"; PAGE_SIZE_EXPLICIT=true; shift 2;;
    --context-length) CONTEXT_LENGTH="$2"; CONTEXT_LENGTH_EXPLICIT=true; shift 2;;
    --chunked-prefill-size) CHUNKED_PREFILL_SIZE="$2"; CHUNKED_PREFILL_EXPLICIT=true; shift 2;;
    --max-prefill-tokens) MAX_PREFILL_TOKENS="$2"; MAX_PREFILL_EXPLICIT=true; shift 2;;
    --enable-radix-cache) DISABLE_RADIX_CACHE=false; shift 1;;
    --skip-warmup) SKIP_WARMUP=true; shift 1;;
    --no-gsm8k-precheck) GSM8K_PRECHECK=false; shift 1;;
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
  --prompt-multiplier N          num_prompts = concurrency * N (default 10)
  --random-range-ratio R         Default: 1.0
  --mem-fraction-static F        Override InferenceX platform default
  --enable-radix-cache           Keep prefix/radix cache on (default off)
  --skip-warmup                  Skip 2048/256 warmup
  --no-gsm8k-precheck            Skip GSM8K accuracy check
  --extra-server-arg ARG         Append one raw launch_server arg; repeatable
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

MODEL_NAME=$(basename "${MODEL_PATH%/}")
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-bench${USER_TAG}"
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

  local max_conc=0
  for c in "${CONCURRENCIES[@]}"; do
    if [ "$c" -gt "$max_conc" ]; then
      max_conc="$c"
    fi
  done

  local cuda_graph_max_bs=""
  local max_running_requests=""
  local scheduler_recv_interval=""
  if is_rocm_gpu_env; then
    if [ "$MEM_FRACTION_EXPLICIT" = false ]; then
      MEM_FRACTION_STATIC=0.8
    fi
    if [ "$CHUNKED_PREFILL_EXPLICIT" = false ]; then
      CHUNKED_PREFILL_SIZE=196608
    fi
    if [ "$MAX_PREFILL_EXPLICIT" = false ]; then
      MAX_PREFILL_TOKENS=196608
    fi
    cuda_graph_max_bs="$max_conc"
    # Match InferenceX benchmarks/single_node/dsr1_fp8_mi355x.sh exactly:
    # they hard-export INT4 to override the rocm700-mi35x image's baked-in INT8.
    export SGLANG_USE_AITER=1
    export RCCL_MSCCL_ENABLE=0
    export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
    export SGLANG_AITER_FP8_PREFILL_ATTN=${SGLANG_AITER_FP8_PREFILL_ATTN:-0}
  else
    if [ "$TP_SIZE" -eq 8 ]; then
      if [ "$MEM_FRACTION_EXPLICIT" = false ]; then
        MEM_FRACTION_STATIC=0.82
      fi
      if [ "$CHUNKED_PREFILL_EXPLICIT" = false ]; then
        CHUNKED_PREFILL_SIZE=32768
      fi
      if [ "$MAX_PREFILL_EXPLICIT" = false ]; then
        MAX_PREFILL_TOKENS=32768
      fi
      max_running_requests=128
      cuda_graph_max_bs=128
      if [ "$max_conc" -ge 16 ]; then
        scheduler_recv_interval=30
      else
        scheduler_recv_interval=10
      fi
    elif [ "$TP_SIZE" -eq 4 ]; then
      if [ "$MEM_FRACTION_EXPLICIT" = false ]; then
        MEM_FRACTION_STATIC=0.95
      fi
      if [ "$CHUNKED_PREFILL_EXPLICIT" = false ]; then
        CHUNKED_PREFILL_SIZE=8192
      fi
      if [ "$MAX_PREFILL_EXPLICIT" = false ]; then
        MAX_PREFILL_TOKENS=8192
      fi
      max_running_requests=32
      cuda_graph_max_bs=32
      scheduler_recv_interval=10
    else
      echo "ERROR: InferenceX B200 recipe only handles TP=4 or TP=8, got TP=${TP_SIZE}" >&2
      exit 1
    fi
    export SGL_ENABLE_JIT_DEEPGEMM=false
    export SGLANG_ENABLE_FLASHINFER_GEMM=true
  fi

  # Mirror InferenceX benchmarks/single_node/dsr1_fp8_mi355x.sh EXACTLY:
  #   - no --page-size  (let sglang auto-pick page_size=1 for aiter legacy MQA path)
  #   - no --context-length (let it default to model's 163840)
  #   - no --enable-metrics, --enable-cache-report
  #   - default --watchdog-timeout (300)
  local cmd=(
    python3 -u -m sglang.launch_server
      --model-path "$MODEL_PATH"
      --tp "$TP_SIZE"
      --host "$HOST"
      --port "$PORT"
      --mem-fraction-static "$MEM_FRACTION_STATIC"
      --trust-remote-code
      --kv-cache-dtype fp8_e4m3
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
      --max-prefill-tokens "$MAX_PREFILL_TOKENS"
      --cuda-graph-max-bs "$cuda_graph_max_bs"
  )

  # Honor explicit --page-size / --context-length overrides if the user set them.
  if [ "$PAGE_SIZE_EXPLICIT" = true ]; then
    cmd+=(--page-size "$PAGE_SIZE")
  fi
  if [ "$CONTEXT_LENGTH_EXPLICIT" = true ]; then
    cmd+=(--context-length "$CONTEXT_LENGTH")
  fi

  if [ "$DISABLE_RADIX_CACHE" = true ]; then
    cmd+=(--disable-radix-cache)
  fi

  if is_rocm_gpu_env; then
    cmd+=(
      --attention-backend aiter
      --num-continuous-decode-steps 8
    )
  else
    cmd+=(
      --max-running-requests "$max_running_requests"
      --scheduler-recv-interval "$scheduler_recv_interval"
      --attention-backend trtllm_mla
      --stream-interval 30
      --ep-size 1
      --moe-runner-backend flashinfer_trtllm
      --enable-flashinfer-allreduce-fusion
      --quantization fp8
    )
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

accuracy_test() {
  local gsm8k_logfile="$LOG_DIR/Accuracy_GSM8K.log"
  if grep -q "$gsm8k_logfile" "$FINISH_LOG"; then
    echo "Found Accuracy_GSM8K.log in ${FINISH_LOG}. Skipping."
    return
  fi

  echo ">>> Running Accuracy check (GSM8K)..."
  local gsm8k_cmd=(
    python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py
      --port "$PORT"
      --num-questions 1200
      --parallel 1200
  )
  log_command "$gsm8k_logfile" "${gsm8k_cmd[@]}"
  echo "$gsm8k_logfile" >> "$FINISH_LOG"
}

run_benchmarks() {
  for io_pair in "${IN_OUT_TOKENS[@]}"; do
    IFS=":" read -r input_tokens output_tokens <<< "$io_pair"
    for c in "${CONCURRENCIES[@]}"; do
      local num_prompts=$(( c * PROMPT_MULTIPLIER ))
      local warmup_requests=$(( c * 2 ))
      local logfile="$LOG_DIR/bench_in${input_tokens}_out${output_tokens}_conc${c}.log"
      if grep -q "$logfile" "$FINISH_LOG"; then
        echo "Found $logfile in ${FINISH_LOG}. Skipping."
        continue
      fi

      # Match InferenceX bench client: --num-warmups 2*conc (sglang default = 1).
      # Cold-start would otherwise pollute the first few requests' TTFT.
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
          --warmup-requests "$warmup_requests"
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
if [ "$GSM8K_PRECHECK" = true ]; then
  accuracy_test
fi
run_benchmarks

echo ">>> Done. Logs in $LOG_DIR"
