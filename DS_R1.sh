#!/usr/bin/env bash
# Usage:
# ./DS_R1.sh --model /data/huggingface/hub/deepseek-ai/DeepSeek-R1
# ./DS_R1.sh --model /data/huggingface/hub/deepseek-ai/DeepSeek-R1 --prof
set -euo pipefail

# ===================== Argument Parsing =====================
MODEL=""
PROF_ENABLED="false"
MTP_ENABLED="false"
MTP_LOG_NAME=""
prof_cmd=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --prof)
      PROF_ENABLED="true"
      shift 1
      ;;
    --mtp)           
      MTP_ENABLED="true"
      MTP_LOG_NAME="-MTP"
      shift 1
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ -z "$MODEL" ]; then
    echo "Error: --model parameter is required."
    echo "Usage: $0 --model /path/to/model [--prof]"
    exit 1
fi

# ===================== User-adjustable params =====================
HOST="localhost"
PORT="8000"
DATASET="random"

input_tokens=70000
output_tokens=200
random_range_ratio=1.0

concurrencies=(1)
CHUNKED_PREFILL_SIZE_VALUES=(131072) 

# ===================== Setup Log Directory =====================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
Model_name=$(basename "$MODEL")
LOG_DIR="henryx-xsgl_v0.5.8-rocm720-mi35x-20260202-preview-aiter-0e0a37/${Model_name}${MTP_LOG_NAME}"
if [ "$PROF_ENABLED" == "true" ]; then
    LOG_DIR="${LOG_DIR}_prof"
    prof_cmd="--profile --profile-num-steps 400 --profile-by-stage"
    export SGLANG_TORCH_PROFILER_DIR="${LOG_DIR}"
fi
mkdir -p "${LOG_DIR}"
echo ">>> [$(date)] Log directory created: ${LOG_DIR}"

# ===================== Functions =====================

print_env_info() {
    echo ">>> [$(date)] Printing related environment variables:"
    echo "--- Environment Snapshot ---"
    env | sort | grep -E '^(SGLANG_|RCCL_|NCCL_|HIP_|HSA_|ROCM_|AITER_|VLLM_|TORCH)' || echo "(No relevant environment variables found)"
    echo "----------------------------"
    echo
}

log_command() {
    local logfile=$1
    shift
    echo ">>> [$(date)] Executing command:" | tee -a "$logfile"
    echo "$*" | tee -a "$logfile"
    echo "---" | tee -a "$logfile"
}

wait_for_server_ready() {
    local logfile=$1
    echo ">>> Waiting for server to be ready (checking: '${logfile}')..." | tee -a "$logfile"
    local timeout=1800
    local elapsed=0
    local interval=10

    while true; do
        if grep -q "The server is fired up and ready to roll!" "$logfile" 2>/dev/null; then
            echo ">>> [$(date)] Server is ready!" | tee -a "$logfile"
            break
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
        if (( elapsed >= timeout )); then
            echo "!!! ERROR: Server did not become ready within ${timeout}s" | tee -a "$logfile"
            exit 1
        fi
    done
}

start_server() {
    local chunked_prefill_size_val=$1
    local logfile="${LOG_DIR}/server_${Model_name}.log"

    echo ">>> [$(date)] Starting SGLang server with --chunked-prefill-size=${chunked_prefill_size_val}" | tee -a "$logfile"

    # Setup ROCm/SGLang Optimized ENV
    export HIP_FORCE_DEV_KERNARG=1
    export HSA_NO_SCRATCH_RECLAIM=1
    export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
    export SGLANG_INT4_WEIGHT=0
    export SGLANG_MOE_PADDING=1
    export SGLANG_ROCM_DISABLE_LINEARQUANT=0
    export SGLANG_ROCM_FUSED_DECODE_MLA=1
    export SGLANG_SET_CPU_AFFINITY=1
    export SGLANG_USE_AITER=1
    export SGLANG_USE_ROCM700A=1
    export SGLANG_AITER_MLA_PERSIST=1
    export NCCL_MIN_NCHANNELS=112
    export VLLM_FP8_PADDING=1
    export VLLM_FP8_ACT_PADDING=1
    export VLLM_FP8_WEIGHT_PADDING=1
    export VLLM_FP8_REDUCE_CONV=1
    export TORCHINDUCTOR_MAX_AUTOTUNE=1
    export TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE=1

    print_env_info | tee -a "$logfile"


    local cmd=(
        python3 -m sglang.launch_server
        --model-path "${MODEL}"
        --tensor-parallel-size 8
        --trust-remote-code
        --chunked-prefill-size "${chunked_prefill_size_val}"
        --host 0.0.0.0
        --port "${PORT}"
        --mem-fraction-static 0.95
        --attention-backend aiter
        --max-running-requests 64
        --disable-radix-cache
        --kv-cache-dtype fp8_e4m3
    )

    if [ "$MTP_ENABLED" == "true" ]; then
        echo ">>> [$(date)] Speculative Decoding (MTP) is ENABLED." | tee -a "$logfile"
        cmd+=(
            --speculative-algorithm EAGLE
            --speculative-num-draft-tokens 4
            --speculative-num-steps 3
            --speculative-eagle-topk 1
        )
    fi

    log_command "$logfile" "${cmd[@]}"

    # Start server in background
    "${cmd[@]}" 2>&1 | tee -a "$logfile" &
    SERVER_PID=$!
    
    wait_for_server_ready "$logfile"

    echo ">>> Running Accuracy check (GSM8K)..."
    python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
        --num-questions 200 --port "$PORT" --parallel 200 2>&1 | tee "$LOG_DIR/Accuracy.log"
}

stop_server() {
    echo ">>> [$(date)] Stopping server PID=${SERVER_PID}"
    kill -9 "${SERVER_PID}" || true
    sleep 5
}

run_benchmarks() {
    local chunked_prefill_size_val=$1

    # Warmup
    local warmup_log="${LOG_DIR}/warmup_${chunked_prefill_size_val}.log"
    python3 -m sglang.bench_serving \
        --host "${HOST}" --port "${PORT}" --model "${MODEL}" \
        --dataset-name "${DATASET}" --random-input "${input_tokens}" \
        --random-output "${output_tokens}" --random-range-ratio "${random_range_ratio}" \
        --max-concurrency 1 --num-prompt 8 2>&1 | tee "$warmup_log"

    # Benchmark Loop
    for c in "${concurrencies[@]}"; do
        local num_prompts=$((c * 16))
        local logfile="${LOG_DIR}/bench_chunked_prefill_${chunked_prefill_size_val}_conc_${c}.log"

        local cmd=(
            python3 -m sglang.bench_serving
            --host "${HOST}"
            --port "${PORT}"
            --model "${MODEL}"
            --dataset-name "${DATASET}"
            --random-input "${input_tokens}"
            --random-output "${output_tokens}"
            --random-range-ratio "${random_range_ratio}"
            --max-concurrency "${c}"
            --num-prompt "${num_prompts}"
        )
        
        # 加上 profiling 參數 (如果是空字串則不會影響)
        if [ "$PROF_ENABLED" == "true" ]; then
            cmd+=($prof_cmd)
        fi

        log_command "$logfile" "${cmd[@]}"
        "${cmd[@]}" 2>&1 | tee -a "$logfile"
        
        # Profiler 檔案處理
        if [ "$PROF_ENABLED" == "true" ]; then
            echo ">>> Processing profiler traces..."
            for file in "${LOG_DIR}"/*.gz; do
                if [[ -f "$file" && "$file" == *"TP-"* ]]; then
                    filename=$(basename "$file")
                    suffix="TP-${filename##*-TP-}"
                    new_name="${Model_name}_conc${c}_${suffix}"
                    mv "$file" "${LOG_DIR}/${new_name}"
                    echo "Renamed: $filename -> $new_name"
                fi
            done
        fi
    done
}

# ===================== Main Loop =====================

for CHUNKED_PREFILL_SIZE in "${CHUNKED_PREFILL_SIZE_VALUES[@]}"; do
    echo "===================================================================="
    echo ">>> Starting test for --chunked-prefill-size=${CHUNKED_PREFILL_SIZE}"
    echo "===================================================================="

    start_server "${CHUNKED_PREFILL_SIZE}"
    run_benchmarks "${CHUNKED_PREFILL_SIZE}"
    stop_server

    echo ">>> Finished chunked_prefill_size=${CHUNKED_PREFILL_SIZE}"
done

echo ">>> All configurations completed. Logs: ${LOG_DIR}"
pkill -9 python || true