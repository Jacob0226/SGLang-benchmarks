#!/usr/bin/env bash
# Usage:
# ./DS_R1.sh # Run all models
# ./DS.sh --model /data/huggingface/hub/deepseek-ai/DeepSeek-R1 
# ./DS.sh --model /data/huggingface/hub/amd/DeepSeek-R1-MXFP4
# ./DS.sh --model /data/huggingface/hub/amd/DeepSeek-R1-MXFP4 --mtp
# ./DS.sh --model /data/huggingface/hub/deepseek-ai/DeepSeek-V3.2-Exp # --prof
set -euo pipefail

# ===================== Argument  =====================
ROOT_FOLDER="results/henryx-xsgl:v0.5.8-rocm720-mi35x-20260202-preview-aiter-0e0a37"
FINISH_LOG="$ROOT_FOLDER/Finish.log"
TASKS=(
    "/data/huggingface/hub/deepseek-ai/DeepSeek-R1|false"
    "/data/huggingface/hub/amd/DeepSeek-R1-MXFP4|false"
    "/data/huggingface/hub/amd/DeepSeek-R1-MXFP4|true"
    "/data/huggingface/hub/deepseek-ai/DeepSeek-V3.2-Exp|false"
)
PROF_ENABLED="false"
MTP_ENABLED="false"
MTP_LOG_NAME=""
prof_cmd="--profile --profile-num-steps 400 --profile-by-stage"

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      SINGLE_MODEL="$2"
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

FINAL_TASKS=()
if [ -n "${SINGLE_MODEL:-}" ]; then
    FINAL_TASKS+=("${SINGLE_MODEL}|${MTP_ENABLED}")
else
    # Run all models
    FINAL_TASKS=("${TASKS[@]}")
fi

# ===================== User-adjustable params =====================
HOST="localhost"
PORT="8552"
DATASET="random"

input_tokens=70000
output_tokens=200
random_range_ratio=1.0

concurrencies=(1 2 4 8 16)
# CHUNKED_PREFILL_SIZE_VALUES=(131072) 

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
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")" -eq 200 ]; do
        echo "Waiting for server to be ready at http://127.0.0.1:$PORT/health..."
        sleep 5
    done

    echo -e "\nServer is READY! Starting warmup..."
}

start_server() {
    local logfile="${LOG_DIR}/server_${MODEL_NAME}.log"

    echo ">>> [$(date)] Starting SGLang server" | tee -a "$logfile"

    if [[ "$MODEL_NAME" =~ (DeepSeek-R1|DeepSeek-R1-MXFP4) ]]; then
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
            --model-path "${MODEL_PATH}"
            --tensor-parallel-size 8
            --trust-remote-code
            --chunked-prefill-size 131072
            --host 0.0.0.0
            --port "${PORT}"
            --mem-fraction-static 0.95
            --attention-backend aiter
            --max-running-requests 64
            --disable-radix-cache
            --kv-cache-dtype fp8_e4m3
        )

        if [ "$CURRENT_MTP" == "true" ]; then
            echo ">>> [$(date)] Speculative Decoding (MTP) is ENABLED." | tee -a "$logfile"
            cmd+=(
                --speculative-algorithm EAGLE
                --speculative-num-draft-tokens 4
                --speculative-num-steps 3
                --speculative-eagle-topk 1
            )
        fi
    elif [[ "$MODEL_NAME" == "DeepSeek-V3.2-Exp" ]]; then
        # AMD (MI355X)
        if command -v rocminfo > /dev/null 2>&1 || [ -d "/opt/rocm" ]; then
            echo ">>> ROCm environment detected."
            export RCCL_MSCCL_ENABLE=0   
            export PYTHONPATH="${PYTHONPATH:-}:/opt/tilelang"
            cmd=(
                python3 -m sglang.launch_server 
                    --model $MODEL_PATH 
                    --mem-fraction-static 0.7 
                    --tp 8 
                    --port $PORT 
                    --trust-remote-code 
                    --disable-radix-cache 
                    --chunked-prefill-size 131072 
                    --nsa-prefill-backend tilelang 
                    --nsa-decode-backend tilelang
            )
        # NVIDIA (B200/H100) 
        else
            echo ">>> NVIDIA environment detected."
            cmd=(
                python3 -m sglang.launch_server 
                --model "$MODEL_PATH" 
                --mem-fraction-static 0.7 
                --tp 8 
                --port "$PORT" 
                --trust-remote-code 
                --disable-radix-cache
            )

            # DP8
            # From: https://lmsys.org/blog/2025-09-29-deepseek-V32/
            # cmd=(
            # python3 -m sglang.launch_server 
            #     --model "$MODEL_PATH" 
            #     --mem-fraction-static 0.7 
            #     --tp 8 
            #     --port "$PORT" 
            #     --trust-remote-code 
            #     --disable-radix-cache  
            #     --dp 8 --enable-dp-attention
            # )
            
            # For long context. From: https://docs.sglang.io/basic_usage/deepseek_v32.html#in-sequence-splitting-default-setting
            # cmd=(
            # python3 -m sglang.launch_server 
            #     --model "$MODEL_PATH" 
            #     --tp 8 
            #     --ep 8 
            #     --dp 2 
            #     --enable-dp-attention 
            #     --enable-nsa-prefill-context-parallel 
            #     --nsa-prefill-cp-mode in-seq-split 
            #     --max-running-requests 32
            # )
        fi
    fi

    log_command "$logfile" "${cmd[@]}"

    # Start server in background
    "${cmd[@]}" 2>&1 | tee -a "$logfile" &
    SERVER_PID=$!
    
    wait_for_server_ready "$logfile"
    if ! grep -q "$LOG_DIR/Accuracy.log" "$FINISH_LOG"; then
        echo ">>> Running Accuracy check (GSM8K)..."
        python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
            --num-questions 200 --port "$PORT" --parallel 200 2>&1 | tee "$LOG_DIR/Accuracy.log"
        echo "$LOG_DIR/Accuracy.log" >> "$FINISH_LOG"
    else
        echo "Found $LOG_DIR/Accuracy.log in ${FINISH_LOG}. Skipping."
    fi
}


run_benchmarks() {
    # Warmup
    local warmup_log="${LOG_DIR}/warmup.log"
    python3 -m sglang.bench_serving \
        --host "${HOST}" --port "${PORT}" --model "${MODEL_PATH}" \
        --dataset-name "${DATASET}" --random-input "${input_tokens}" \
        --random-output "${output_tokens}" --random-range-ratio "${random_range_ratio}" \
        --max-concurrency 1 --num-prompt 8 2>&1 | tee "$warmup_log"

    # Benchmark Loop
    for c in "${concurrencies[@]}"; do
        local num_prompts=$((c * 16))
        local logfile="${LOG_DIR}/bench_conc_${c}.log"

        local cmd=(
            python3 -m sglang.bench_serving
            --host "${HOST}"
            --port "${PORT}"
            --model "${MODEL_PATH}"
            --dataset-name "${DATASET}"
            --random-input "${input_tokens}"
            --random-output "${output_tokens}"
            --random-range-ratio "${random_range_ratio}"
            --max-concurrency "${c}"
            --num-prompt "${num_prompts}"
        )
        
        # Add profiling args
        if [ "$PROF_ENABLED" == "true" ]; then
            cmd+=($prof_cmd)
        fi

        if ! grep -q "$logfile" "$FINISH_LOG"; then
            echo "Running: $logfile"
            log_command "$logfile" "${cmd[@]}"
            "${cmd[@]}" 2>&1 | tee -a "$logfile"
            echo "$logfile" >> "$FINISH_LOG"
        else
            echo "Found $logfile in ${FINISH_LOG}. Skipping."
        fi

        # --- Move and Rename TorchProfiler files ---
        if [ "$PROF_ENABLED" == "true" ]; then
            echo ">>> Processing profiler traces..."
            for file in "${LOG_DIR}"/*.gz; do
                if [[ -f "$file" && "$file" == *"TP-"* ]]; then
                    filename=$(basename "$file")
                    suffix="TP-${filename##*-TP-}"
                    new_name="${MODEL_NAME}_conc${c}_${suffix}"
                    mv "$file" "${LOG_DIR}/${new_name}"
                    echo "Renamed: $filename -> $new_name"
                fi
            done
        fi
    done
}

# ===================== Main Loop =====================
for task in "${FINAL_TASKS[@]}"; do
    IFS="|" read -r MODEL_PATH CURRENT_MTP <<< "$task"
    MODEL_NAME=$(basename "${MODEL_PATH%/}")
    MTP_TAG=""
    [ "$CURRENT_MTP" == "true" ] && MTP_TAG="-MTP"

    # Create log folder
    LOG_DIR="$ROOT_FOLDER/${MODEL_NAME}${MTP_TAG}"
    [ "$PROF_ENABLED" == "true" ] && LOG_DIR="${LOG_DIR}_prof"
    mkdir -p "$LOG_DIR"

    if [ "$PROF_ENABLED" == "true" ]; then
        export SGLANG_TORCH_PROFILER_DIR=$ROOT_FOLDER
    fi

    echo ">>> Log Directory: ${LOG_DIR}"
    echo "------------------------------------------------------------"

    start_server
    run_benchmarks

    # stop_server
    echo ">>> All configurations completed. Logs: ${LOG_DIR}. Stop server..."
    pkill -9 python || true
    sleep 10
done


# ===================== Profiler parsing =====================
echo ">>> Starting recursive post-processing of profiler traces..."
find "$ROOT_FOLDER" -type f -name "*.gz" | while read -r file; do
    # get file name (E.g., /A/B/C/123.gz --> /A/B/C/123)
    base_path="${file%.gz}"
    
    # define csv filename
    output_csv="${base_path}.csv"
    
    echo "Processing: $file"
    echo "Output to: $output_csv"
    
    # Parsing
    python3 "$HOME/SGLang-benchmarks/parse_torch_profiler.py" \
        --file "$file" \
        --out "$output_csv"
done

echo ">>> All post-processing completed."