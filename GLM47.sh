#!/usr/bin/env bash
# Usage:
# ./GLM47.sh
# ./GLM47.sh --mtp
# ./GLM47.sh --prof
set -euo pipefail
set -x

MTP_ENABLED="false"
PROF_ENABLED="false"
MTP_TAG=""
ACTION_TAG=""
MODEL_PATH="/data/huggingface/hub/zai-org/GLM-4.7"
MODEL_NAME=$(basename "${MODEL_PATH%/}")
CURRENT_DIR=$(pwd)
while [[ $# -gt 0 ]]; do
  case $1 in
    --mtp)
        MTP_ENABLED="true"
        MTP_TAG="-MTP"
        shift 1
        ;;
    --prof)
        PROF_ENABLED="true"
        shift 1
        ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# ===================== Server and Benchmark Setting =====================
HOST="localhost"
PORT="8552"
DATASET="random"
in_out_tokens=("1000:1000" "8000:1000")
random_range_ratio=1.0
concurrencies=(1 2 4 8 16)

# ===================== Argument  =====================
DOCKER="henryx/xsgl:v0.5.8-rocm720-mi35x-20260202-preview-aiter-0e0a37"
SPECIAL_TAG=""
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}${MTP_TAG}${SPECIAL_TAG}"
FINISH_LOG="$LOG_DIR/Finish.log"
PROF_CMD="--profile"
mkdir -p "$LOG_DIR"
touch "$FINISH_LOG"

log_command() {
    local logfile=$1
    shift 
    
    # Record command
    echo ">>>Executing command:" | tee -a "$logfile"
    echo "$*" | tee -a "$logfile"  
    echo "---" | tee -a "$logfile"

    # Execute command
    "$@" 2>&1 | tee -a "$logfile"
}

start_server() {
    local logfile="${LOG_DIR}/server_GLM47.log"
    echo ">>> Starting SGLang server" | tee "$logfile"

    local cmd=(
        python3 -m sglang.launch_server 
            --model $MODEL_PATH
            --tp 8 
            --host $HOST
            --port $PORT
            --tool-call-parser glm47  
            --reasoning-parser glm45
            --disable-radix-cache
    )

    if [ "$MTP_ENABLED" == "true" ]; then
        # ROCm GPU (MI355X) currently only support triton backend in speculative decoding
        echo ">>> Speculative Decoding (MTP) is ENABLED." | tee -a "$logfile"
        cmd+=(
            --speculative-algorithm EAGLE
            --speculative-num-draft-tokens 4
            --speculative-num-steps 3
            --speculative-eagle-topk 1
            --attention-backend triton
        )
    fi

    # Start server in background
    echo ">>> Executing command:" | tee -a "$logfile"
    echo "${cmd[*]}" | tee -a "$logfile"
    echo "---" | tee -a "$logfile"
    "${cmd[@]}" 2>&1 | tee -a "$logfile" &

    echo ">>> Waiting for server to be ready (checking: '${logfile}')..." | tee -a "$logfile"
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:$PORT/health")" -eq 200 ]; do
        echo "Waiting for server to be ready at http://${HOST}:$PORT/health..."
        sleep 5
    done
}

warmup() {
    local warmup_log="${LOG_DIR}/warmup.log"
    local warmup_cmd=(
        python3 -m sglang.bench_serving 
        --host $HOST 
        --port "${PORT}" 
        --model "${MODEL_PATH}" 
        --dataset-name "${DATASET}" 
        --random-input 256
        --random-output 256
        --random-range-ratio "${random_range_ratio}"
        --max-concurrency 2 
        --num-prompt 8 
    )
    log_command "$warmup_log" "${warmup_cmd[@]}"
}

accuracy_test() {
    # GSM8K
    gsm8k_logfile=$LOG_DIR/Accuracy_GSM8K.log
    if ! grep -q "$gsm8k_logfile" "$FINISH_LOG"; then
        echo ">>> Running Accuracy check (GSM8K)..."
        gsm8k_cmd=(
            python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py 
                --port "$PORT" 
                --num-questions 1200 
                --parallel 1200
        )
        log_command "$gsm8k_logfile" "${gsm8k_cmd[@]}"
        echo "$gsm8k_logfile" >> "$FINISH_LOG"
    else
        echo "Found Accuracy_GSM8K.log in ${FINISH_LOG}. Skipping."
    fi
}

run_benchmarks() {
    # Benchmark Loop
    for io_pair in "${in_out_tokens[@]}"; do
        IFS=":" read -r input_tokens output_tokens <<< "$io_pair"
        for c in "${concurrencies[@]}"; do
            local num_prompts=$((c * 16))
            local logfile="${LOG_DIR}/bench_in${input_tokens}_out${output_tokens}_conc${c}.log"

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
                cmd+=("${PROF_CMD}")
            fi

            if ! grep -q "$logfile" "$FINISH_LOG"; then
                echo "Running: $logfile"
                log_command "$logfile" "${cmd[@]}"
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
    done
}


# ------------------- Start -----------------
start_server
warmup
accuracy_test
run_benchmarks

















# GLM-4.7 is not a VLM.
# MMMU test
# TOKEN_LIST=(512 1024 2048 4096 8192 16384 131072)
# TOKEN_LIST=(16384 131072)
# ITERATIONS=3
# mkdir -p "$LOG_DIR/MMMU"
# for iter in $(seq 1 $((ITERATIONS))); do
#     for token in "${TOKEN_LIST[@]}"; do
#         mmmu_logfile="$LOG_DIR/MMMU/MMMU_Token${token}_Iter${iter}.log"
#         mmmu_cmd=(
#             python3 /sgl-workspace/sglang/benchmark/mmmu/bench_sglang.py  
#                 --port "$PORT" 
#                 --concurrency 900 
#                 --parallel 900 
#                 --temperature 0
#                 --max-new-tokens "$token" 
#                 --result-file "$LOG_DIR/MMMU/MMMU_Token${token}_ResultFile_Iter${iter}.jsonl"
#                 --raw-result-file "$LOG_DIR/MMMU/MMMU_Token${token}_RawResultFile_Iter${iter}.jsonl"
#         )
        
#         if ! grep -q "$mmmu_logfile" "$FINISH_LOG" 2>/dev/null; then
#             echo ">>> Running MMMU: Token $token, Iteration $iter"

#             start_time=$(date +%s)
#             log_command "$mmmu_logfile" "${mmmu_cmd[@]}"
#             end_time=$(date +%s)
#             elapsed=$((end_time - start_time))
#             formatted_time=$(date -u -d "@$elapsed" +"%H:%M:%S")
#             echo "-------------------------------------------" >> "$mmmu_logfile"
#             echo "Execution Time $formatted_time (HH:MM:SS)" >> "$mmmu_logfile"

#             echo "$mmmu_logfile" >> "$FINISH_LOG"
#             mv $CURRENT_DIR/answer_sglang.json $LOG_DIR/MMMU/MMMU_Token${token}_answer_sglang_Iter${iter}.json
#         else
#             echo "Found $mmmu_logfile in Finish.log. Skipping."
#         fi
#     done
# done
# # stop_server
# echo ">>> All configurations completed. Logs: ${LOG_DIR}. Stop server..."
# pkill -9 python || true
# sleep 10
