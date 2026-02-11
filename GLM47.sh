#!/usr/bin/env bash
# Usage:
# ./GLM47.sh
set -euo pipefail
set -x

MTP_ENABLED="false"
MTP_LOG_NAME=""
MODEL_PATH="/data/huggingface/hub/zai-org/GLM-4.7"
MODEL_NAME=$(basename "${MODEL_PATH%/}")
CURRENT_DIR=$(pwd)
while [[ $# -gt 0 ]]; do
  case $1 in
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

# ===================== Argument  =====================
DOCKER="henryx/xsgl:v0.5.8-rocm720-mi35x-20260202-preview-aiter-0e0a37"
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}${MTP_LOG_NAME}"
FINISH_LOG="$LOG_DIR/Finish.log"
PORT="8552"
mkdir -p "$LOG_DIR"

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
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT/health")" -eq 200 ]; do
        echo "Waiting for server to be ready at http://127.0.0.1:$PORT/health..."
        sleep 5
    done
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

    # MMMU
    mmmu_logfile=$LOG_DIR/Accuracy_MMMU.log
    if ! grep -q "$mmmu_logfile" "$FINISH_LOG"; then
        echo ">>> Running Accuracy check (MMMU)..."
        token=2048
        mmmu_cmd=(
            python3 /sgl-workspace/sglang/benchmark/mmmu/bench_sglang.py  
                --port $PORT --concurrency 900 --parallel 900 
                --temperature 0
                --max-new-tokens $token 
                --result-file  $LOG_DIR/MMMU_Token${token}_ResultFile.jsonl
                --raw-result-file $LOG_DIR/MMMU_Token${token}_RawResultFile.jsonl
        )
        log_command "$mmmu_logfile" "${mmmu_cmd[@]}"
        echo "$mmmu_logfile" >> "$FINISH_LOG"
        mv $CURRENT_DIR/answer_sglang.json $LOG_DIR/answer_sglang.json
    else
        echo "Found Accuracy_MMMU.log in ${FINISH_LOG}. Skipping."
    fi
}

# ------------------- Start -----------------
start_server
accuracy_test

# MMMU test
TOKEN_LIST=(512 1024 2048 4096 8192 16384)
ITERATIONS=3
mkdir -p "$LOG_DIR/MMMU"
for token in "${TOKEN_LIST[@]}"; do
    for iter in $(seq 1 $((ITERATIONS))); do
        mmmu_logfile="$LOG_DIR/MMMU/MMMU_Token${token}_Iter${iter}.log"
        mmmu_cmd=(
            python3 /sgl-workspace/sglang/benchmark/mmmu/bench_sglang.py  
                --port "$PORT" 
                --concurrency 900 
                --parallel 900 
                --temperature 0
                --max-new-tokens "$token" 
                --extra-request-body '{"extra_body": {"thinking": {"type": "enabled"}}}'
                --result-file "$LOG_DIR/MMMU/MMMU_Token${token}_ResultFile_Iter${iter}.jsonl"
                --raw-result-file "$LOG_DIR/MMMU/MMMU_Token${token}_RawResultFile_Iter${iter}.jsonl"
        )
        
        if ! grep -q "$mmmu_logfile" "$FINISH_LOG" 2>/dev/null; then
            echo ">>> Running MMMU: Token $token, Iteration $iter"
            log_command "$mmmu_logfile" "${mmmu_cmd[@]}"
            echo "$mmmu_logfile" >> "$FINISH_LOG"
            mv $CURRENT_DIR/answer_sglang.json $LOG_DIR/MMMU/MMMU_Token${token}_answer_sglang_Iter${iter}.json
        else
            echo "Found $mmmu_logfile in Finish.log. Skipping."
        fi
    done
done
# stop_server
echo ">>> All configurations completed. Logs: ${LOG_DIR}. Stop server..."
pkill -9 python || true
sleep 10