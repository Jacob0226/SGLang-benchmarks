#!/usr/bin/env bash
# Usage:
# ./GLM.sh
# ./GLM.sh --mtp --prof
# ./GLM.sh 
# ./GLM.sh --model /data/huggingface/hub/zai-org/GLM-5-FP8
set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing'

MTP_ENABLED="false"
PROF_ENABLED="false"
MTP_TAG=""
MODEL_PATH="/data/huggingface/hub/zai-org/GLM-5-FP8"
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
    --model)
        MODEL_PATH="$2"
        shift 2
        ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done
MODEL_NAME=$(basename "${MODEL_PATH%/}")
if [[ "${MODEL_NAME}" == *GLM-5* ]]; then
    echo ">>> Detected GLM-5, ensuring transformers is up-to-date..."
    python3 -m pip install -U --no-cache-dir \
        "git+https://github.com/huggingface/transformers.git@6ed9ee36f608fd145168377345bfc4a5de12e1e2"
fi

# ===================== Server and Benchmark Setting =====================
# InferenceMax tuning (from InferenceX/glm5_fp8_mi355x.sh)
export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0
export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
HOST="localhost"
PORT="8552"
DATASET="random"
in_out_tokens=("1024:1024" "8192:1024")
random_range_ratio=0.8
concurrencies=(4 8 16 32 64)
PROMPT_MULTIPLIER=10
PROF_CMD=(--profile --profile-num-steps 5 --profile-by-stage)

# ===================== Argument  =====================
DOCKER="rocm/sgl-dev:v0.5.9-rocm720-mi35x-20260326" # MI355
# DOCKER="lmsysorg/sglang:v0.5.9-cu130-runtime" # B200
SPECIAL_TAG="-bench"
SPECIAL_TAG2="-InferenceMax-FP8Cache"
if [ "$PROF_ENABLED" == "true" ]; then
    SPECIAL_TAG="-prof"
    concurrencies=(4)
    PROMPT_MULTIPLIER=2 # Faster for no cuda graph profiling

    # Debug
    in_out_tokens=("1024:1024")
    concurrencies=(4)
fi
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}${MTP_TAG}${SPECIAL_TAG}${SPECIAL_TAG2}"
FINISH_LOG="$LOG_DIR/Finish.log"
mkdir -p "$LOG_DIR"
touch "$FINISH_LOG"
if [ "$PROF_ENABLED" == "true" ]; then
    export SGLANG_TORCH_PROFILER_DIR=$LOG_DIR
fi

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

is_rocm_gpu_env() {
    [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1
}

list_profiler_dirs() {
    find "${LOG_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | grep -E '^[0-9]+(\.[0-9]+)?$' || true
}

rename_profiler_artifacts() {
    local input_tokens=$1
    local output_tokens=$2
    local c=$3
    local num_prompts=$4
    local before_dirs=$5
    local after_dirs=$6
    local target_dir_name="prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}"
    local target_dir_path="${LOG_DIR}/${target_dir_name}"
    local new_dirs

    new_dirs=$(comm -13 <(printf '%s\n' "${before_dirs}" | sort) <(printf '%s\n' "${after_dirs}" | sort))
    if [ -z "${new_dirs}" ]; then
        echo "No new profiler directory found under ${LOG_DIR}"
        return 0
    fi

    local src_dir src_dir_path
    src_dir=$(printf '%s\n' "${new_dirs}" | tail -n 1)
    src_dir_path="${LOG_DIR}/${src_dir}"
    if [ "${src_dir}" != "${target_dir_name}" ]; then
        if [ -e "${target_dir_path}" ]; then
            target_dir_path="${LOG_DIR}/${target_dir_name}_$(date +%s)"
            echo "Target directory exists. Using ${target_dir_path}"
        fi
        mv "${src_dir_path}" "${target_dir_path}"
        echo "Renamed profiler dir: ${src_dir} -> $(basename "${target_dir_path}")"
    fi

    local trace_file filename tp_rank new_name
    for trace_file in "${target_dir_path}"/*-TP-*.trace.json.gz; do
        [ -f "${trace_file}" ] || continue
        filename=$(basename "${trace_file}")
        tp_rank=$(sed -E 's/^.*-TP-([0-9]+)\.trace\.json\.gz$/\1/' <<< "${filename}")
        new_name="in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}-TP-${tp_rank}.trace.json.gz"
        mv "${trace_file}" "${target_dir_path}/${new_name}"
        echo "Renamed trace: ${filename} -> ${new_name}"
    done
}

rename_profiler_artifacts_by_stage() {
    local input_tokens=$1
    local output_tokens=$2
    local c=$3
    local num_prompts=$4
    local before_dirs=$5
    local after_dirs=$6
    local target_dir_name="prof_in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}"
    local target_dir_path="${LOG_DIR}/${target_dir_name}"
    local new_dirs

    new_dirs=$(comm -13 <(printf '%s\n' "${before_dirs}" | sort) <(printf '%s\n' "${after_dirs}" | sort))
    if [ -z "${new_dirs}" ]; then
        echo "No new profiler directory found under ${LOG_DIR}"
        return 0
    fi

    local src_dir src_dir_path
    src_dir=$(printf '%s\n' "${new_dirs}" | tail -n 1)
    src_dir_path="${LOG_DIR}/${src_dir}"
    if [ "${src_dir}" != "${target_dir_name}" ]; then
        if [ -e "${target_dir_path}" ]; then
            target_dir_path="${LOG_DIR}/${target_dir_name}_$(date +%s)"
            echo "Target directory exists. Using ${target_dir_path}"
        fi
        mv "${src_dir_path}" "${target_dir_path}"
        echo "Renamed profiler dir: ${src_dir} -> $(basename "${target_dir_path}")"
    fi

    local trace_file filename tp_rank stage new_name
    for trace_file in "${target_dir_path}"/*-TP-*.trace.json.gz; do
        [ -f "${trace_file}" ] || continue
        filename=$(basename "${trace_file}")
        tp_rank=$(sed -E 's/^.*-TP-([0-9]+)-(EXTEND|DECODE)\.trace\.json\.gz$/\1/' <<< "${filename}")
        stage=$(sed -E 's/^.*-TP-([0-9]+)-(EXTEND|DECODE)\.trace\.json\.gz$/\2/' <<< "${filename}")

        if [ "${tp_rank}" = "${filename}" ] || [ "${stage}" = "${filename}" ]; then
            echo "Skip unmatched trace name: ${filename}"
            continue
        fi

        new_name="in${input_tokens}_out${output_tokens}_conc${c}_p${num_prompts}-TP-${tp_rank}-${stage}.trace.json.gz"
        mv "${trace_file}" "${target_dir_path}/${new_name}"
        echo "Renamed trace: ${filename} -> ${new_name}"
    done
}

prof_cmd_has_profile_by_stage() {
    local arg
    for arg in "${PROF_CMD[@]}"; do
        if [ "${arg}" = "--profile-by-stage" ]; then
            return 0
        fi
    done
    return 1
}

start_server() {
    local logfile="${LOG_DIR}/server_${MODEL_NAME}.log"
    echo ">>> Starting SGLang server" | tee "$logfile"

    local cmd=(
        python3 -m sglang.launch_server
            --model $MODEL_PATH
            --tp 8
            --host $HOST
            --port $PORT
            --tool-call-parser glm47
            --reasoning-parser glm45
            --watchdog-timeout 1200
            --mem-fraction-static 0.85
            --kv-cache-dtype fp8_e4m3
            --disable-radix-cache
            --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}'
            --watchdog-timeout 1200
    )

    if is_rocm_gpu_env; then
        cmd+=(
            --nsa-prefill-backend tilelang
            --nsa-decode-backend tilelang
        )
    fi

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

    if [ ${#EXTRA_SERVER_ARGS[@]} -gt 0 ]; then
        cmd+=("${EXTRA_SERVER_ARGS[@]}")
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
        --random-input 2048
        --random-output 256
        --random-range-ratio "${random_range_ratio}"
        --max-concurrency 4 
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
            local num_prompts=$((c * PROMPT_MULTIPLIER))
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
                cmd+=("${PROF_CMD[@]}")
            fi

            if ! grep -q "$logfile" "$FINISH_LOG"; then
                echo "Running: $logfile"
                local profiler_dirs_before=""
                local profiler_dirs_after=""
                if [ "$PROF_ENABLED" == "true" ]; then
                    profiler_dirs_before=$(list_profiler_dirs) # Get the current folders under $LOG_DIR
                fi
                log_command "$logfile" "${cmd[@]}"
                echo "$logfile" >> "$FINISH_LOG"

                if [ "$PROF_ENABLED" == "true" ]; then
                    profiler_dirs_after=$(list_profiler_dirs) # Get the current folders under $LOG_DIR. This time will have another torch profiler folder
                    echo ">>> Processing profiler traces..."
                    if prof_cmd_has_profile_by_stage; then
                        rename_profiler_artifacts_by_stage "${input_tokens}" "${output_tokens}" "${c}" "${num_prompts}" "${profiler_dirs_before}" "${profiler_dirs_after}"
                    else
                        rename_profiler_artifacts "${input_tokens}" "${output_tokens}" "${c}" "${num_prompts}" "${profiler_dirs_before}" "${profiler_dirs_after}"
                    fi
                fi
            else
                echo "Found $logfile in ${FINISH_LOG}. Skipping."
            fi
        done
    done
}


# ------------------- Start -----------------
if [ "$PROF_ENABLED" == "true" ]; then
    PROF_SERVER_MODES=("default" "no-cuda-graph")
else
    PROF_SERVER_MODES=("default")
fi

BASE_LOG_DIR="$LOG_DIR"

for PROF_MODE in "${PROF_SERVER_MODES[@]}"; do
    EXTRA_SERVER_ARGS=()
    if [ "$PROF_MODE" == "no-cuda-graph" ]; then
        EXTRA_SERVER_ARGS=(--disable-cuda-graph)
        LOG_DIR="${BASE_LOG_DIR}/no-cuda-graph"
        mkdir -p "$LOG_DIR"
        FINISH_LOG="$LOG_DIR/Finish.log"
        touch "$FINISH_LOG"
        export SGLANG_TORCH_PROFILER_DIR=$LOG_DIR
    else
        LOG_DIR="${BASE_LOG_DIR}"
    fi

    echo ">>> [${PROF_MODE}] Starting server and benchmarks..."
    start_server
    warmup
    if [ "$PROF_MODE" == "default" ]; then
        accuracy_test
    fi
    run_benchmarks

    pkill -9 python || true
    sleep 10
done














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
