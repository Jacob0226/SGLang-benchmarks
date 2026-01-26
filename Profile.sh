#!/bin/bash
set -x

# Usage:
# ./Profile.sh --model /data/huggingface/hub/deepseek-ai/DeepSeek-V3.2-Exp --prof Torch_Profiler
# ./Profile.sh --model /data/huggingface/hub/deepseek-ai/DeepSeek-V3.2-Exp

# Disable NUMA balancing for better performance consistency on multi-socket systems
echo 0 > /proc/sys/kernel/numa_balancing

# --- Default Values ---
MODEL=""
PROF="None" # Default to benchmark only
TAG=0126


# --- Parse Arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --prof)
      PROF="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Usage: $0 --model /path/to/model --prof [None|Torch_Profiler]"
      exit 1
      ;;
  esac
done

# --- Check arguments ---
if [ -z "$MODEL" ]; then
    echo "Error: Missing required argument --model"
    exit 1
fi

# --- Define LaunchServer Function ---
LaunchServer() {
    echo "Starting SGLang server..."

    # Launch server in background
    if command -v rocminfo > /dev/null 2>&1 || [ -d "/opt/rocm" ]; then
        echo ">>> ROCm environment detected."
        export RCCL_MSCCL_ENABLE=0 
        export SGLANG_USE_AITER=1
        export PYTHONPATH="${PYTHONPATH}:/opt/tilelang"
        python3 -m sglang.launch_server \
            --model "$MODEL" \
            --mem-fraction-static 0.7 \
            --tp 8 \
            --port "$PORT" \
            --trust-remote-code \
            --disable-radix-cache \
            --chunked-prefill-size 131072 \
            --nsa-prefill-backend tilelang \
            --nsa-decode-backend tilelang \
            --attention-backend aiter 2>&1 | tee sglang_server.log &
    else
        echo ">>> NVIDIA environment detected."
        # NVIDIA (B200/H100) 
        python3 -m sglang.launch_server \
            --model "$MODEL" \
            --mem-fraction-static 0.7 \
            --tp 8 \
            --port "$PORT" \
            --trust-remote-code \
            --disable-radix-cache  2>&1 | tee sglang_server.log &
            # --chunked-prefill-size 131072
    fi
    
    SERVER_PID=$!

    # --- Wait for the Server to be Ready ---
    until [ "$(curl -s -o /dev/null -w "%{http_code}" "$URL/health")" -eq 200 ]; do
        echo "Waiting for server to be ready at $URL/health..."
        sleep 5
    done

    echo -e "\nServer is READY! Starting warmup..."

    # --- Warmup ---
    python3 -m sglang.bench_serving \
        --port $PORT \
        --backend sglang \
        --model "$MODEL" \
        --dataset-name random \
        --random-input 256 \
        --random-output 256 \
        --random-range-ratio 1.0 \
        --num-prompts 8 \
        --max-concurrency 2
}

# --------------------------------Start --------------------------------------------
# --- Configuration ---
PORT=30000
URL="http://127.0.0.1:$PORT"
CONFIGS=( 
    # ilen olen concurrency 
    " 8000 1000 1"
    " 8000 1000 2"
    " 8000 1000 4"
    " 8000 1000 8"

    " 1000 1000 1"
    " 1000 1000 2"
    " 1000 1000 4"
    " 1000 1000 8"

    " 1000 8000 1"
    " 1000 8000 2"
    " 1000 8000 4"
    " 1000 8000 8"
    # "   200 200  1" 
    # "   200 200  8"
    # "   200 200  32"
    # "  2048 200  1" 
    # "  2048 200  8"
    # "  2048 200  32"
    # "700000 200  1" 
    # "700000 200  2" 
    # "700000 200  4"
    # "700000 200  8"
    # "700000 200 16"        
)

N_LOOOP=3
WORK="bench"
if [ "$PROF" == "Torch_Profiler" ]; then # if profiling, run only 1 time
    N_LOOOP=1
    WORK="prof"
fi 

ROOT_FOLDER="$HOME/prof/${TAG}_${WORK}"
FINISH_LOG="$ROOT_FOLDER/Finish.log"
prof_cmd=""
if [ "$PROF" == "Torch_Profiler" ]; then
    prof_cmd="--profile"
    mkdir -p "$ROOT_FOLDER"
    export SGLANG_TORCH_PROFILER_DIR=$HOME/prof/${TAG}_${WORK}
fi

# Start the server
LaunchServer

# --- Main Loop ---
for config in "${CONFIGS[@]}"; do
    
    # Read variables from the config string
    read -r ilen olen concurrency  <<< "$config"
    prompt=$((concurrency*8))
    
    # Prepare output folder
    o_folder="$ROOT_FOLDER/i${ilen}-o${olen}-n${prompt}-concurrency${concurrency}"
    mkdir -p "$o_folder"

    # --- Run Benchmark ---
    # Fixed the Python-style loop to Bash-style
    for (( i=1; i<=N_LOOOP; i++ )); do
        LOG_FILE="${o_folder}/bench_${i}.log"
        
        # Check if already benchmarked
        if ! grep -q "$LOG_FILE" "$FINISH_LOG"; then
            echo "Running: $LOG_FILE"
            python3 -m sglang.bench_serving \
                --port $PORT \
                --backend sglang \
                --model "$MODEL" \
                --dataset-name random \
                --random-input "$ilen" \
                --random-output "$olen" \
                --random-range-ratio 1.0 \
                --num-prompts "$prompt" \
                --max-concurrency "$concurrency" \
                $prof_cmd 2>&1 | tee "$LOG_FILE"
            
            echo "$LOG_FILE" >> "$FINISH_LOG"
        else
            echo "Found $LOG_FILE in ${FINISH_LOG}. Skipping."
        fi
    done
    
    # --- Process TorchProfiler files ---
    if [ "$PROF" == "Torch_Profiler" ]; then
        mv $SGLANG_TORCH_PROFILER_DIR/*.gz $o_folder/
        echo "Process TorchProfiler files"
        # for file in "$o_folder"/*; do
        #     if [[ "$file" == *.trace.json.gz ]]; then
        #         # Get filename without the extension
        #         base_name=$(basename "$file" .trace.json.gz)
        #         python3 "$HOME/SGLang-benchmarks/parse_torch_profiler.py" \
        #             --file "$file" \
        #             --out "${o_folder}/${base_name}.csv"
        #     fi
        # done
    fi

done

# Kill the server to clean up for the next config
pkill -9 python
sleep 10