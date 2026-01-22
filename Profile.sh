#!/bin/bash
set -x

# Disable NUMA balancing for better performance consistency on multi-socket systems
echo 0 > /proc/sys/kernel/numa_balancing

# --- Default Values ---
MODEL=""
PROF="None" # Default to benchmark only

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

# --- Configuration ---
PORT=30000
URL="http://127.0.0.1:$PORT"
CONFIGS=( 
	# ilen olen concurrency 
    # "   200 200  1" 
    "   200 200  8"
    "   200 200  32"
    "  2048 200  1" 
    "  2048 200  8"
    "  2048 200  32"
    "700000 200  1" 
    "700000 200  2" 
    # "700000 200  4"
    # "700000 200  8"
    # "700000 200 16"        
)

# --- Define LaunchServer Function ---
LaunchServer() {
    echo "Starting SGLang server..."
    export SGLANG_INT4_WEIGHT=0 

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
            --mem-fraction-static 0.8 \
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


# Start the server
LaunchServer
# --- Main Loop ---
for config in "${CONFIGS[@]}"; do
    
    # Read variables from the config string
    read -r ilen olen concurrency  <<< "$config"
	prompt=$((concurrency*8))
    
    # Prepare output folder
    o_folder="$HOME/prof/0122/i${ilen}-o${olen}-n${prompt}-concurrency${concurrency}"
    mkdir -p "$o_folder"

    

    # --- Start Profiling ---
    if [ "$PROF" == "Torch_Profiler" ]; then
        echo "Starting Torch Profiler..."
        curl -X POST "http://localhost:$PORT/start_profile" \
             -H "Content-Type: application/json" \
             -d "{
                \"output_dir\": \"$o_folder\", 
                \"activities\": [\"CPU\", \"GPU\"],
                \"merge_profiles\": false
             }"
    fi

    # --- Run Benchmark ---
    python3 -m sglang.bench_serving \
        --port $PORT \
        --backend sglang \
        --model "$MODEL" \
        --dataset-name random \
        --random-input "$ilen" \
        --random-output "$olen" \
        --random-range-ratio 1.0 \
        --num-prompts "$prompt" \
        --max-concurrency  "$concurrency " 2>&1 | tee "${o_folder}/bench.log"
    
    # --- Stop Profiling ---
    if [ "$PROF" == "Torch_Profiler" ]; then
        echo "Stopping Torch Profiler..."
        curl -X POST "http://localhost:$PORT/stop_profile" \
             -H "Content-Type: application/json"
        
        # Give some time for the server to dump the trace files
        sleep 10

        # # --- Process TorchProfiler files ---
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
    kill $SERVER_PID
    sleep 10
