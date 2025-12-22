#!/bin/bash
set -x

# --- Parse Arguments ---
# This loop looks for the -m flag. 
# "m:" means the m option requires an argument.
while getopts "m:" opt; do
  case $opt in
    m)
      MODEL=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
  esac
done

# --- Check if MODEL is provided ---
if [ -z "$MODEL" ]; then
    echo "Error: Model name is required. Use -m <model_name>"
    exit 1
fi

# --- Configuration ---
PORT=30000
URL="http://127.0.0.1:$PORT"

# --- 1. Launch SGLang Server in the background ---
echo "Starting SGLang server..."
python ~/SGLang-benchmarks/SERVER.py --model $MODEL &

# --- 2. Wait for the Server to be Ready ---
# Polling loop: Checks the HTTP status code of the /health endpoint.
# -s: Silent mode (hides progress bar)
# -o /dev/null: Discards the empty response body
# -w "%{http_code}": Only prints the HTTP status code (e.g., 200)
until [ "$(curl -s -o /dev/null -w "%{http_code}" "$URL/health")" -eq 200 ]; do
    echo "Waiting for server to be ready at $URL/health..."
    sleep 2
done

echo -e "\nServer is READY!"

# --- 3. Warmup ---
python3 -m sglang.bench_serving \
	--port $PORT \
	--backend sglang \
	--model /data/grok-1-W4A8KV8/ \
	--tokenizer /data/Xenova/grok-1-tokenizer/ \
	--dataset-name random \
	--random-input 1024 \
	--random-output 1024 \
	--random-range-ratio 1.0 \
	--num-prompts 100 \
	--request-rate 4 

# --- 4. Profile ---
curl http://localhost:$PORT/start_profile -H "Content-Type: application/json"   -d '{
	"output_dir": "/tmp", 
	"activities": ["RPD"],
	"merge_profiles": false
}'
python3 -m sglang.bench_serving \
	--port $PORT \
	--backend sglang \
	--model /data/grok-1-W4A8KV8/ \
	--tokenizer /data/Xenova/grok-1-tokenizer/ \
	--dataset-name random \
	--random-input 1024 \
	--random-output 1024 \
	--random-range-ratio 1.0 \
	--num-prompts 300 \
	--request-rate 4 2>&1 | tee ${MODEL}_bench.log
curl http://localhost:$PORT/stop_profile -H "Content-Type: application/json"

# --- 5. Process RPD file ---
sqlite3 trace.rpd ".mode csv" ".header on" ".output trace.csv" "select * from top;" ".output stdout"
mv trace.rpd $MODEL.rpd
mv trace.csv $MODEL.csv


# --- 6. Kill server ---
kill $(ss -ltnp "sport = :$PORT" | grep -oP 'pid=\K\d+')