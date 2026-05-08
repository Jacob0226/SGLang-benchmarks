#!/usr/bin/env bash
# cascade_dsr1.sh — self-contained DSR1-0528 cascade benchmark.
#
# Single-purpose script: launch SGLang with HiCache (L1+L2+L3 file backend)
# on DSR1-0528 + multi-turn workload tuned to walk the cache hierarchy
# round by round, so MI355X vs B200 differences in HBM and DRAM capacity
# show up as TTFT / cache-hit-rate inflection points across rounds.
#
# Doesn't depend on HiCache.sh or any other helper script — just bash +
# sglang inside the docker image. Writes everything (logs, metadata,
# plot) into one bench folder so a copy of that folder is reproducible.
#
# Workload (hard-coded to land cache-fill events in observable rounds):
#   N=300 clients × R=4096 tokens/req × num_rounds (default 10)
#   per-rank cache occupancy ≈ 41 GB × round  (DSR1 MLA replicated, FP8 KV)
#   --hicache-size 192 GB host pool per rank, --hicache-size 0 (use sized)
#   write_through to all tiers; L3 = local file under /tmp (auto-cleaned)
#
# Override common knobs via flags below. Anything else: edit the
# constants in this file.
#
# Usage on MI355X:
#   ./cascade_dsr1.sh --tag MI355X_cascade \
#       --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507
# Usage on B200:
#   ./cascade_dsr1.sh --tag B200_cascade --docker lmsysorg/sglang:v0.5.9-cu130
#
# Plot once both done (matplotlib needed; runs inside docker if local
# python doesn't have it):
#   docker exec <container> python3 plot_cascade.py \
#       --tags MI355X_cascade B200_cascade --hicache-size 192

set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' || true

# ============================== Defaults ==============================
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TAG=""
DOCKER="untagged-docker"
TP_SIZE=8
HOST="localhost"
PORT="30000"
# --hicache-size:
#   "auto"   = pick the largest value that fits in this box's MemAvailable
#              minus 200 GB host headroom, divided across TP ranks. Lets
#              MI355X (3 TB DRAM) max out to ~320 GB/rank while B200 with
#              2 TB DRAM lands around ~192 GB/rank automatically — same
#              command line on both, each platform shows its real ceiling.
#   <number> = explicit per-rank GB (override auto)
HICACHE_SIZE="auto"
HOST_HEADROOM_GB=200
NUM_CLIENTS=300
NUM_ROUNDS=10
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
WAIT_FOR_SERVER_SEC=900     # 15 min cap; bail out if SGLang doesn't /health

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL_PATH="$2"; shift 2;;
    --tag)            TAG="$2"; shift 2;;
    --docker)         DOCKER="$2"; shift 2;;
    --tp)             TP_SIZE="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --hicache-size)   HICACHE_SIZE="$2"; shift 2;;
    --host-headroom-gb) HOST_HEADROOM_GB="$2"; shift 2;;
    --num-clients)    NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)     NUM_ROUNDS="$2"; shift 2;;
    --request-length) REQUEST_LENGTH="$2"; shift 2;;
    --max-parallel)   MAX_PARALLEL="$2"; shift 2;;
    --request-rate)   REQUEST_RATE="$2"; shift 2;;
    -h|--help) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \?//' | head -n -1; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

if [ -z "$TAG" ]; then
  echo "ERROR: --tag is required (e.g. MI355X_cascade or B200_cascade)" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model path doesn't exist: $MODEL_PATH" >&2
  exit 1
fi
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# ============================== Auto-size hicache ==============================
# When --hicache-size is "auto", measure the box's actual MemAvailable and
# size the host KV pool to its safe maximum. This is the whole point of the
# cross-platform comparison: MI355X (3 TB DRAM) and B200 (~2 TB DRAM) each
# get to use their own DRAM ceiling, so the cascade reflects true platform
# capacity. Headroom (default 200 GB) covers OS page cache during model
# load, HiCache pinned-memory staging buffers, NUMA fragmentation slack,
# and the SGLang process group's anonymous memory.
if [ "$HICACHE_SIZE" = "auto" ]; then
  MEM_AVAIL_GB=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
  USABLE_GB=$(( MEM_AVAIL_GB - HOST_HEADROOM_GB ))
  if [ "$USABLE_GB" -le 0 ]; then
    echo "ERROR: only ${MEM_AVAIL_GB} GB MemAvailable, can't reserve" \
         "${HOST_HEADROOM_GB} GB headroom. Use --host-headroom-gb or" \
         "--hicache-size N." >&2
    exit 1
  fi
  PER_RANK=$(( USABLE_GB / TP_SIZE ))
  PER_RANK=$(( PER_RANK / 32 * 32 ))   # 32 GB align for clean numbers
  [ "$PER_RANK" -lt 32  ] && PER_RANK=32
  [ "$PER_RANK" -gt 512 ] && PER_RANK=512   # diminishing returns past this
  HICACHE_SIZE="$PER_RANK"
  echo ">>> auto --hicache-size: ${HICACHE_SIZE} GB per rank" \
       "(MemAvailable=${MEM_AVAIL_GB} GB, headroom=${HOST_HEADROOM_GB} GB," \
       "TP=${TP_SIZE} ranks; total host pool = $(( HICACHE_SIZE * TP_SIZE )) GB)"
else
  echo ">>> manual --hicache-size: ${HICACHE_SIZE} GB per rank" \
       "(total host pool = $(( HICACHE_SIZE * TP_SIZE )) GB)"
fi

# ============================== Output dir ==============================
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-cascade-${TAG}/L3_file/size_${HICACHE_SIZE}"
mkdir -p "$LOG_DIR"
echo ">>> bench folder: $LOG_DIR"

# ============================== Platform detection ==============================
is_rocm() { [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; }

# Detect HBM size for sanity messaging.
get_gpu_hbm_gb() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showmeminfo vram 2>/dev/null \
      | awk '/VRAM Total Memory/ {print int($NF/1024/1024/1024); exit}'
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | awk '{print int($1/1024)}'
  else
    echo 0
  fi
}
HBM_GB=$(get_gpu_hbm_gb)
echo ">>> detected HBM: ${HBM_GB} GB per GPU, TP=${TP_SIZE}"

# ============================== NUMA interleave ==============================
NUMA_NODES=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null \
              | sed 's|.*/node||' | sort -n | paste -sd,)
NUMACTL_PREFIX=()
if command -v numactl >/dev/null 2>&1 && [[ "$NUMA_NODES" == *,* ]]; then
  NUMACTL_PREFIX=(numactl --interleave="$NUMA_NODES")
  echo ">>> NUMA interleave: --interleave=${NUMA_NODES}"
fi

# ============================== Host snapshot ==============================
{
  echo "=== cascade_dsr1.sh host snapshot @ $(date '+%F %T %Z') ==="
  echo "--- lscpu ---"
  lscpu | grep -E "Architecture|Vendor|Model name|CPU\(s\)|Socket|Core|Thread|NUMA"
  echo "--- /proc/meminfo ---"
  grep -E "^MemTotal:|^MemAvailable:|^MemFree:|^Cached:" /proc/meminfo
  echo "--- NUMA per-node DRAM ---"
  for n in /sys/devices/system/node/node[0-9]*; do
    [ -d "$n" ] || continue
    nid=$(basename "$n" | sed 's/node//')
    mem=$(awk '/MemTotal/{print int($4/1024/1024)" GB"}' "$n/meminfo")
    cpus=$(cat "$n/cpulist" 2>/dev/null)
    printf "  node %s: DRAM=%s, CPUs=%s\n" "$nid" "$mem" "$cpus"
  done
  echo "--- GPU info ---"
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showid 2>&1 | grep "Device Name" | head -1
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
  fi
} | tee "$LOG_DIR/host_info.log" >/dev/null

# ============================== Bench meta ==============================
python3 - "$LOG_DIR/bench_meta.json" <<PY
import json, sys
data = {
    "cache_mode": "L3_file",
    "model_path": "$MODEL_PATH",
    "model_family": "deepseek",
    "tp_size": $TP_SIZE,
    "kv_cache_dtype": "fp8_e4m3",
    "mem_fraction_static": 0.85,
    "host_headroom_gb": 200,
    "hbm_gb": $HBM_GB,
    "device_pool_gb": int($HBM_GB * 0.85 - 84),
    "hicache_size_gb": $HICACHE_SIZE,
    "bench_mode": "multiturn",
    "num_clients": $NUM_CLIENTS,
    "num_rounds": $NUM_ROUNDS,
    "request_length": $REQUEST_LENGTH,
    "output_length": $OUTPUT_LENGTH,
    "max_parallel": $MAX_PARALLEL,
    "request_rate": $REQUEST_RATE,
    "enable_round_barrier": True,
    "disable_random_sample": True,
    "tag": "$TAG",
    "docker": "$DOCKER",
}
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
PY

# ============================== L3 file store (in /tmp, container-local) ==============================
HICACHE_FILE_STORE_DIR="/tmp/cascade_dsr1_l3_${TAG}_${HICACHE_SIZE}"
rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null
mkdir -p "$HICACHE_FILE_STORE_DIR"
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_FILE_STORE_DIR"
trap 'rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null; pkill -9 -f sglang.launch_server 2>/dev/null || true' EXIT

# ============================== Server launch ==============================
SERVER_LOG="$LOG_DIR/server.log"
SERVER_CMD=(
  "${NUMACTL_PREFIX[@]}"
  python3 -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --tp "$TP_SIZE"
    --host "$HOST" --port "$PORT"
    --mem-fraction-static 0.85
    --watchdog-timeout 1200
    --enable-metrics
    --enable-cache-report
    --trust-remote-code
    --reasoning-parser deepseek-r1
    --kv-cache-dtype fp8_e4m3
    --page-size 64
    --context-length 65536
    --chunked-prefill-size 32768
    --max-prefill-tokens 32768
    --enable-hierarchical-cache
    --hicache-size "$HICACHE_SIZE"
    --hicache-mem-layout page_first_direct
    --hicache-io-backend kernel
    --hicache-write-policy write_through
    --hicache-storage-backend file
    --hicache-storage-prefetch-policy best_effort
)
if is_rocm; then
  export ROCM_QUICK_REDUCE_QUANTIZATION=INT4 SAFETENSORS_FAST_GPU=1
  SERVER_CMD+=(--attention-backend aiter)
else
  export SGL_ENABLE_JIT_DEEPGEMM=1
  SERVER_CMD+=(
    --attention-backend trtllm_mla
    --moe-runner-backend flashinfer_trtllm
    --enable-flashinfer-allreduce-fusion
  )
fi

echo ">>> launching SGLang (cache_mode=L3_file, hicache-size=${HICACHE_SIZE} GB)"
echo "${SERVER_CMD[*]}" | tee "$SERVER_LOG"
"${SERVER_CMD[@]}" 2>&1 | tee -a "$SERVER_LOG" &

# ============================== Wait for /health ==============================
deadline=$(( $(date +%s) + WAIT_FOR_SERVER_SEC ))
echo ">>> waiting up to ${WAIT_FOR_SERVER_SEC}s for /health 200..."
while [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERROR: server didn't come up — see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
echo ">>> server ready"

# ============================== Warmup ==============================
echo ">>> warmup (random 1024/128 × 8 prompts × 4 concurrency)"
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --model "$MODEL_PATH" --dataset-name random \
  --random-input 1024 --random-output 128 --random-range-ratio 1.0 \
  --max-concurrency 4 --num-prompt 8 --output-file /dev/null \
  2>&1 | tee "$LOG_DIR/warmup.log"

# ============================== Bench: multiturn ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
if [ -z "$BENCH_SCRIPT" ]; then
  echo "ERROR: bench_multiturn.py not found" >&2
  exit 1
fi

# Flush radix tree so each run starts cold (HiCache itself controls L2/L3).
curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
sleep 2

echo ">>> running cascade multiturn (N=${NUM_CLIENTS} × R=${REQUEST_LENGTH} × ${NUM_ROUNDS} rounds)"
python3 "$BENCH_SCRIPT" \
  --host "$HOST" --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --num-clients "$NUM_CLIENTS" \
  --num-rounds "$NUM_ROUNDS" \
  --request-length "$REQUEST_LENGTH" \
  --output-length "$OUTPUT_LENGTH" \
  --max-parallel "$MAX_PARALLEL" \
  --request-rate "$REQUEST_RATE" \
  --ready-queue-policy random \
  --log-file "$LOG_DIR/bench_multiturn.jsonl" \
  --tag "${MODEL_NAME}-${TAG}" \
  --disable-random-sample \
  --disable-auto-run \
  --enable-round-barrier \
  2>&1 | tee "$LOG_DIR/bench_multiturn.log"

# ============================== Cleanup ==============================
echo ">>> stopping server"
pkill -9 -f sglang.launch_server || true
sleep 5

if [ -d "$HICACHE_FILE_STORE_DIR" ]; then
  sz=$(du -sh "$HICACHE_FILE_STORE_DIR" 2>/dev/null | cut -f1)
  rm -rf "$HICACHE_FILE_STORE_DIR"
  echo ">>> cleaned L3 file store (${sz:-?} reclaimed)"
fi

echo ">>> done. results in: $LOG_DIR"
echo "    plot: python3 plot_cascade.py --tags $TAG --hicache-size $HICACHE_SIZE"
