#!/usr/bin/env bash
# cascade_dsr1.sh — self-contained DSR1-0528 cascade benchmark.
#
# Launch SGLang on DSR1-0528 with a multi-turn workload tuned to walk the
# cache hierarchy round by round, so MI355X vs B200 differences in HBM
# and DRAM capacity show up as TTFT / cache-hit-rate inflection points.
# Same script supports the three cache hierarchy depths via --cache-mode:
#
#   L1       GPU radix cache only (default SGLang, no HiCache flags).
#            No host pool, no disk — baseline for "GPU memory only" runs.
#   L2       L1 + host DRAM pool (--enable-hierarchical-cache + --hicache-size).
#            No external storage backend.
#   L3_file  L1 + L2 + local-file external store at /tmp (NVMe-backed in
#            most docker images). The default; this is the full cascade
#            run that gives you the L1→L2→L3 spike pattern.
#
# Doesn't depend on HiCache.sh or any other helper script — just bash +
# sglang inside the docker image. Writes everything (logs, metadata,
# plot) into one bench folder so a copy of that folder is reproducible.
#
# Workload (hard-coded to land cache-fill events in observable rounds):
#   N=300 clients × R=4096 tokens/req × num_rounds (default 10)
#   per-rank cache occupancy ≈ 41 GB × round  (DSR1 MLA replicated, FP8 KV)
#   write_through to all tiers; L3 store goes under /tmp (auto-cleaned)
#
# All workload defaults below match the original a0367ca cascade_dsr1.sh
# config (the one that produced the 2026-05-08 reference jsonl). Pass
# --num-rounds 15 / --hicache-size N / --chunked-prefill-size N etc. on
# the CLI to override for ablation runs (e.g. extending the cascade past
# the L2 ceiling, sweeping host-pool size, or stress-testing the AMD
# scheduler with longer chunked-prefills).
#
# Override common knobs via flags below. Anything else: edit the
# constants in this file.
#
# GSM8K precheck (ON by default; --no-gsm8k-precheck to opt out): runs a
# 1200-question GSM8K accuracy bench AFTER warmup and BEFORE the cascade
# bench. Log file name + workload match GLM.sh's accuracy_test()
# (--num-questions 1200 --parallel 1200, log -> $LOG_DIR/Accuracy_GSM8K.log)
# so the cascade and GLM benchmark trees produce comparable accuracy
# artifacts. The parsed accuracy is also recorded in bench_meta.json so
# each cascade run carries a model-correctness signature. After the GSM8K
# phase, /flush_cache + drop_caches (best-effort) clean SGLang radix tree
# + OS page cache so the cascade bench still starts cold. Adds ~70-90 sec
# per run on MI355X (parallel=1200 finishes the 1200 questions in ~70 s wall).
#
# Usage on MI355X (run all three modes for a complete cascade picture):
#   ./cascade_dsr1.sh --tag MI355X_cascade --cache-mode L1 \
#       --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507
#   ./cascade_dsr1.sh --tag MI355X_cascade --cache-mode L2 \
#       --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507
#   ./cascade_dsr1.sh --tag MI355X_cascade --cache-mode L3_file \
#       --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507
# Usage on B200 (mirror): same three commands with --tag B200_cascade
#   and --docker lmsysorg/sglang:v0.5.9-cu130.
#
# All three modes write under the same parent dir
#   results/<docker>/<MODEL>-cascade-<TAG>/{L1,L2/size_<N>,L3_file/size_<N>}/
# so plot_cascade.py can pick them up by absolute path easily.
#
# Plot once both done (matplotlib needed; runs inside docker if local
# python doesn't have it). Pass each platform's bench_multiturn.jsonl
# explicitly via --MI355X / --B200, and use absolute paths so the output
# PNG lands in a predictable spot regardless of cwd:
#   python3 plot_cascade.py \
#       --Title "DSR1-0528 cascade L3_file: MI355X vs B200" \
#       --MI355X $HOME/SGLang-benchmarks/results/<rocm-docker>/DeepSeek-R1-0528-cascade-MI355X_cascade/L3_file/size_<N>/bench_multiturn.jsonl \
#       --B200   $HOME/SGLang-benchmarks/results/<cuda-docker>/DeepSeek-R1-0528-cascade-B200_cascade/L3_file/size_<N>/bench_multiturn.jsonl \
#       --out    $HOME/SGLang-benchmarks/results/cascade_dsr1.png

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
# --cache-mode picks one point in the cache hierarchy:
#   L1       GPU radix cache only (no --enable-hierarchical-cache)
#   L2       + host DRAM pool      (no external storage)
#   L3_file  + local file backend  (full cascade; the default)
CACHE_MODE="L3_file"
# --hicache-size: per-rank host KV pool in GB. Default "auto" lets each
# platform max out its own DRAM (MI355X ~320 GB/rank on a 3 TB box,
# B200 ~192 GB/rank on a 2 TB box) — same command line on both. Pass an
# explicit number for cross-platform fairness (same host pool size on
# both MI355X and B200), e.g. --hicache-size 192 to reproduce the
# original a0367ca config.
#   "auto"   = pick the largest value that fits in this box's MemAvailable
#              minus host headroom, divided across TP ranks. Use this for
#              "let each platform win on its own DRAM" runs.
#   <number> = explicit per-rank GB (cross-platform fairness — same host
#              pool size on both MI355X and B200).
HICACHE_SIZE=auto
# HOST_HEADROOM_GB: how much DRAM to reserve OUTSIDE the HiCache host pool.
# On a 3 TB MI355X box with TP=8, the auto sizer does:
#     PER_RANK = (MemAvail - HOST_HEADROOM) / 8, aligned down to 32 GB
# i.e. raising headroom by 256 GB drops PER_RANK by 32 GB.
#
# Headroom must absorb (write_through hot path, MI355X DSR1-FP8):
#   ~671 GB DSR1-FP8 weights briefly held in host RAM during model load
#   ~ 80 GB HiCache pinned-memory staging buffers (TP=8, page_first_direct)
#   ~ 50 GB OS page cache for /data/huggingface/...
#   ~ 30 GB SGLang process group anonymous memory (per-rank workers)
#   + safety margin for write_through bursts that briefly double-allocate.
#
# Default 400 GB was chosen after the 200 GB run on jacchang_HiCache pinned
# 2560/3024 GB and slowed L2 prefill 10-200x (cascade_dsr1.sh L2 v3 run,
# 2026-05-12). On a 3 TB box this still gives PER_RANK=320 GB / total 2560
# GB pool, because raising headroom 200->400 only shaved 25 GB/rank pre-
# alignment. Bump higher (e.g. 900) if write_through is starving the host.
HOST_HEADROOM_GB=400
NUM_CLIENTS=300
NUM_ROUNDS=15
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
# CUDA graph capture range: SGLang pre-captures graphs for batch sizes
# 1..N, each costing GPU memory. SGLang's own default (~512) wastes
# memory because round-barrier + max-parallel caps actual batch at
# <=MAX_PARALLEL. When unset (0), we couple to MAX_PARALLEL so capture
# matches reality.
CUDA_GRAPH_MAX_BS=0
# Prefill chunking. Default 32768 matches the original a0367ca config
# (SGLang's own default is 8192). a0c0522 bumped this to 65536 to save
# chunk-switch overhead on the round-11+ 60K-token prompts, but that
# change has been linked to AMD scheduler instability under high
# concurrency — keep at 32768 by default and pass --chunked-prefill-size
# 65536 explicitly for ablation runs.
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
WAIT_FOR_SERVER_SEC=900     # 15 min cap; bail out if SGLang doesn't /health
# GSM8K precheck. ON by default so every cascade run carries a model-
# correctness signature in bench_meta.json (paired with the per-run
# Accuracy_GSM8K.log under $LOG_DIR). Adds ~70-90 sec per cascade run;
# pass --no-gsm8k-precheck to opt out when you only care about cache /
# perf numbers. Defaults match GLM.sh's accuracy_test() (1200 / 1200)
# for cross-bench parity. Higher parallel for cascade-style 4096-token
# prompts can crash the AMD scheduler, but GSM8K prompts are short
# (~600 tokens) so parallel=1200 is safe and matches what we already
# use in GLM benchmarks.
GSM8K_PRECHECK="true"
GSM8K_NUM_QUESTIONS=1200
GSM8K_PARALLEL=1200

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL_PATH="$2"; shift 2;;
    --tag)            TAG="$2"; shift 2;;
    --docker)         DOCKER="$2"; shift 2;;
    --cache-mode)     CACHE_MODE="$2"; shift 2;;
    --tp)             TP_SIZE="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --hicache-size)   HICACHE_SIZE="$2"; shift 2;;
    --host-headroom-gb) HOST_HEADROOM_GB="$2"; shift 2;;
    --num-clients)    NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)     NUM_ROUNDS="$2"; shift 2;;
    --request-length) REQUEST_LENGTH="$2"; shift 2;;
    --max-parallel)   MAX_PARALLEL="$2"; shift 2;;
    --request-rate)   REQUEST_RATE="$2"; shift 2;;
    --cuda-graph-max-bs)   CUDA_GRAPH_MAX_BS="$2"; shift 2;;
    --chunked-prefill-size) CHUNKED_PREFILL_SIZE="$2"; shift 2;;
    --max-prefill-tokens)   MAX_PREFILL_TOKENS="$2"; shift 2;;
    --gsm8k-precheck) GSM8K_PRECHECK="true"; shift 1;;
    --no-gsm8k-precheck) GSM8K_PRECHECK="false"; shift 1;;
    --gsm8k-num-questions) GSM8K_NUM_QUESTIONS="$2"; shift 2;;
    --gsm8k-parallel) GSM8K_PARALLEL="$2"; shift 2;;
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
case "$CACHE_MODE" in
  L1|L2|L3_file) ;;
  *) echo "ERROR: --cache-mode must be one of: L1, L2, L3_file (got: '$CACHE_MODE')" >&2; exit 1;;
esac
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# ============================== Auto-size hicache ==============================
# L1 doesn't use a host pool at all (just GPU radix cache), so any
# --hicache-size value is meaningless. We skip sizing entirely and set
# HICACHE_SIZE=0 as a sentinel (treated as null in bench_meta.json).
#
# When --hicache-size is "auto" for L2 / L3_file, measure the box's
# actual MemAvailable and size the host KV pool to its safe maximum.
# This is the whole point of the cross-platform comparison: MI355X
# (3 TB DRAM) and B200 (~2 TB DRAM) each get to use their own DRAM
# ceiling, so the cascade reflects true platform capacity. Headroom
# (default 400 GB; see HOST_HEADROOM_GB block above for breakdown)
# covers DSR1-FP8 weights held briefly in host DRAM during model load,
# OS page cache, HiCache pinned-memory staging buffers, NUMA frag slack,
# and the SGLang process group's anonymous memory.
if [ "$CACHE_MODE" = "L1" ]; then
  HICACHE_SIZE=0
  echo ">>> CACHE_MODE=L1: skipping host pool sizing (GPU radix cache only)"
elif [ "$HICACHE_SIZE" = "auto" ]; then
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
# L1 has no hicache-size, so its dir is just .../<TAG>/L1/.
# L2 / L3_file include the size in the path so a later size sweep doesn't
# overwrite earlier runs.
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-cascade-${TAG}"
if [ "$CACHE_MODE" = "L1" ]; then
  LOG_DIR="${BASE_LOG_DIR}/L1"
else
  LOG_DIR="${BASE_LOG_DIR}/${CACHE_MODE}/size_${HICACHE_SIZE}"
fi
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
# Emit hicache_size_gb as JSON null when cache_mode=L1 (the field is
# meaningless without a host pool), otherwise the integer per-rank GB.
# We inject Python's None (not the JSON literal `null`) since the value
# is interpolated into a Python source heredoc; json.dumps then emits
# the proper JSON null.
if [ "$CACHE_MODE" = "L1" ]; then
  META_HICACHE_VAL="None"
else
  META_HICACHE_VAL="$HICACHE_SIZE"
fi
# GSM8K precheck fields. Initialized with accuracy=None; updated in-place
# after the GSM8K phase actually runs (see update_bench_meta_accuracy).
if [ "$GSM8K_PRECHECK" = "true" ]; then
  META_GSM8K_ENABLED="True"
  META_GSM8K_NQ="$GSM8K_NUM_QUESTIONS"
else
  META_GSM8K_ENABLED="False"
  META_GSM8K_NQ="None"
fi
python3 - "$LOG_DIR/bench_meta.json" <<PY
import json, sys
data = {
    "cache_mode": "$CACHE_MODE",
    "model_path": "$MODEL_PATH",
    "model_family": "deepseek",
    "tp_size": $TP_SIZE,
    "kv_cache_dtype": "fp8_e4m3",
    "mem_fraction_static": 0.85,
    "host_headroom_gb": 200,
    "hbm_gb": $HBM_GB,
    "device_pool_gb": int($HBM_GB * 0.85 - 84),
    "hicache_size_gb": $META_HICACHE_VAL,
    "bench_mode": "multiturn",
    "num_clients": $NUM_CLIENTS,
    "num_rounds": $NUM_ROUNDS,
    "request_length": $REQUEST_LENGTH,
    "output_length": $OUTPUT_LENGTH,
    "max_parallel": $MAX_PARALLEL,
    "request_rate": $REQUEST_RATE,
    "enable_round_barrier": True,
    "disable_random_sample": True,
    "gsm8k_precheck_enabled": $META_GSM8K_ENABLED,
    "gsm8k_precheck_num_questions": $META_GSM8K_NQ,
    "gsm8k_precheck_accuracy": None,  # filled in after the GSM8K phase
    "tag": "$TAG",
    "docker": "$DOCKER",
}
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
PY

# Helper: update gsm8k_precheck_accuracy in bench_meta.json after GSM8K
# runs. Takes one argument: a Python literal (number or None).
update_bench_meta_accuracy() {
  local acc_pyval="$1"
  python3 - <<PY
import json
p = "$LOG_DIR/bench_meta.json"
with open(p) as f:
    data = json.load(f)
data["gsm8k_precheck_accuracy"] = $acc_pyval
with open(p, "w") as f:
    json.dump(data, f, indent=2)
PY
}

# ============================== L3 file store (in /tmp, container-local) ==============================
# Only set up the on-disk L3 store for L3_* modes — L1 / L2 don't need it.
HICACHE_FILE_STORE_DIR=""
if [[ "$CACHE_MODE" == L3_* ]]; then
  HICACHE_FILE_STORE_DIR="/tmp/cascade_dsr1_l3_${TAG}_${HICACHE_SIZE}"
  rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null
  mkdir -p "$HICACHE_FILE_STORE_DIR"
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_FILE_STORE_DIR"
fi
# Trap uses :- so L1/L2 (HICACHE_FILE_STORE_DIR="") doesn't trip set -u.
# Also kill the cache_monitor sidecar if it's still alive on exit.
trap 'rm -rf "${HICACHE_FILE_STORE_DIR:-}" 2>/dev/null; [ -n "${CACHE_MONITOR_PID:-}" ] && kill "${CACHE_MONITOR_PID}" 2>/dev/null; pkill -9 -f sglang.launch_server 2>/dev/null || true' EXIT

# ============================== Server launch ==============================
SERVER_LOG="$LOG_DIR/server.log"
# Default cuda-graph-max-bs to MAX_PARALLEL when user didn't override.
# That keeps captured graphs to the actual concurrent batch sizes the
# bench will produce (under round-barrier + max-parallel client cap).
if [ "$CUDA_GRAPH_MAX_BS" -le 0 ]; then
  CUDA_GRAPH_MAX_BS="$MAX_PARALLEL"
fi
echo ">>> cuda-graph-max-bs=${CUDA_GRAPH_MAX_BS} (coupled to max-parallel)"
echo ">>> chunked-prefill-size=${CHUNKED_PREFILL_SIZE}, max-prefill-tokens=${MAX_PREFILL_TOKENS}"

# Common cmd (everything that doesn't depend on cache mode).
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
    --kv-cache-dtype fp8_e4m3
    --context-length 65536
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS"
)
# Cache-mode-specific HiCache flags. L1 = no HiCache (default RadixCache);
# L2 = HiCache to host DRAM; L3_file adds the on-disk file backend.
case "$CACHE_MODE" in
  L1)
    : # nothing — default SGLang behavior (GPU-only RadixCache)
    ;;
  L2)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-mem-layout page_first_direct
      --hicache-io-backend kernel
      --hicache-write-policy write_through
    )
    ;;
  L3_file)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-mem-layout page_first_direct
      --hicache-io-backend kernel
      --hicache-write-policy write_through
      --hicache-storage-backend file
      --hicache-storage-prefetch-policy best_effort
    )
    ;;
esac

if is_rocm; then
  # MI355X DSR1-FP8 server config — aligned with InferenceX
  # benchmarks/single_node/dsr1_fp8_mi355x.sh + AMD ROCm 7.0 official doc
  # (https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html).
  #
  # SGLANG_USE_AITER=1                       enable aiter kernels (docker
  #                                          default already sets this; pinned
  #                                          here so behavior is explicit).
  # ROCM_QUICK_REDUCE_QUANTIZATION=NONE      do NOT quantize AllReduce. INT4
  #                                          (the value the SKILL.md notes for
  #                                          GLM-5) is too lossy for DSR1 and
  #                                          tanks GSM8K accuracy from 0.93+
  #                                          to ~0.01 (verified 2026-05-12 in
  #                                          tools/gsm8k_dsr1_minfix_test.sh).
  # Also dropped from SERVER_CMD common args:
  #   --page-size 64           non-standard for FP8 KV + aiter MLA;
  #                            no public reference uses page_size>1 (AMD doc,
  #                            InferenceX, Clint Greene's MXFP4 cmd all use
  #                            default page_size=1). page_size=64 corrupts KV
  #                            indexing → garbage outputs + occasional HSA fault.
  #   --reasoning-parser deepseek-r1   strips <think>...</think> server-side;
  #                            for cascade workload (output_length=1) and
  #                            GSM8K precheck the parser can eat the actual
  #                            answer text. Not needed for these benches.
  export SAFETENSORS_FAST_GPU=1
  export SGLANG_USE_AITER=1
  export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
  SERVER_CMD+=(--attention-backend aiter)
else
  export SGL_ENABLE_JIT_DEEPGEMM=1
  SERVER_CMD+=(
    --attention-backend trtllm_mla
    --moe-runner-backend flashinfer_trtllm
    --enable-flashinfer-allreduce-fusion
  )
fi

echo ">>> launching SGLang (cache_mode=${CACHE_MODE}, hicache-size=${HICACHE_SIZE} GB)"
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

# ============================== Optional: GSM8K precheck ==============================
# Runs SGLang's built-in GSM8K accuracy bench against the SAME server,
# logs the accuracy into bench_meta.json so each cascade run carries a
# correctness signature. Both warmup and GSM8K are subsequently flushed
# (radix tree + OS page cache) before the cascade bench, so cascade
# still starts from a cold cache.
if [ "$GSM8K_PRECHECK" = "true" ]; then
  GSM8K_SCRIPT=""
  for c in /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
           "$HOME/work-space/sglang/benchmark/gsm8k/bench_sglang.py"; do
    [ -f "$c" ] && GSM8K_SCRIPT="$c" && break
  done
  if [ -z "$GSM8K_SCRIPT" ]; then
    echo ">>> WARNING: --gsm8k-precheck requested but bench_sglang.py not found; skipping" >&2
  else
    # File names match GLM.sh's accuracy_test() so the cascade tree and
    # the GLM benchmark tree produce comparable accuracy artifacts.
    GSM8K_LOG="$LOG_DIR/Accuracy_GSM8K.log"
    GSM8K_RESULT_JSONL="$LOG_DIR/Accuracy_GSM8K.jsonl"
    echo ">>> GSM8K precheck: ${GSM8K_NUM_QUESTIONS} questions, parallel=${GSM8K_PARALLEL}"
    # cd into LOG_DIR so the bench script's tmp_output_*.txt and any other
    # cwd-relative outputs land beside the rest of this run's artifacts.
    if (
        cd "$LOG_DIR"
        # NOTE: bench_sglang.py's --backend is a *frontend* selector
        # (srt / srt-no-parallel / srt-raw / gpt-*) defined in
        # python/sglang/test/test_utils.py:select_sglang_backend(). It is NOT
        # the server-side --backend used by bench_serving.py (sglang/vllm/tgi).
        # Don't pass --backend here — the default "srt" is what we want and
        # what GLM.sh's accuracy_test() uses.
        python3 "$GSM8K_SCRIPT" \
          --host "$HOST" --port "$PORT" \
          --num-questions "$GSM8K_NUM_QUESTIONS" \
          --parallel "$GSM8K_PARALLEL" \
          --result-file "$GSM8K_RESULT_JSONL" \
          2>&1 | tee "$GSM8K_LOG"
    ); then
      # bench_sglang.py prints "Accuracy: 0.930" on stdout. Grep is robust
      # enough; falls back to None if the line isn't there for any reason.
      GSM8K_ACC=$(grep -oP '^Accuracy:\s+\K[0-9.]+' "$GSM8K_LOG" | tail -1 || true)
      if [ -n "$GSM8K_ACC" ]; then
        echo ">>> GSM8K accuracy: ${GSM8K_ACC}"
        update_bench_meta_accuracy "$GSM8K_ACC"
      else
        echo ">>> WARNING: GSM8K finished but no Accuracy line parsed; leaving accuracy=null in bench_meta.json" >&2
        update_bench_meta_accuracy "None"
      fi
    else
      echo ">>> WARNING: GSM8K precheck failed (rc=$?); continuing with cascade bench, accuracy=null in bench_meta.json" >&2
      update_bench_meta_accuracy "None"
    fi
  fi
fi

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

# Flush radix tree so each run starts cold (HiCache itself controls L2/L3
# tier eviction). This wipes any KV pages left over from warmup or the
# optional GSM8K precheck.
curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
sleep 2

# Best-effort drop of OS page cache so L3 NVMe reads are truly cold on
# round 1. Inside SGLang docker images we are usually root and can write
# /proc/sys/vm/drop_caches directly; on stripped-down hosts we fall back
# to passwordless sudo, then to a warning if neither works (cascade
# correctness is unaffected — only L3 cold-read latency may be slightly
# inflated by warmup/GSM8K page cache hits).
sync
if echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
  echo ">>> dropped OS page cache"
elif command -v sudo >/dev/null 2>&1 && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
  echo ">>> dropped OS page cache (via sudo)"
else
  echo ">>> WARNING: could not drop OS page cache (need root or passwordless sudo);" \
       "L3 cold reads in round 1 may be slightly inflated by leftover page cache" >&2
fi
sleep 2

# Start the per-tier cache hit-rate monitor as a sidecar. It detects
# bench_multiturn.py's round-barrier crossings via sglang:num_requests_total
# and emits one line + one CSV row per finished round, with L1 / L2 / L3
# token-hit rates attributable to THAT round only (delta vs the previous
# round's snapshot — first round's delta uses the baseline captured at
# monitor start, so warmup / GSM8K precheck hits never pollute the cascade
# rates). No waiting for bench_multiturn.jsonl at the very end.
CACHE_MONITOR_SCRIPT="$(dirname "$(readlink -f "$0")")/cache_monitor.py"
if [ -f "$CACHE_MONITOR_SCRIPT" ]; then
  echo ">>> starting cache_monitor sidecar (time-driven 10s sample;"
  echo "    one CSV row per sample fuses /metrics cache hits + /proc/meminfo"
  echo "    host RAM, with round_index column to group rows by round)"
  python3 "$CACHE_MONITOR_SCRIPT" \
      --url "http://${HOST}:${PORT}/metrics" \
      --interval 10 \
      --num-clients "$NUM_CLIENTS" \
      --num-rounds "$NUM_ROUNDS" \
      --csv "$LOG_DIR/cache_tiers.csv" \
      > "$LOG_DIR/cache_monitor.log" 2>&1 &
  CACHE_MONITOR_PID=$!
  echo ">>> cache_monitor pid=${CACHE_MONITOR_PID}; tail -f ${LOG_DIR}/cache_monitor.log"
else
  echo ">>> WARNING: ${CACHE_MONITOR_SCRIPT} not found; skipping cache+host monitor" >&2
  CACHE_MONITOR_PID=""
fi

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

if [ -n "${CACHE_MONITOR_PID:-}" ] && kill -0 "$CACHE_MONITOR_PID" 2>/dev/null; then
  echo ">>> stopping cache_monitor (pid=${CACHE_MONITOR_PID})"
  kill "$CACHE_MONITOR_PID" 2>/dev/null || true
  wait "$CACHE_MONITOR_PID" 2>/dev/null || true
fi

# ============================== Cleanup ==============================
echo ">>> stopping server"
pkill -9 -f sglang.launch_server || true
sleep 5

if [ -n "$HICACHE_FILE_STORE_DIR" ] && [ -d "$HICACHE_FILE_STORE_DIR" ]; then
  sz=$(du -sh "$HICACHE_FILE_STORE_DIR" 2>/dev/null | cut -f1)
  rm -rf "$HICACHE_FILE_STORE_DIR"
  echo ">>> cleaned L3 file store (${sz:-?} reclaimed)"
fi

# Auto-suggest the right --MI355X / --B200 flag for plot_cascade.py
# based on the user's --tag (so the printed command can be copy-pasted
# straight into a shell). Falls back to --MI355X for unknown tags.
case "$TAG" in
  *MI355X*|*MI300*|*MI325*|*MI250*|*MI210*|*ROCm*|*rocm*|*amd*) PLOT_FLAG="--MI355X" ;;
  *B200*|*H200*|*H100*|*A100*|*L40*|*nvidia*|*NVIDIA*)          PLOT_FLAG="--B200"  ;;
  *)                                                            PLOT_FLAG="--MI355X" ;;  # generic fallback
esac

# Title text for the suggested plot — L1 has no hicache size, L2/L3 do.
if [ "$CACHE_MODE" = "L1" ]; then
  PLOT_TITLE_SUFFIX=""
else
  PLOT_TITLE_SUFFIX=" (hicache=${HICACHE_SIZE} GB)"
fi

echo ">>> done. results in: $LOG_DIR"
echo "    plot (single platform, paths resolved to absolute):"
echo "      python3 plot_cascade.py \\"
echo "          --Title \"${MODEL_NAME} cascade ${CACHE_MODE} ${TAG}${PLOT_TITLE_SUFFIX}\" \\"
echo "          ${PLOT_FLAG} ${LOG_DIR}/bench_multiturn.jsonl \\"
echo "          --out ${LOG_DIR}/cascade_${TAG}_${CACHE_MODE}.png"
echo "    plot (cross-platform, after running on the OTHER box too):"
echo "      python3 plot_cascade.py \\"
echo "          --Title \"${MODEL_NAME} cascade ${CACHE_MODE}: MI355X vs B200\" \\"
echo "          --MI355X <path-to-MI355X-${CACHE_MODE}-bench_multiturn.jsonl> \\"
echo "          --B200   <path-to-B200-${CACHE_MODE}-bench_multiturn.jsonl> \\"
echo "          --out    \$HOME/SGLang-benchmarks/results/cascade_${MODEL_NAME}_${CACHE_MODE}.png"
