#!/usr/bin/env bash
# HiCache.sh — benchmark SGLang HiCache (hierarchical KV cache) on B200 / MI355X.
#
# Models supported (preset):
#   deepseek-r1-0528   primary, native FP8 block-scale weights, MLA attention,
#                      reasoning-parser deepseek-r1
#   gpt-oss-120b       secondary, native MxFP4 weights (auto-detected from
#                      config.json — do NOT pass --quantization), GQA attention,
#                      reasoning-parser gpt-oss
#
# Cache hierarchy (HiCache extends RadixCache: HiRadixCache(RadixCache) — the
# two flags are mutually exclusive, so the modes below pick exactly one path):
#   L1 = GPU HBM         (RadixAttention prefix tree)
#   L2 = host DRAM       (--hicache-size N GB per rank, default "auto" picks
#                         a value that fits in detected MemAvailable; see
#                         auto_hicache_size() — overridable via --hicache-size)
#   L3 = external store  (file / hf3fs / mooncake / nixl)
#
# Cache modes (recommended sweep: no_radix → radix → hicache → hicache_file):
#   no_radix       --disable-radix-cache      no prefix sharing at all (lower
#                  bound; do NOT also pass --enable-hierarchical-cache)
#   radix          (default SGLang)           L1 only (GPU radix cache)
#   hicache        --enable-hierarchical-cache  L1 + L2 (no external store)
#   hicache_file   above + --hicache-storage-backend file   (L3 = local files,
#                  cheap and repeatable for sanity-checking the L3 code path)
#   hicache_hf3fs  above + --hicache-storage-backend hf3fs  (L3 = DeepSeek 3FS)
#   hicache_mooncake above + --hicache-storage-backend mooncake (L3 = Mooncake)
#
# Benchmarks (from sgl-project/sglang/benchmark/hicache/):
#   multiturn      synthetic multi-turn conversations (recommended; primary
#                  HiCache benchmark per LMSys 2025-09-10 blog)
#                  --disable-random-sample makes the prompts deterministic
#                  (random-token, no real dataset);
#                  --enable-round-barrier waits for ALL clients to finish
#                  round N before releasing round N+1, which gives prefetch
#                  time to populate L2/L3 → cache hit numbers actually reflect
#                  HiCache capability rather than scheduling jitter.
#   longcontext    long-context shared-prefix QA. Auto-downloads the
#                  preprocessed loogle_wiki_qa.json dataset from
#                  huggingface.co/datasets/xiezhq/loogle-wiki-qa to
#                  $HOME/SGLang-benchmarks/datasets/ on first run.
#   random_long    sglang.bench_serving with very long random inputs (mimics
#                  the legacy long-context single-prompt workflow; mostly
#                  exercises prefill — does NOT measure cache reuse since
#                  every prompt is fresh random tokens)
#
# Usage:
#   ./HiCache.sh                                     # DS-R1-0528, hicache, multiturn
#   ./HiCache.sh --bench longcontext                 # needs loogle dataset
#   ./HiCache.sh --cache-mode hicache_file
#   ./HiCache.sh --sweep                             # run all 4 cache modes back-to-back
#   ./HiCache.sh --sweep --hicache-size-sweep "64 128 256"   # cross-platform fairness
#   ./HiCache.sh --model-preset gpt-oss-120b
#   ./HiCache.sh --model /data/path/to/MyModel       # override preset
#   ./HiCache.sh --tp 4 --tag 0508_TP4
#   ./HiCache.sh --docker rocm/sgl-dev:v0.5.10rc0-rocm720-mi35x-20260412
#   ./HiCache.sh --docker lmsysorg/sglang:v0.5.9-cu130
#
# Refs:
#   https://docs.sglang.io/advanced_features/hicache_best_practices.html
#   https://docs.sglang.io/advanced_features/hicache_design.html
#   https://github.com/sgl-project/sglang/tree/main/benchmark/hicache
#   https://lmsys.org/blog/2025-09-10-sglang-hicache/

set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' || true

# ============================== Defaults ==============================
MODEL_PRESET="deepseek-r1-0528"
MODEL_PATH=""
CACHE_MODE="hicache"
BENCH_MODE="multiturn"
SWEEP_MODES="false"
TP_SIZE=8
PORT="30000"
HOST="localhost"
USER_TAG=""
DOCKER="untagged-docker"
DATASET_PATH=""

# NUMA interleave: prepend `numactl --interleave=<all online NUMA nodes>`
# to the SGLang server launch so the HiCache L2 pinned-memory pool is
# spread across all NUMA nodes instead of landing on whichever node
# happened to fault first. Disable with --no-numa-interleave.
USE_NUMA_INTERLEAVE="true"

# --hicache-size is per-TP-rank (in GB). Default "auto" picks a value at
# startup from detected MemAvailable so the same script runs on boxes
# with different DRAM (e.g. 3 TiB MI355X vs 2 TB B200) without tuning.
# The formula reserves 400 GB of host headroom — NOT for GPU activations
# (those live on GPU and are bounded by --mem-fraction-static), but for
# CPU-side things that compete with HiCache L2 for DRAM:
#   - Linux page cache (dominates while loading hundreds of GB of weights
#     from /data; getting squeezed slows model load to a crawl)
#   - HiCache's own pinned-memory staging buffers for L1↔L2 / L2↔L3 DMA
#   - NUMA-fragmentation slack across 8 ranks pinning DRAM concurrently
#   - SGLang scheduler / tokenizer / HTTP server / per-batch host tensors
# Per-rank pool is rounded down to a 32 GB multiple and clamped to
# [32, 512] GB (cache-hit returns flatten well before 512 GB per rank).
HICACHE_SIZE="auto"

# Cross-platform fairness sweep: when comparing MI355X vs B200, fix
# --hicache-size to the same set of values on both boxes so HiCache L2
# capacity is identical between platforms (DRAM is an OEM choice, not a
# GPU spec). Default sweeps 64 / 128 / 256 GB per rank; for each size
# the script checks that size × TP_SIZE fits in MemAvailable minus the
# 400 GB host headroom and skips (with a log entry) if it doesn't —
# so the same line runs on a 3 TiB MI355X box and a 2 TB B200 box.
# Pass an empty string ("") or a single-value list to disable sweep.
HICACHE_SIZE_SWEEP="64 128 256"

# Multi-turn defaults (from LMSys blog reference run)
NUM_CLIENTS=80
NUM_ROUNDS=10
REQUEST_LENGTH=2048
OUTPUT_LENGTH=1
MAX_PARALLEL=4
REQUEST_RATE=16
DISABLE_AUTO_RUN="true"
ENABLE_ROUND_BARRIER="true"

# random_long defaults (mimics legacy workflow)
RAND_ISL_LIST="32768 65536"
RAND_OSL=2048
RAND_NUM_PROMPTS=4
RAND_CONCURRENCY=1
RAND_WARMUP=2

while [[ $# -gt 0 ]]; do
  case $1 in
    --model-preset)   MODEL_PRESET="$2"; shift 2;;
    --model)          MODEL_PATH="$2"; shift 2;;
    --cache-mode)     CACHE_MODE="$2"; shift 2;;
    --bench)          BENCH_MODE="$2"; shift 2;;
    --sweep)          SWEEP_MODES="true"; shift 1;;
    --tp)             TP_SIZE="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --host)           HOST="$2"; shift 2;;
    --tag)            USER_TAG="-$2"; shift 2;;
    --docker)         DOCKER="$2"; shift 2;;
    --dataset-path)   DATASET_PATH="$2"; shift 2;;
    --hicache-size)   HICACHE_SIZE="$2"; shift 2;;
    --hicache-size-sweep) HICACHE_SIZE_SWEEP="$2"; shift 2;;
    --no-numa-interleave) USE_NUMA_INTERLEAVE="false"; shift 1;;
    --num-clients)    NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)     NUM_ROUNDS="$2"; shift 2;;
    --request-length) REQUEST_LENGTH="$2"; shift 2;;
    --output-length)  OUTPUT_LENGTH="$2"; shift 2;;
    --max-parallel)   MAX_PARALLEL="$2"; shift 2;;
    --request-rate)   REQUEST_RATE="$2"; shift 2;;
    --rand-isl)       RAND_ISL_LIST="$2"; shift 2;;
    --rand-osl)       RAND_OSL="$2"; shift 2;;
    --rand-num-prompts) RAND_NUM_PROMPTS="$2"; shift 2;;
    --rand-concurrency) RAND_CONCURRENCY="$2"; shift 2;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \?//'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# ============================== Model preset ==============================
# Defaults assume models are pre-downloaded under /data/huggingface/hub/.
# Pass --model PATH to override (HF identifier or any local path is accepted).
case "$MODEL_PRESET" in
  deepseek-r1-0528)
    [ -z "$MODEL_PATH" ] && MODEL_PATH="/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528"
    REASONING_PARSER="deepseek-r1"
    TOOL_CALL_PARSER=""    # set to deepseekv3 if you need tool calling
    MODEL_FAMILY="deepseek"
    ;;
  gpt-oss-120b)
    [ -z "$MODEL_PATH" ] && MODEL_PATH="/data/huggingface/hub/openai/gpt-oss-120b"
    REASONING_PARSER="gpt-oss"
    TOOL_CALL_PARSER=""
    MODEL_FAMILY="gpt-oss"
    ;;
  custom)
    [ -z "$MODEL_PATH" ] && { echo "--model required when --model-preset custom" >&2; exit 1; }
    REASONING_PARSER=""
    TOOL_CALL_PARSER=""
    MODEL_FAMILY="custom"
    ;;
  *)
    echo "Unknown --model-preset: $MODEL_PRESET" >&2
    exit 1
    ;;
esac
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# Sanity-check the model path early. Local paths must exist; HF identifiers
# (no leading "/") are passed through untouched and resolved by SGLang itself.
if [[ "$MODEL_PATH" == /* ]] && [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model path does not exist: $MODEL_PATH" >&2
  echo "       Pass --model PATH (local dir or HF identifier) to override." >&2
  exit 1
fi

# ============================== Resolve --hicache-size ==============================
auto_hicache_size() {
  # Pick a per-rank hicache-size (GB) that fits in detected MemAvailable
  # with 400 GB host-side headroom (page cache during model load, pinned
  # memory for transfers, NUMA fragmentation, scheduler/tokenizer/HTTP).
  # Note: GPU-side activations are bounded by --mem-fraction-static and
  # are NOT what this 400 GB is for. Rounds down to nearest 32 GB; clamps
  # to [32, 512] GB so the pool stays in a sensible range on big-RAM boxes.
  local mem_avail_kb mem_avail_gb headroom_gb usable_gb per_rank
  mem_avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  mem_avail_gb=$(( mem_avail_kb / 1024 / 1024 ))
  headroom_gb=400
  usable_gb=$(( mem_avail_gb - headroom_gb ))
  if [ "$usable_gb" -le 0 ]; then
    echo "ERROR: only ${mem_avail_gb} GB MemAvailable, cannot reserve" >&2
    echo "       ${headroom_gb} GB headroom for HiCache. Pass --hicache-size N" >&2
    echo "       manually (per-rank, in GB) or free up DRAM." >&2
    return 1
  fi
  per_rank=$(( usable_gb / TP_SIZE ))
  per_rank=$(( per_rank / 32 * 32 ))
  [ "$per_rank" -lt 32  ] && per_rank=32
  [ "$per_rank" -gt 512 ] && per_rank=512
  echo "$per_rank"
}

# Cache MemAvailable once at startup so the per-size DRAM check in the
# sweep loop is consistent and cheap (no repeated /proc/meminfo reads).
MEM_AVAIL_GB=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
HOST_HEADROOM_GB=400
USABLE_HOST_GB=$(( MEM_AVAIL_GB - HOST_HEADROOM_GB ))

if [ -n "$HICACHE_SIZE_SWEEP" ]; then
  echo ">>> --hicache-size-sweep: '${HICACHE_SIZE_SWEEP}' GB per rank" \
       "(each value runs once per hicache_* mode; sizes that don't fit" \
       "in MemAvailable=${MEM_AVAIL_GB} GB minus ${HOST_HEADROOM_GB} GB" \
       "headroom across TP=${TP_SIZE} ranks will be skipped)"
elif [ "$HICACHE_SIZE" = "auto" ]; then
  HICACHE_SIZE=$(auto_hicache_size) || exit 1
  echo ">>> auto --hicache-size: ${HICACHE_SIZE} GB per rank" \
       "(TP=${TP_SIZE}, MemAvailable=${MEM_AVAIL_GB} GB," \
       "total host pool ≈ $(( HICACHE_SIZE * TP_SIZE )) GB)"
else
  echo ">>> manual --hicache-size: ${HICACHE_SIZE} GB per rank" \
       "(total host pool ≈ $(( HICACHE_SIZE * TP_SIZE )) GB)"
fi

# ============================== Output dirs ==============================
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-HiCache${USER_TAG}"
mkdir -p "$BASE_LOG_DIR"

# ============================== Helpers ==============================
log_command() {
  local logfile=$1
  shift
  echo ">>> Executing command:" | tee -a "$logfile"
  echo "$*"                       | tee -a "$logfile"
  echo "---"                      | tee -a "$logfile"
  "$@" 2>&1 | tee -a "$logfile"
}

is_rocm_gpu_env() {
  [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1
}

wait_for_server() {
  local logfile=$1
  echo ">>> Waiting for server to be ready (checking: '${logfile}')..." | tee -a "$logfile"
  until [ "$(curl -s -o /dev/null -w "%{http_code}" "http://${HOST}:${PORT}/health")" -eq 200 ]; do
    sleep 5
  done
}

flush_cache() {
  curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
  sleep 2
}

stop_server() {
  pkill -9 -f "sglang.launch_server" || true
  pkill -9 python || true
  sleep 10
}

# ============================== Host snapshot ==============================
# Records lscpu / NUMA / GPU-to-NUMA / RAM into the per-run LOG_DIR so we
# can always reconstruct what hardware a benchmark was taken on.
snapshot_host_info() {
  local logfile=$1
  {
    echo "=== HiCache.sh host snapshot @ $(date '+%F %T %Z') ==="
    echo
    echo "--- lscpu ---"
    lscpu | grep -E "Architecture|Vendor|Model name|CPU\(s\)|Socket|Core|Thread|NUMA"
    echo
    echo "--- /proc/meminfo (selected) ---"
    grep -E "^MemTotal:|^MemAvailable:|^MemFree:|^Cached:|^Buffers:|^Dirty:|^HugePages_Total" /proc/meminfo
    echo
    echo "--- NUMA per-node DRAM + CPU list ---"
    for n in /sys/devices/system/node/node[0-9]*; do
      [ -d "$n" ] || continue
      local nid mem cpus
      nid=$(basename "$n" | sed 's/node//')
      mem=$(awk '/MemTotal/{print int($4/1024/1024)" GB"}' "$n/meminfo")
      cpus=$(cat "$n/cpulist" 2>/dev/null)
      printf "  node %s: DRAM=%s, CPUs=%s\n" "$nid" "$mem" "$cpus"
    done
    echo
    echo "--- GPU → NUMA node ---"
    for d in /sys/class/drm/card[0-9]*/device/numa_node; do
      [ -f "$d" ] || continue
      local card n
      card=$(echo "$d" | sed 's|/device/numa_node||;s|.*/||')
      n=$(cat "$d")
      printf "  %-8s NUMA %s\n" "$card" "$n"
    done | sort -u
    echo
    echo "--- numactl ---"
    if command -v numactl >/dev/null 2>&1; then
      numactl -H 2>&1 | head -20
    else
      echo "  numactl: NOT INSTALLED (apt-get install numactl)"
    fi
    echo "================================================="
  } | tee "$logfile"
}

# ============================== NUMA interleave detection ==============================
# List all online NUMA nodes as a comma-separated string ("0,1" or "0,1,2,3").
detect_numa_nodes() {
  ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null \
    | sed 's|.*/node||' | sort -n | paste -sd,
}

NUMA_NODES=$(detect_numa_nodes)
NUMACTL_PREFIX=()
if [ "$USE_NUMA_INTERLEAVE" = "true" ]; then
  if ! command -v numactl >/dev/null 2>&1; then
    echo ">>> WARNING: numactl not installed; skipping NUMA interleave."
    echo "    On a multi-NUMA box this can hurt HiCache L2 throughput."
    echo "    Install with: apt-get install numactl  (or yum install numactl)"
  elif [ -z "$NUMA_NODES" ] || [[ "$NUMA_NODES" != *,* ]]; then
    echo ">>> Single NUMA node detected (${NUMA_NODES:-none}); no interleave needed."
  else
    NUMACTL_PREFIX=(numactl --interleave="$NUMA_NODES")
    echo ">>> NUMA interleave: --interleave=${NUMA_NODES} (${NUMA_NODES//,/ + } nodes)"
  fi
fi

# ============================== LooGLE auto-download ==============================
# bench_long_context.py expects a flat JSON with keys {"queries", "contexts"}.
# The original LooGLE distribution (bigai-nlco/LooGLE) ships .jsonl per task,
# which does NOT match. The SGLang team's preprocessed copy lives at
# huggingface.co/datasets/xiezhq/loogle-wiki-qa as a single file.
ensure_loogle_dataset() {
  local dataset_dir="$HOME/SGLang-benchmarks/datasets"
  local target="$dataset_dir/loogle_wiki_qa.json"
  if [ -f "$target" ]; then
    DATASET_PATH="$target"
    return 0
  fi
  mkdir -p "$dataset_dir"
  echo ">>> loogle_wiki_qa.json not found, downloading to $dataset_dir ..."

  local url="https://huggingface.co/datasets/xiezhq/loogle-wiki-qa/resolve/main/loogle_wiki_qa.json"
  if command -v wget >/dev/null 2>&1; then
    wget -q --show-progress -O "$target.part" "$url" \
      || { echo "ERROR: wget failed for $url" >&2; rm -f "$target.part"; return 1; }
  elif command -v curl >/dev/null 2>&1; then
    curl -fSL --progress-bar -o "$target.part" "$url" \
      || { echo "ERROR: curl failed for $url" >&2; rm -f "$target.part"; return 1; }
  else
    echo "ERROR: neither wget nor curl is available; cannot auto-download." >&2
    echo "       Either install one, or download manually and pass --dataset-path:" >&2
    echo "         $url" >&2
    return 1
  fi
  mv "$target.part" "$target"
  echo ">>> Downloaded $(du -h "$target" | cut -f1) to $target"
  DATASET_PATH="$target"
}

# ============================== Server launch ==============================
build_server_cmd() {
  # Args: $1=cache_mode  $2=logfile  ; sets global SERVER_CMD array
  local mode=$1 logfile=$2
  local cmd=(
    "${NUMACTL_PREFIX[@]}"
    python3 -m sglang.launch_server
      --model-path "$MODEL_PATH"
      --tp "$TP_SIZE"
      --host "$HOST"
      --port "$PORT"
      --mem-fraction-static 0.85
      --watchdog-timeout 1200
      --enable-metrics
      --enable-cache-report
      --trust-remote-code
  )

  [ -n "$REASONING_PARSER" ] && cmd+=(--reasoning-parser "$REASONING_PARSER")
  [ -n "$TOOL_CALL_PARSER" ] && cmd+=(--tool-call-parser "$TOOL_CALL_PARSER")

  # ---------- Model-family-specific tuning ----------
  case "$MODEL_FAMILY" in
    deepseek)
      # DSR1-0528: native FP8 block-scale weights, auto-detected from
      # config.json's quantization_config (quant_method=fp8, weight_block_
      # size=[128,128]). MLA attention, 64-page for HiCache I/O efficiency,
      # FP8 KV cache to halve KV memory.
      cmd+=(
        --kv-cache-dtype fp8_e4m3
        --page-size 64
        --context-length 65536
        --chunked-prefill-size 32768
        --max-prefill-tokens 32768
      )
      ;;
    gpt-oss)
      # GPT-OSS 120B: native MxFP4 weights, auto-detected from config.json
      # (quant_method=mxfp4). DO NOT pass --quantization here — it would
      # conflict with the MxFP4 path. GQA attention, 131k native context.
      cmd+=(
        --page-size 64
        --context-length 65536
        --chunked-prefill-size 32768
        --max-prefill-tokens 32768
      )
      ;;
  esac

  # ---------- Platform-specific backend ----------
  if is_rocm_gpu_env; then
    # MI355X (ROCm) — let SGLang pick the default attention backend
    # (aiter for DSR1 MLA, triton for GPT-OSS GQA). Quick-reduce keeps
    # all-reduce cheap.
    export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
    export SAFETENSORS_FAST_GPU=1
    case "$MODEL_FAMILY" in
      deepseek) cmd+=(--attention-backend aiter);;
      gpt-oss)  cmd+=(--attention-backend aiter);;
    esac
  else
    # B200 (CUDA / Blackwell)
    case "$MODEL_FAMILY" in
      deepseek)
        cmd+=(
          --attention-backend trtllm_mla
          --moe-runner-backend flashinfer_trtllm
          --enable-flashinfer-allreduce-fusion
        )
        ;;
      gpt-oss)
        # GPT-OSS 120B on B200: native MxFP4 weights are picked up
        # automatically. flashinfer attention works well; we leave the
        # MoE runner at default for MxFP4 compatibility.
        cmd+=(
          --attention-backend flashinfer
        )
        ;;
    esac
    # JIT DeepGEMM is recommended for DSR1 / GLM-5 family on Blackwell.
    [ "$MODEL_FAMILY" = "deepseek" ] && export SGL_ENABLE_JIT_DEEPGEMM=1
  fi

  # ---------- Cache-mode-specific args ----------
  case "$mode" in
    no_radix)
      cmd+=(--disable-radix-cache)
      ;;
    radix)
      :
      ;;
    hicache)
      cmd+=(
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE"
        --hicache-mem-layout page_first_direct
        --hicache-io-backend kernel
        --hicache-write-policy write_through
      )
      ;;
    hicache_file)
      local store_dir="${LOG_DIR}/hicache_file_store"
      mkdir -p "$store_dir"
      cmd+=(
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE"
        --hicache-mem-layout page_first_direct
        --hicache-io-backend kernel
        --hicache-write-policy write_through
        --hicache-storage-backend file
        --hicache-storage-prefetch-policy best_effort
      )
      export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$store_dir"
      ;;
    hicache_hf3fs)
      # NOTE: requires 3FS cluster reachable from this host. The
      # `direct` I/O backend with `page_first_direct` is the recommended
      # combo (see best-practices doc).
      cmd+=(
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE"
        --hicache-mem-layout page_first_direct
        --hicache-io-backend direct
        --hicache-write-policy write_through
        --hicache-storage-backend hf3fs
        --hicache-storage-prefetch-policy wait_complete
      )
      ;;
    hicache_mooncake)
      # NOTE: requires Mooncake metadata server + RDMA NICs configured;
      # pre-set MOONCAKE_* env vars before running.
      cmd+=(
        --enable-hierarchical-cache
        --hicache-size "$HICACHE_SIZE"
        --hicache-mem-layout page_first_direct
        --hicache-io-backend direct
        --hicache-write-policy write_through
        --hicache-storage-backend mooncake
        --hicache-storage-prefetch-policy timeout
      )
      ;;
    *)
      echo "Unknown cache mode: $mode" >&2
      exit 1
      ;;
  esac

  echo ">>> Starting SGLang server (cache_mode=$mode)" | tee "$logfile"
  echo "${cmd[*]}" | tee -a "$logfile"
  "${cmd[@]}" 2>&1 | tee -a "$logfile" &
  SERVER_PID=$!
  wait_for_server "$logfile"
}

# ============================== Warmup ==============================
warmup() {
  local warmup_log="${LOG_DIR}/warmup.log"
  local cmd=(
    python3 -m sglang.bench_serving
      --backend sglang
      --host "$HOST" --port "$PORT"
      --model "$MODEL_PATH"
      --dataset-name random
      --random-input 1024 --random-output 128
      --random-range-ratio 1.0
      --max-concurrency 4 --num-prompt 8
      --output-file /dev/null
  )
  log_command "$warmup_log" "${cmd[@]}"
}

# ============================== Bench: multiturn ==============================
bench_multiturn() {
  local logfile="${LOG_DIR}/bench_multiturn.log"
  local jsonl="${LOG_DIR}/bench_multiturn.jsonl"
  flush_cache

  # Try repo path first (when running inside the SGLang docker image), fall
  # back to a working-dir checkout under ~/work-space.
  local script
  for candidate in \
      /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
      "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py" ; do
    [ -f "$candidate" ] && script="$candidate" && break
  done
  if [ -z "${script:-}" ]; then
    echo "ERROR: cannot find bench_multiturn.py — clone sgl-project/sglang into ~/work-space/sglang or run inside the SGLang docker image." >&2
    return 1
  fi

  local cmd=(
    python3 "$script"
      --host "$HOST" --port "$PORT"
      --model-path "$MODEL_PATH"
      --num-clients "$NUM_CLIENTS"
      --num-rounds "$NUM_ROUNDS"
      --request-length "$REQUEST_LENGTH"
      --output-length "$OUTPUT_LENGTH"
      --max-parallel "$MAX_PARALLEL"
      --request-rate "$REQUEST_RATE"
      --ready-queue-policy random
      --log-file "$jsonl"
      --tag "${MODEL_NAME}-${cache_mode}"
      --disable-random-sample
  )
  [ "$DISABLE_AUTO_RUN" = "true" ]    && cmd+=(--disable-auto-run)
  [ "$ENABLE_ROUND_BARRIER" = "true" ] && cmd+=(--enable-round-barrier)
  log_command "$logfile" "${cmd[@]}"
}

# ============================== Bench: longcontext ==============================
bench_longcontext() {
  local logfile="${LOG_DIR}/bench_longcontext.log"
  local jsonl="${LOG_DIR}/bench_longcontext.jsonl"
  flush_cache

  if [ -z "$DATASET_PATH" ]; then
    ensure_loogle_dataset || return 1
  fi
  if [ ! -f "$DATASET_PATH" ]; then
    echo "ERROR: longcontext bench needs a loogle dataset at $DATASET_PATH." >&2
    echo "       Auto-download failed; pass --dataset-path PATH manually." >&2
    return 1
  fi

  local script
  for candidate in \
      /sgl-workspace/sglang/benchmark/hicache/bench_long_context.py \
      "$HOME/work-space/sglang/benchmark/hicache/bench_long_context.py" ; do
    [ -f "$candidate" ] && script="$candidate" && break
  done
  if [ -z "${script:-}" ]; then
    echo "ERROR: cannot find bench_long_context.py" >&2
    return 1
  fi

  local cmd=(
    python3 "$script"
      --host "$HOST" --port "$PORT"
      --model-path "$MODEL_PATH"
      --dataset-path "$DATASET_PATH"
      --num-clients "$NUM_CLIENTS"
      --log-file "$jsonl"
      --tag "${MODEL_NAME}-${cache_mode}"
  )
  log_command "$logfile" "${cmd[@]}"
}

# ============================== Bench: random_long ==============================
bench_random_long() {
  for isl in $RAND_ISL_LIST; do
    local logfile="${LOG_DIR}/bench_random_in${isl}_out${RAND_OSL}_conc${RAND_CONCURRENCY}.log"
    flush_cache
    local cmd=(
      python3 -m sglang.bench_serving
        --backend sglang
        --host "$HOST" --port "$PORT"
        --model "$MODEL_PATH"
        --dataset-name random
        --random-input-len "$isl"
        --random-output-len "$RAND_OSL"
        --num-prompt "$RAND_NUM_PROMPTS"
        --max-concurrency "$RAND_CONCURRENCY"
        --warmup-requests "$RAND_WARMUP"
        --output-file /dev/null
    )
    log_command "$logfile" "${cmd[@]}"
  done
}

# ============================== Run-one ==============================
run_one() {
  # Caller is expected to have set LOG_DIR and (for hicache_* modes)
  # HICACHE_SIZE so build_server_cmd picks them up.
  local cache_mode=$1
  mkdir -p "$LOG_DIR"
  if [ "$BENCH_MODE" = "multiturn" ] || [ "$BENCH_MODE" = "longcontext" ]; then
    export SGLANG_TORCH_PROFILER_DIR="$LOG_DIR"
  fi

  snapshot_host_info "${LOG_DIR}/host_info.log"

  local server_log="${LOG_DIR}/server.log"
  build_server_cmd "$cache_mode" "$server_log"
  warmup
  case "$BENCH_MODE" in
    multiturn)   bench_multiturn ;;
    longcontext) bench_longcontext ;;
    random_long) bench_random_long ;;
    *) echo "Unknown bench mode: $BENCH_MODE" >&2; stop_server; exit 1 ;;
  esac
  stop_server
}

# ============================== Main ==============================
if [ "$SWEEP_MODES" = "true" ]; then
  CACHE_MODES=("no_radix" "radix" "hicache" "hicache_file")
else
  CACHE_MODES=("$CACHE_MODE")
fi

# When --hicache-size-sweep is set, hicache_* modes run once per size;
# non-hicache modes (no_radix, radix) ignore size and run only once.
if [ -n "$HICACHE_SIZE_SWEEP" ]; then
  SIZE_LIST=( $HICACHE_SIZE_SWEEP )
else
  SIZE_LIST=( "$HICACHE_SIZE" )
fi

SKIPPED_LOG="${BASE_LOG_DIR}/skipped.log"
: > "$SKIPPED_LOG"

for cm in "${CACHE_MODES[@]}"; do
  if [[ "$cm" == hicache* ]]; then
    sizes_to_run=( "${SIZE_LIST[@]}" )
  else
    sizes_to_run=( "_unused_" )
  fi
  for sz in "${sizes_to_run[@]}"; do
    if [ "$sz" != "_unused_" ]; then
      total_pool_gb=$(( sz * TP_SIZE ))
      if [ "$total_pool_gb" -gt "$USABLE_HOST_GB" ]; then
        msg="[skip] ${cm}/size_${sz}: needs ${total_pool_gb} GB host pool" \
            "(${sz} × TP=${TP_SIZE}), only ${USABLE_HOST_GB} GB usable" \
            "(MemAvailable=${MEM_AVAIL_GB} GB − ${HOST_HEADROOM_GB} GB headroom)"
        echo "$msg"
        echo "$msg" >> "$SKIPPED_LOG"
        continue
      fi
      HICACHE_SIZE="$sz"
      LOG_DIR="${BASE_LOG_DIR}/${cm}/size_${sz}"
      banner_size="  hicache_size=${sz} GB"
    else
      LOG_DIR="${BASE_LOG_DIR}/${cm}"
      banner_size=""
    fi
    echo "================================================================"
    echo ">>> [HiCache.sh] cache_mode=${cm}${banner_size}  bench=${BENCH_MODE}  model=${MODEL_NAME}"
    echo "================================================================"
    run_one "$cm"
  done
done

if [ -s "$SKIPPED_LOG" ]; then
  echo ">>> Skipped runs (DRAM insufficient) — see $SKIPPED_LOG"
  cat "$SKIPPED_LOG"
fi
echo ">>> All done. Results under: $BASE_LOG_DIR"
