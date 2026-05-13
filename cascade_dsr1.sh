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
# GSM8K precheck (ON by default; --no-gsm8k-precheck to opt out): after
# warmup and BEFORE the cascade, runs SGLang's built-in GSM8K accuracy
# bench against the same server (--gsm8k-num-questions / --gsm8k-parallel
# tunable). Accuracy is written into bench_meta.json so every cascade
# run carries a correctness signature — "fast" server config is
# meaningless if it's giving garbage answers. The radix tree is flushed
# afterwards so the cascade still starts cold.
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
HICACHE_SIZE=192            # per rank, GB
MEM_FRACTION_STATIC=0.85    # SGLang --mem-fraction-static. Lower it (e.g. 0.7)
                            # for smaller models like Qwen3-32B where weights
                            # are tiny; raise it (e.g. 0.9) only when KV pool
                            # pressure is the bottleneck. Affects both device
                            # KV-pool size and "host > device" HiCache check.
CACHE_MODE="L3_file"        # one of: none|L1|L2|L3_file
                            #   none    = --disable-radix-cache (no radix, no HiCache)
                            #   L1      = radix cache only (no HiCache, GPU only)
                            #   L2      = radix + HiCache device + host (no file backend)
                            #   L3_file = radix + HiCache device + host + file
                            # On MI355X --cache-mode=none also makes page_size>1
                            # correctness-safe (the _transform_table_1_to_real
                            # bug requires radix prefix sharing to misalign).
NO_CACHE="false"            # Legacy; --no-cache is now an alias for --cache-mode none
PAGE_SIZE_OVERRIDE=""       # If non-empty, overrides the model-family-detected
                            # PAGE_SIZE. Use --page-size to set. Useful for
                            # controlled DSR1 page_size=1-vs-64 experiments
                            # combined with --cache-mode none.
NUM_CLIENTS=300
NUM_ROUNDS=15
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
WAIT_FOR_SERVER_SEC=900     # 15 min cap; bail out if SGLang doesn't /health

# GSM8K precheck. ON by default so every cascade run carries a model-
# correctness signature in bench_meta.json (and a full per-question
# Accuracy_GSM8K.log under $LOG_DIR). Adds ~70-90 sec per cascade run;
# pass --no-gsm8k-precheck to opt out when you only care about cache /
# throughput numbers. 1200 questions × parallel=1200 mirrors GLM.sh
# accuracy_test() so MI355X and B200 results are directly comparable.
GSM8K_PRECHECK="true"
GSM8K_NUM_QUESTIONS=1200
GSM8K_PARALLEL=1200
GSM8K_ONLY="false"          # If true, exit cleanly after GSM8K precheck.
                            # Skips warmup is NOT skipped (needed to compile
                            # cuda graphs first), and skips the cascade
                            # multi-turn benchmark. Useful for quick
                            # accuracy verification (10-15 min vs 30+).

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL_PATH="$2"; shift 2;;
    --tag)            TAG="$2"; shift 2;;
    --docker)         DOCKER="$2"; shift 2;;
    --tp)             TP_SIZE="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --hicache-size)   HICACHE_SIZE="$2"; shift 2;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2;;
    --cache-mode)     CACHE_MODE="$2"; shift 2;;
    --no-cache)       NO_CACHE="true"; shift 1;;
    --page-size)      PAGE_SIZE_OVERRIDE="$2"; shift 2;;
    --num-clients)    NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)     NUM_ROUNDS="$2"; shift 2;;
    --request-length) REQUEST_LENGTH="$2"; shift 2;;
    --max-parallel)   MAX_PARALLEL="$2"; shift 2;;
    --request-rate)   REQUEST_RATE="$2"; shift 2;;
    --gsm8k-precheck)      GSM8K_PRECHECK="true"; shift 1;;
    --no-gsm8k-precheck)   GSM8K_PRECHECK="false"; shift 1;;
    --gsm8k-only)          GSM8K_PRECHECK="true"; GSM8K_ONLY="true"; shift 1;;
    --gsm8k-num-questions) GSM8K_NUM_QUESTIONS="$2"; shift 2;;
    --gsm8k-parallel)      GSM8K_PARALLEL="$2"; shift 2;;
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

# ============================== Model-family detection ==============================
# Server launch args (page_size, reasoning parser, context length) are model-
# family dependent. Detect from the model directory name so the same script
# can run DSR1 / DSV3 (MLA) and Qwen / Llama / Mistral (GQA) without flags.
#
# - MLA family (DSR1/DSV3/Kimi-K2): aiter MLA path is correctness-broken at
#   page_size > 1 (see _transform_table_1_to_real in aiter_backend.py).
#   Forced to page_size=1; also enables --reasoning-parser deepseek-r1.
# - GQA family (Qwen3, Llama-3.x, Mixtral, Mistral): no MLA bug, but we
#   still use page_size=1 because that's also what AMD's HiCache reference
#   command uses (no measured win at 64 yet on aiter HiCache path).
# - NSA / DSA family (DSV3.2, GLM-5 DSA): blocked separately by a Triton
#   power-of-2 bug; not safe on this docker, just print a warning.
#
# Fallback for unknown names: GQA defaults with a conservative context len.
case "$MODEL_NAME" in
  # NSA / DSA family — must come BEFORE the generic MLA `DeepSeek-V3*`
  # glob below, otherwise `DeepSeek-V3.2-Exp` matches the wrong arm.
  DeepSeek-V3.2*)
    MODEL_FAMILY="NSA"
    REASONING_PARSER_FLAGS=(--reasoning-parser deepseek-r1)
    PAGE_SIZE=64
    CONTEXT_LENGTH=65536
    echo ">>> NSA model (DeepSeek-V3.2 DSA); will route attention through tilelang" >&2
    ;;
  GLM-5*|glm-5*|GLM-5.1*|glm-5.1*)
    MODEL_FAMILY="NSA"
    # GLM-5/5.1 uses its own reasoning + tool-call parser (matches GLM.sh).
    # Without these, bench_sglang.py's GSM8K answer extraction picks up
    # intermediate numbers from the un-stripped <think>...</think> block
    # → accuracy collapses from ~0.94 to ~0.82.
    REASONING_PARSER_FLAGS=(
      --reasoning-parser glm45
      --tool-call-parser glm47
    )
    PAGE_SIZE=64
    CONTEXT_LENGTH=65536
    echo ">>> NSA model (GLM-5/5.1); reasoning-parser=glm45, tool-call-parser=glm47, attention via tilelang" >&2
    ;;
  # MLA family
  DeepSeek-R1*|DeepSeek-V3*|DeepSeek-V2*|Kimi-K2*|deepseek-*|DeepSeek-R1-0528*)
    MODEL_FAMILY="MLA"
    REASONING_PARSER_FLAGS=(--reasoning-parser deepseek-r1)
    PAGE_SIZE=1
    CONTEXT_LENGTH=65536
    ;;
  Qwen3-32B|Qwen3-*|Qwen2.5-*|Qwen2-*)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=1
    CONTEXT_LENGTH=40960
    ;;
  Llama-3.*|Llama3-*|Meta-Llama-3*)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=1
    CONTEXT_LENGTH=65536
    ;;
  Mixtral-*|Mistral-*|mistral-*)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=1
    CONTEXT_LENGTH=32768
    ;;
  *)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=1
    CONTEXT_LENGTH=32768
    echo ">>> NOTE: unrecognized model name '$MODEL_NAME', defaulting to GQA family" >&2
    echo "    page_size=$PAGE_SIZE, context_length=$CONTEXT_LENGTH; override via MODEL_FAMILY env if wrong" >&2
    ;;
esac

# Apply --page-size override if user passed one. Doing this AFTER the family
# detection so the override always wins, but before LOG_DIR / SERVER_CMD use
# $PAGE_SIZE so they see the final value.
if [ -n "$PAGE_SIZE_OVERRIDE" ]; then
  echo ">>> page_size override: model-family default $PAGE_SIZE → $PAGE_SIZE_OVERRIDE" >&2
  PAGE_SIZE="$PAGE_SIZE_OVERRIDE"
fi

# --no-cache (legacy) is an alias for --cache-mode none; let it win if both passed.
if [ "$NO_CACHE" = "true" ]; then
  CACHE_MODE="none"
fi

# Validate CACHE_MODE.
case "$CACHE_MODE" in
  none|L1|L2|L3_file) ;;
  *)
    echo "ERROR: --cache-mode must be one of: none, L1, L2, L3_file (got '$CACHE_MODE')" >&2
    exit 1
    ;;
esac

# NSA on ROCm + radix cache: tilelang NSA backend lacks the
# page_table_1_flattened + dequantize_k_cache_paged fixup that
# flashmla_sparse does (see nsa_backend.py:1432-1438 vs :1448-1455).
# Without the fixup, GSM8K drops from 0.94 → 0.82 because of bad KV
# reads on prefix-shared blocks.
#
# Workaround applied in SERVER_CMD: when CACHE_MODE != none we override
# --nsa-prefill-backend to flashmla_sparse (which has the fixup).
# Decode stays on tilelang. This is experimental — verify with GSM8K
# precheck. Leave correctness_at_risk=True until empirically confirmed.
CORRECTNESS_AT_RISK="false"
if { [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; } \
   && [ "$MODEL_FAMILY" = "NSA" ] && [ "$CACHE_MODE" != "none" ]; then
  CORRECTNESS_AT_RISK="true"
  echo ">>> WARNING: NSA family on ROCm + cache_mode=$CACHE_MODE is correctness-broken." >&2
  echo "    tilelang NSA prefill on ROCm lacks the page_table_1_flattened fixup" >&2
  echo "    that flashmla_sparse has on CUDA. With radix cache on, prefix-shared" >&2
  echo "    KV reads come from wrong physical slots → GSM8K 0.94 → ~0.82." >&2
  echo "    flashmla_sparse can't be used on ROCm (sgl_kernel.flashmla_ops is" >&2
  echo "    CUDA-only). For accurate runs use --cache-mode none. This run is" >&2
  echo "    throughput-only; bench_meta correctness_at_risk=true marks it." >&2
fi

echo ">>> model family: $MODEL_FAMILY (page_size=$PAGE_SIZE, context_length=$CONTEXT_LENGTH, cache_mode=$CACHE_MODE, correctness_at_risk=$CORRECTNESS_AT_RISK)"

# Auto-clamp NUM_ROUNDS so per-client cumulative input doesn't overflow
# context. cascade_dsr1.sh's multi-turn workload appends REQUEST_LENGTH
# tokens each round, so total ≈ REQUEST_LENGTH * NUM_ROUNDS at the last
# round. Leave 4K slack for output + sysprompt + safety margin.
_total_seq=$(( REQUEST_LENGTH * NUM_ROUNDS + 4096 ))
if [ "$_total_seq" -gt "$CONTEXT_LENGTH" ]; then
  _max_rounds=$(( (CONTEXT_LENGTH - 4096) / REQUEST_LENGTH ))
  echo ">>> WARNING: NUM_ROUNDS=$NUM_ROUNDS * REQUEST_LENGTH=$REQUEST_LENGTH + 4K slack" >&2
  echo "    = $_total_seq tokens exceeds CONTEXT_LENGTH=$CONTEXT_LENGTH for $MODEL_FAMILY" >&2
  echo "    clamping NUM_ROUNDS from $NUM_ROUNDS to $_max_rounds" >&2
  NUM_ROUNDS=$_max_rounds
fi

# ============================== Output dir ==============================
# none / L1 don't use HICACHE_SIZE so omit the size_X suffix; L2 / L3_file do.
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
case "$CACHE_MODE" in
  none|L1)
    LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-cascade-${TAG}/${CACHE_MODE}"
    ;;
  L2|L3_file)
    LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-cascade-${TAG}/${CACHE_MODE}/size_${HICACHE_SIZE}"
    ;;
esac
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
    "correctness_at_risk": $([ "$CORRECTNESS_AT_RISK" = "true" ] && echo "True" || echo "False"),
    "model_path": "$MODEL_PATH",
    "model_name": "$MODEL_NAME",
    "model_family": "$MODEL_FAMILY",
    "page_size": $PAGE_SIZE,
    "context_length": $CONTEXT_LENGTH,
    "tp_size": $TP_SIZE,
    "kv_cache_dtype": "fp8_e4m3",
    "mem_fraction_static": $MEM_FRACTION_STATIC,
    "host_headroom_gb": 200,
    "hbm_gb": $HBM_GB,
    "device_pool_gb": int($HBM_GB * $MEM_FRACTION_STATIC - 84),
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
    "gsm8k_precheck_enabled": $META_GSM8K_ENABLED,
    "gsm8k_precheck_num_questions": $META_GSM8K_NQ,
    "gsm8k_precheck_accuracy": None,
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
# Only L3_file mode actually uses the file backend. none / L1 / L2 skip it.
if [ "$CACHE_MODE" = "L3_file" ]; then
  HICACHE_FILE_STORE_DIR="/tmp/cascade_dsr1_l3_${TAG}_${HICACHE_SIZE}"
  rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null
  mkdir -p "$HICACHE_FILE_STORE_DIR"
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_FILE_STORE_DIR"
  # Graceful kill: SIGTERM lets SGLang flush stdout/stderr buffers and write
  # final "Shutting down" lines into server.log; only fall back to SIGKILL
  # if it hasn't exited after 3s. Avoids losing the trailing N hundred
  # Prefill-batch lines on every cleanup.
  trap 'pkill -TERM -f sglang.launch_server 2>/dev/null; sleep 3;
        pkill -9 -f sglang.launch_server 2>/dev/null;
        rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null || true' EXIT
else
  HICACHE_FILE_STORE_DIR=""
  trap 'pkill -TERM -f sglang.launch_server 2>/dev/null; sleep 3;
        pkill -9 -f sglang.launch_server 2>/dev/null || true' EXIT
fi

# ============================== Server launch ==============================
SERVER_LOG="$LOG_DIR/server.log"

# Force Python to flush stdout/stderr line-by-line so server.log captures
# every "Prefill batch / POST /generate" line in real time (default
# block-buffering loses up to ~8KB of trailing logs when the server is
# killed at end of run, and makes tail -f look frozen during the run).
export PYTHONUNBUFFERED=1

SERVER_CMD=(
  "${NUMACTL_PREFIX[@]}"
  python3 -u -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --tp "$TP_SIZE"
    --host "$HOST" --port "$PORT"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --watchdog-timeout 1200
    --enable-metrics
    --enable-cache-report
    --trust-remote-code
    "${REASONING_PARSER_FLAGS[@]}"
    --kv-cache-dtype fp8_e4m3
    --page-size "$PAGE_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --chunked-prefill-size 32768
    --max-prefill-tokens 32768
)
# --- Cache mode flags ---
case "$CACHE_MODE" in
  none)
    # Disable BOTH radix cache and HiCache. Without prefix sharing every
    # request re-prefills its full context, but per-attention-step kernels
    # run unimpeded — useful as an MI355X "what would throughput look like
    # if the page-size aiter bug didn't force page_size=1?" baseline.
    SERVER_CMD+=(--disable-radix-cache)
    ;;
  L1)
    # Radix cache only (prefix sharing on device GPU). No HiCache offload
    # to host or file. Smallest prefix-aware setup. Server defaults to
    # this when --enable-hierarchical-cache is omitted.
    :
    ;;
  L2)
    # Radix + HiCache device + host (no file backend). Host RAM acts as
    # the L2 spill tier; nothing persists past process exit.
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-mem-layout page_first_direct
      --hicache-io-backend kernel
      --hicache-write-policy write_through
    )
    ;;
  L3_file)
    # Radix + HiCache device + host + file (full hierarchy).
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

# --- NSA family: route attention through tilelang on ROCm ---
# GLM-5 / GLM-5.1 / DSV3.2 use NSA (DSA). On ROCm aiter's dense MHA
# fallback path inside _concat_and_cast_mha_k() trips a Triton
# `arange's range must be a power of 2` error when qk_nope_head_dim is
# not power-of-2 (GLM-5 ships qk_nope_head_dim=192). tilelang skips
# that fallback entirely — same workaround as GLM.sh.
#
# We tried flashmla_sparse (which has prefix-sharing fixup that tilelang
# lacks) but it requires the CUDA-only sgl_kernel.flashmla_ops extension
# and crashes on ROCm at first prefill. So tilelang is the only option
# on ROCm right now, and we live with the radix-prefix-sharing accuracy
# bug — see CORRECTNESS_AT_RISK warning above.
if [ "$MODEL_FAMILY" = "NSA" ]; then
  SERVER_CMD+=(
    --nsa-prefill-backend tilelang
    --nsa-decode-backend tilelang
  )
fi

# --- Attention backend ---
# aiter for ROCm except NSA (tilelang covers NSA path); trtllm_mla for B200.
if is_rocm; then
  # export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
  export SAFETENSORS_FAST_GPU=1
  if [ "$MODEL_FAMILY" != "NSA" ]; then
    SERVER_CMD+=(--attention-backend aiter)
  fi
else
  export SGL_ENABLE_JIT_DEEPGEMM=1
  SERVER_CMD+=(
    --attention-backend trtllm_mla
    --moe-runner-backend flashinfer_trtllm
    --enable-flashinfer-allreduce-fusion
  )
fi

# --- Friendly launch banner ---
case "$CACHE_MODE" in
  none) echo ">>> launching SGLang (cache_mode=none, --disable-radix-cache, page_size=${PAGE_SIZE})" ;;
  L1)   echo ">>> launching SGLang (cache_mode=L1, radix-cache only, page_size=${PAGE_SIZE})" ;;
  L2)   echo ">>> launching SGLang (cache_mode=L2, hicache-size=${HICACHE_SIZE} GB device+host, page_size=${PAGE_SIZE})" ;;
  L3_file) echo ">>> launching SGLang (cache_mode=L3_file, hicache-size=${HICACHE_SIZE} GB device+host+file, page_size=${PAGE_SIZE})" ;;
esac
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
# Runs SGLang's built-in GSM8K accuracy bench against the same server,
# logs the score into bench_meta.json (gsm8k_precheck_accuracy) so each
# cascade run carries a correctness signature. Speed numbers are
# meaningless if accuracy is broken — this is the gate that catches
# server-config drifts (e.g. ROCM_QUICK_REDUCE_QUANTIZATION=INT4 tanking
# DSR1 from 0.93 to 0.01) before we trust the cascade throughput.
#
# After GSM8K, the cascade still flushes radix tree so it starts cold.
# ${VAR:-} guard: if a previously-started older cascade_dsr1.sh process
# happens to re-read the file from this point on (file edited mid-run),
# unset GSM8K_PRECHECK won't trip set -u; the precheck just gets skipped.
if [ "${GSM8K_PRECHECK:-false}" = "true" ]; then
  GSM8K_SCRIPT=""
  for c in /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
           "$HOME/work-space/sglang/benchmark/gsm8k/bench_sglang.py"; do
    [ -f "$c" ] && GSM8K_SCRIPT="$c" && break
  done
  if [ -z "$GSM8K_SCRIPT" ]; then
    echo ">>> WARNING: --gsm8k-precheck requested but bench_sglang.py not found; skipping" >&2
  else
    GSM8K_LOG="$LOG_DIR/Accuracy_GSM8K.log"
    GSM8K_RESULT_JSONL="$LOG_DIR/Accuracy_GSM8K.jsonl"
    echo ">>> GSM8K precheck: ${GSM8K_NUM_QUESTIONS} questions, parallel=${GSM8K_PARALLEL}"
    # cd into LOG_DIR so the bench script's tmp_output_*.txt and any other
    # cwd-relative artifacts land alongside this run's other outputs.
    # NOTE: do NOT pass --backend here. bench_sglang.py's --backend is a
    # frontend selector (srt / srt-no-parallel / srt-raw / gpt-*) — see
    # python/sglang/test/test_utils.py:select_sglang_backend(). It is NOT
    # the bench_serving.py --backend (sglang/vllm/tgi). The default "srt"
    # is what GLM.sh's accuracy_test() uses and what we want.
    if (
        cd "$LOG_DIR"
        python3 "$GSM8K_SCRIPT" \
          --host "$HOST" --port "$PORT" \
          --num-questions "$GSM8K_NUM_QUESTIONS" \
          --parallel "$GSM8K_PARALLEL" \
          --result-file "$GSM8K_RESULT_JSONL" \
          2>&1 | tee "$GSM8K_LOG"
    ); then
      # bench_sglang.py prints "Accuracy: 0.930" on stdout. Grep is
      # robust enough; falls back to None if the line isn't there.
      GSM8K_ACC=$(grep -oP '^Accuracy:\s+\K[0-9.]+' "$GSM8K_LOG" | tail -1 || true)
      if [ -n "$GSM8K_ACC" ]; then
        echo ">>> GSM8K accuracy: ${GSM8K_ACC}  (log: $GSM8K_LOG)"
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

# Early-exit when --gsm8k-only was passed: skip the cascade benchmark and
# tear the server down. EXIT trap cleans up server + L3 file dir.
if [ "${GSM8K_ONLY:-false}" = "true" ]; then
  echo ">>> --gsm8k-only set; skipping cascade multiturn benchmark."
  echo ">>> done. results in: $LOG_DIR"
  if [ "${GSM8K_PRECHECK:-false}" = "true" ]; then
    echo "    accuracy log:    $LOG_DIR/Accuracy_GSM8K.log"
    echo "    bench_meta:      $LOG_DIR/bench_meta.json"
  fi
  exit 0
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
# GSM8K precheck so the cascade still measures cold-cache cascading.
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
# Graceful: SIGTERM first so SGLang gets to flush its final log lines
# (Shutting down / per-rank summary / etc.) into server.log; then SIGKILL
# anything still hanging.
echo ">>> stopping server (graceful: SIGTERM then SIGKILL after 5s)"
pkill -TERM -f sglang.launch_server 2>/dev/null || true
sleep 5
pkill -9 -f sglang.launch_server 2>/dev/null || true
sleep 1

if [ -d "$HICACHE_FILE_STORE_DIR" ]; then
  sz=$(du -sh "$HICACHE_FILE_STORE_DIR" 2>/dev/null | cut -f1)
  rm -rf "$HICACHE_FILE_STORE_DIR"
  echo ">>> cleaned L3 file store (${sz:-?} reclaimed)"
fi

echo ">>> done. results in: $LOG_DIR"
# Same ${VAR:-} guard as the precheck block — protects a running older
# cascade if it re-reads the file mid-execution.
if [ "${GSM8K_PRECHECK:-false}" = "true" ]; then
  echo "    accuracy log: $LOG_DIR/Accuracy_GSM8K.log"
  echo "    (gsm8k_precheck_accuracy field also lives in $LOG_DIR/bench_meta.json)"
fi
echo "    plot: python3 plot_cascade.py --tags $TAG --hicache-size $HICACHE_SIZE"
