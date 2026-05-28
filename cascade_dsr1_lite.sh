#!/usr/bin/env bash
# cascade_dsr1_lite.sh — DSR1-0528 cascade benchmark on ROCm aiter.
# Slim version of cascade_dsr1.sh. No NSA/GQA/MXFP4/EAGLE/CUDA dispatch.

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TAG=""
DOCKER="untagged-docker"
TP_SIZE=8
HOST="localhost"
PORT=30000
CACHE_MODE="L3_file"
CACHE_MODES=""
HICACHE_SIZE=auto
HOST_HEADROOM_GB=400
NUM_CLIENTS=300
NUM_ROUNDS=10
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
CUDA_GRAPH_MAX_BS=0
# Aligned to InferenceX dsr1_fp8_mi355x.sh (validated 2026-05-28):
#   chunked_prefill_size = max_prefill_tokens = 196608  (was 32768)
#   mem_fraction_static  = 0.8                          (was 0.85; sglang
#                                                        internal-scales to ~0.68)
# Larger prefill window matters most for HiCache cold-start rounds where
# many requests' 4K prefixes hit the server simultaneously.
CHUNKED_PREFILL_SIZE=196608
MAX_PREFILL_TOKENS=196608
MEM_FRACTION_STATIC=0.8
MEM_FRACTION_EXPLICIT=false   # flipped true when --mem-fraction-static is passed
L1_SIZE=""                    # GB/rank; when set, mem-fraction-static is auto-derived
                              # via compute_profile_params.py (mutually exclusive
                              # with --mem-fraction-static)
# PAGE_SIZE / CONTEXT_LENGTH default to empty: sglang auto-picks page_size=1
# (aiter MLA legacy path) and context_length=model native (163840 for DSR1).
# Earlier baseline (PAGE_SIZE=64, CONTEXT_LENGTH=65536) caused 2000ms TTFT in
# disable-radix-cache runs and reproducible GPU memory access faults at higher
# concurrency -- root-caused to PR #25556 fix #2 (cuda_graph_kv_indices buffer
# overrun, see HiCachePatch/pr25556-explained.md). To use page_size>1, pass
# --page-size N explicitly AND apply local-patches/HiCachePatch/apply-all.sh
# on the sglang checkout first.
PAGE_SIZE=""
PAGE_SIZE_EXPLICIT=false
CONTEXT_LENGTH=""
CONTEXT_LENGTH_EXPLICIT=false
HICACHE_WRITE_POLICY="write_through"
# Layout × io backend compatibility matrix (server_args.py:3108-3125
# silently rewrites incompatible pairs, so we pin the recommended one).
#
#   layout              kernel (rec.)   direct
#   ----------------    -------------   ------
#   layer_first         OK              OK
#   page_first          OK              -> forced to page_first_direct
#   page_first_direct   -> forced direct OK
#
# We want page-first organisation (better for L3 file backend, which
# reads / writes whole pages) AND kernel io backend (GPU-assisted KV
# transfer kernels, recommended in SGLang docs over plain cudaMemcpy).
# That intersection is `page_first + kernel`, NOT `page_first_direct
# + kernel` (which silently downgrades io to direct).
HICACHE_MEM_LAYOUT="page_first"
HICACHE_IO_BACKEND="kernel"
GSM8K_PRECHECK="true"
GSM8K_NUM_QUESTIONS=1200
GSM8K_PARALLEL=1200
WAIT_FOR_SERVER_SEC=1500
ATTENTION_BACKEND=""   # "" = auto-detect by vendor: NV->trtllm_mla, AMD->aiter
OUTPUT_DIR_OVERRIDE=""

ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)               MODEL_PATH="$2"; shift 2;;
    --tag)                 TAG="$2"; shift 2;;
    --docker)              DOCKER="$2"; shift 2;;
    --cache-mode)          CACHE_MODE="$2"; shift 2;;
    --cache-modes)         CACHE_MODES="$2"; shift 2;;
    --tp)                  TP_SIZE="$2"; shift 2;;
    --port)                PORT="$2"; shift 2;;
    --hicache-size)        HICACHE_SIZE="$2"; shift 2;;
    --host-headroom-gb)    HOST_HEADROOM_GB="$2"; shift 2;;
    --num-clients)         NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)          NUM_ROUNDS="$2"; shift 2;;
    --request-length)      REQUEST_LENGTH="$2"; shift 2;;
    --max-parallel)        MAX_PARALLEL="$2"; shift 2;;
    --request-rate)        REQUEST_RATE="$2"; shift 2;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; MEM_FRACTION_EXPLICIT=true; shift 2;;
    --L1-size)             L1_SIZE="$2"; shift 2;;
    --page-size)           PAGE_SIZE="$2"; PAGE_SIZE_EXPLICIT=true; shift 2;;
    --context-length)      CONTEXT_LENGTH="$2"; CONTEXT_LENGTH_EXPLICIT=true; shift 2;;
    --hicache-mem-layout)  HICACHE_MEM_LAYOUT="$2"; shift 2;;
    --hicache-io-backend)  HICACHE_IO_BACKEND="$2"; shift 2;;
    --attention-backend)   ATTENTION_BACKEND="$2"; shift 2;;
    --output-dir)          OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
    --gsm8k-num-questions) GSM8K_NUM_QUESTIONS="$2"; shift 2;;
    --no-gsm8k-precheck)   GSM8K_PRECHECK="false"; shift 1;;
    -h|--help)
      cat <<EOF
Usage: $0 --tag TAG --docker DOCKER [--cache-mode MODE | --cache-modes 'MODE1 MODE2 ...']
Modes: none | L1 | L2 | L3_file
Output:
  --output-dir DIR                 override default results/<docker>/<model>/bench-<tag>
                                   root. Per-mode subdirs are still created
                                   under this directory.

Memory sizing (pick one):
  --mem-fraction-static F        explicit fraction (default 0.85)
  --L1-size N                    GB/rank for L1 (GPU radix); auto-derives
                                 --mem-fraction-static via
                                 compute_profile_params.py. Requires
                                 --hicache-size when cache-mode is L2/L3_file.
                                 Mutually exclusive with --mem-fraction-static.
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[ -z "$TAG" ] && { echo "ERROR: --tag required" >&2; exit 1; }
[ ! -d "$MODEL_PATH" ] && { echo "ERROR: bad model path: $MODEL_PATH" >&2; exit 1; }
case "$CACHE_MODE" in
  none|L1|L2|L3_file) ;;
  *) echo "ERROR: bad --cache-mode: $CACHE_MODE (none|L1|L2|L3_file)" >&2; exit 1;;
esac
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# ============================== Chain dispatcher ==============================
# --cache-modes "none L1 L2 L3_file" → re-exec self per mode, pkill+sleep between.
if [ -n "$CACHE_MODES" ]; then
  # Pre-create base log dir so we can capture chain dispatcher stdout to it.
  # Mirrors the single-mode BASE_LOG_DIR computation below (line 219); kept
  # in sync manually because we need it before any child runs.
  if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
    CHAIN_BASE_LOG_DIR="$OUTPUT_DIR_OVERRIDE"
  else
    CHAIN_DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
    CHAIN_BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$CHAIN_DOCKER_FILENAME/${MODEL_NAME}/bench-${TAG}"
  fi
  mkdir -p "$CHAIN_BASE_LOG_DIR"
  CHAIN_LOG="$CHAIN_BASE_LOG_DIR/chain.log"
  : > "$CHAIN_LOG"
  exec > >(tee -a "$CHAIN_LOG") 2>&1

  FORWARD_ARGS=()
  i=0
  while [ "$i" -lt "${#ORIG_ARGS[@]}" ]; do
    case "${ORIG_ARGS[$i]}" in
      --cache-modes) i=$((i + 2)) ;;
      *) FORWARD_ARGS+=("${ORIG_ARGS[$i]}"); i=$((i + 1)) ;;
    esac
  done
  for MODE in $CACHE_MODES; do
    echo ""
    echo ">>> ============================================================"
    echo ">>> chain: starting cache_mode=${MODE}  (t=$(date +%H:%M:%S))"
    echo ">>> ============================================================"
    if ! "$0" --cache-mode "$MODE" "${FORWARD_ARGS[@]}"; then
      rc=$?
      echo ">>> chain: cache_mode=${MODE} FAILED with exit code ${rc}; continuing to next mode"
    fi
    pkill -9 sglang 2>/dev/null || true
    sleep 10
  done
  echo ">>> chain done (${CACHE_MODES})  (t=$(date +%H:%M:%S))"

  # Cross-mode summary: combine all per-mode bench_multiturn.jsonl +
  # cache_tiers.csv into a single $BASE_LOG_DIR/cascade_summary.csv. Re-
  # derive BASE_LOG_DIR here because the chain dispatcher exits before
  # the single-mode path computes it.
  CHAIN_MODEL_NAME=$(basename "${MODEL_PATH%/}")
  if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
    CHAIN_BASE_LOG_DIR="$OUTPUT_DIR_OVERRIDE"
  else
    CHAIN_DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
    CHAIN_BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$CHAIN_DOCKER_FILENAME/${CHAIN_MODEL_NAME}/bench-${TAG}"
  fi
  CHAIN_SUMMARIZER="$(dirname "$(readlink -f "$0")")/summarize_cascade.py"
  if [ -f "$CHAIN_SUMMARIZER" ] && [ -d "$CHAIN_BASE_LOG_DIR" ]; then
    python3 "$CHAIN_SUMMARIZER" cross_mode "$CHAIN_BASE_LOG_DIR" || true
  fi
  exit 0
fi

# ============================== Auto-derive mem-fraction-static ==============================
# When --L1-size is given, run compute_profile_params.py to get the
# mem-fraction-static that exactly fits weights/rank + L1 + buffer on
# the local GPU. Lets you run identical L1 sizes across MI355X/B200
# without hand-computing the fraction (HBM/rank differs per platform).
#
# Placed after the chain dispatcher: in chain mode the parent exits
# above without ever using MEM_FRACTION_STATIC, while each re-exec'd
# child hits this block exactly once for its own --cache-mode.
if [ -n "$L1_SIZE" ]; then
  if [ "$MEM_FRACTION_EXPLICIT" = true ]; then
    echo "ERROR: pass either --L1-size (auto mem-fraction) OR --mem-fraction-static, not both" >&2
    exit 1
  fi
  # Helper validates L2 >= L1 and uses L2 only for PROFILE_TARGET math
  # (which we don't consume here). For none/L1 modes L2 is irrelevant,
  # so we alias L2 = L1 just to pass validation. For L2/L3_file modes
  # we need a real --hicache-size.
  HELPER_L2="$L1_SIZE"
  if [ "$CACHE_MODE" = "L2" ] || [ "$CACHE_MODE" = "L3_file" ]; then
    if [ "$HICACHE_SIZE" = "auto" ]; then
      echo "ERROR: --L1-size with --cache-mode=$CACHE_MODE also requires explicit --hicache-size (no 'auto')" >&2
      exit 1
    fi
    HELPER_L2="$HICACHE_SIZE"
  fi
  HELPER="$(dirname "$(readlink -f "$0")")/compute_profile_params.py"
  [ -f "$HELPER" ] || { echo "ERROR: $HELPER not found" >&2; exit 1; }
  echo ">>> deriving --mem-fraction-static from --L1-size=${L1_SIZE} (L2-placeholder=${HELPER_L2})"
  PARAMS=$(python3 "$HELPER" \
    --model "$MODEL_PATH" \
    --tp "$TP_SIZE" \
    --L1-size "$L1_SIZE" \
    --L2-size "$HELPER_L2" \
    --num-clients "$NUM_CLIENTS" \
    --request-length "$REQUEST_LENGTH" 2>&1) || { echo "$PARAMS" >&2; exit 1; }
  echo "$PARAMS" | grep -E '^(WARN|ERROR|INFO)' >&2 || true
  # Whitelist only the vars we actually consume below. Helper also prints
  # NUM_ROUNDS=..., which would silently shadow the user's --num-rounds.
  # See bench_meta.json mismatch postmortem (cascade-FairCompare_0520).
  ALLOW='^(MEM_FRACTION_STATIC|PROFILE_TARGET_ROUND_1IDX|WEIGHTS_GB_PER_RANK|HBM_GB_PER_RANK)='
  eval "$(echo "$PARAMS" | grep -E "$ALLOW")"
  echo "    weights/rank=${WEIGHTS_GB_PER_RANK}GB  HBM/rank=${HBM_GB_PER_RANK}GB"
  echo "    derived: --mem-fraction-static=${MEM_FRACTION_STATIC}"
fi

# ============================== Auto-size hicache ==============================
if [ "$CACHE_MODE" = "none" ] || [ "$CACHE_MODE" = "L1" ]; then
  HICACHE_SIZE=0
elif [ "$HICACHE_SIZE" = "auto" ]; then
  sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  # Wait for MemAvailable to settle (prior pinned-host pool may still be releasing).
  prev=0
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    cur=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
    if [ "$prev" -gt 0 ] && [ $((cur - prev)) -le 5 ] && [ $((prev - cur)) -le 5 ]; then
      echo ">>> MemAvail settled at ${cur} GB"
      break
    fi
    prev=$cur
    sleep 5
  done
  MEM_AVAIL_GB=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
  USABLE_GB=$(( MEM_AVAIL_GB - HOST_HEADROOM_GB ))
  [ "$USABLE_GB" -le 0 ] && { echo "ERROR: only ${MEM_AVAIL_GB} GB avail, need >${HOST_HEADROOM_GB} headroom" >&2; exit 1; }
  PER_RANK=$(( USABLE_GB / TP_SIZE / 32 * 32 ))
  [ "$PER_RANK" -lt 32 ] && PER_RANK=32
  [ "$PER_RANK" -gt 512 ] && PER_RANK=512
  HICACHE_SIZE="$PER_RANK"
  echo ">>> auto hicache-size: ${HICACHE_SIZE} GB/rank (total $(( HICACHE_SIZE * TP_SIZE )) GB)"
fi

# ============================== Output dir + meta ==============================
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
  BASE_LOG_DIR="$OUTPUT_DIR_OVERRIDE"
else
  BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}/bench-${TAG}"
fi
case "$CACHE_MODE" in
  none|L1)  LOG_DIR="${BASE_LOG_DIR}/${CACHE_MODE}" ;;
  L2)       LOG_DIR="${BASE_LOG_DIR}/L2_size_${HICACHE_SIZE}" ;;
  # L3_file mode also uses HICACHE_SIZE as its L2 (host) pool size, so we
  # encode it as "L3file_L2_size_<N>" to disambiguate from a plain L2 run.
  L3_file)  LOG_DIR="${BASE_LOG_DIR}/L3file_L2_size_${HICACHE_SIZE}" ;;
esac
mkdir -p "$LOG_DIR"

{
  printf '%s' "$0"
  for a in "${ORIG_ARGS[@]}"; do
    printf ' %q' "$a"
  done
  printf '\n'
} > "$LOG_DIR/cmdline.txt"

META_HICACHE=$([ "$CACHE_MODE" = "none" ] || [ "$CACHE_MODE" = "L1" ] && echo "null" || echo "$HICACHE_SIZE")
META_PAGE_SIZE=$([ "$PAGE_SIZE_EXPLICIT" = true ] && echo "$PAGE_SIZE" || echo "null")
cat > "$LOG_DIR/bench_meta.json" <<EOF
{
  "cache_mode": "$CACHE_MODE",
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "page_size": $META_PAGE_SIZE,
  "tp_size": $TP_SIZE,
  "hicache_size_gb": $META_HICACHE,
  "hicache_write_policy": "$HICACHE_WRITE_POLICY",
  "hicache_mem_layout": "$HICACHE_MEM_LAYOUT",
  "hicache_io_backend": "$HICACHE_IO_BACKEND",
  "mem_fraction_static": $MEM_FRACTION_STATIC,
  "num_clients": $NUM_CLIENTS,
  "num_rounds": $NUM_ROUNDS,
  "request_length": $REQUEST_LENGTH,
  "max_parallel": $MAX_PARALLEL,
  "request_rate": $REQUEST_RATE,
  "tag": "$TAG",
  "docker": "$DOCKER",
  "gsm8k_precheck_accuracy": null
}
EOF

# ============================== L3 file store + disk check ==============================
HICACHE_FILE_STORE_DIR=""
if [[ "$CACHE_MODE" == L3_* ]]; then
  HICACHE_FILE_STORE_DIR="/tmp/cascade_dsr1_l3_${TAG}_${HICACHE_SIZE}"
  rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null
  mkdir -p "$HICACHE_FILE_STORE_DIR"
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_FILE_STORE_DIR"

  L2_TOTAL_GB=$(( HICACHE_SIZE * TP_SIZE ))
  L3_FREE_GB=$(df -BG --output=avail "$HICACHE_FILE_STORE_DIR" | tail -1 | tr -dc '0-9')
  echo ">>> L3 disk: free=${L3_FREE_GB} GB, L2 total=${L2_TOTAL_GB} GB"
  if [ "${L3_FREE_GB:-0}" -le "$L2_TOTAL_GB" ]; then
    echo "ERROR: L3 free (${L3_FREE_GB} GB) <= L2 total (${L2_TOTAL_GB} GB)" >&2
    exit 1
  fi
fi

# ============================== Trap ==============================
trap '
  rm -rf "${HICACHE_FILE_STORE_DIR:-}" 2>/dev/null
  [ -n "${CACHE_MONITOR_PID:-}" ] && kill "${CACHE_MONITOR_PID}" 2>/dev/null
  pkill -9 sglang 2>/dev/null || true
  sleep 10
  true' EXIT

# ============================== Platform detection ==============================
# attention_backend / moe_runner_backend / env vars are vendor-specific.
# We auto-detect once and gate every NV-only or AMD-only flag below.
VENDOR="unknown"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  VENDOR="nvidia"
fi
if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1; then
  # Pick AMD only when nvidia-smi isn't there OR ROCM_PATH is set (i.e.
  # we're inside an ROCm container). Same heuristic as the profile scripts.
  if [ "$VENDOR" = "unknown" ] || [ -n "${ROCM_PATH:-}" ]; then
    VENDOR="amd"
  fi
fi
echo ">>> platform: ${VENDOR}"

if [ -z "$ATTENTION_BACKEND" ]; then
  case "$VENDOR" in
    nvidia) ATTENTION_BACKEND="trtllm_mla";;   # MLA-aware NV backend (DSR1)
    amd)    ATTENTION_BACKEND="aiter";;        # AMD ROCm aiter
    *)      ATTENTION_BACKEND="aiter";;        # legacy default
  esac
fi

# ============================== Server cmd ==============================
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1
if [ "$VENDOR" = "amd" ]; then
  # Aligned to InferenceX dsr1_fp8_mi355x.sh:
  #   SGLANG_USE_AITER=1                       aiter prefill/decode path
  #   RCCL_MSCCL_ENABLE=0                      pin RCCL path (no MSCCL)
  #   ROCM_QUICK_REDUCE_QUANTIZATION=INT4      quick-allreduce quantized to INT4
  #                                            (was NONE; InferenceX uses INT4
  #                                            with no accuracy regression)
  #   SGLANG_AITER_FP8_PREFILL_ATTN=0          bf16 prefill kernel
  #                                            (gfx95 default is True; InferenceX
  #                                            relies on it but on this hardware
  #                                            bf16 and fp8 prefill are within
  #                                            noise -- see 8combinations sweep)
  export SGLANG_USE_AITER=1
  export RCCL_MSCCL_ENABLE=0
  export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
  # Honor caller-set value so sweeps can flip fp8 prefill on/off without
  # editing the script. Default off matches InferenceX behavior on this
  # hardware (bf16 prefill within noise of fp8 prefill per 8combinations).
  export SGLANG_AITER_FP8_PREFILL_ATTN=${SGLANG_AITER_FP8_PREFILL_ATTN:-0}
fi

SERVER_CMD=(
  python3 -u -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --tp "$TP_SIZE"
    --host "$HOST" --port "$PORT"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --enable-metrics
    --trust-remote-code
    --kv-cache-dtype fp8_e4m3
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --attention-backend "$ATTENTION_BACKEND"
)
# Only opt-in --page-size / --context-length when explicitly overridden. The
# defaults let sglang auto-pick page_size=1 (aiter MLA legacy) and
# context_length=163840 (model native), matching InferenceX.
[ "$PAGE_SIZE_EXPLICIT" = true ] && SERVER_CMD+=(--page-size "$PAGE_SIZE")
[ "$CONTEXT_LENGTH_EXPLICIT" = true ] && SERVER_CMD+=(--context-length "$CONTEXT_LENGTH")
if [ "$VENDOR" = "nvidia" ]; then
  # Matches the previously-validated B200 cascade (May-12 run): use
  # FlashInfer's TRT-LLM kernels for MoE + fused allreduce.
  SERVER_CMD+=(
    --moe-runner-backend flashinfer_trtllm
    --enable-flashinfer-allreduce-fusion
  )
fi
# Only pass --cuda-graph-max-bs if user explicitly overrode (>0).
[ "$CUDA_GRAPH_MAX_BS" -gt 0 ] && SERVER_CMD+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")

case "$CACHE_MODE" in
  none) SERVER_CMD+=(--disable-radix-cache);;
  L1)   :;;
  L2)
    # page_first_direct: page-contiguous in L2 (zero-copy L2<->L3) while keeping
    # same-layer tokens grouped within a page for aggregated L2->GPU transfers.
    # Requires sgl-kernel with PR #10339+ on ROCm (>= v0.5.4); v0.5.11 covers both
    # MI355X (rocm720 image) and B200 (cu130 image).
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-io-backend "$HICACHE_IO_BACKEND"
      --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
    );;
  L3_file)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-io-backend "$HICACHE_IO_BACKEND"
      --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
      --hicache-storage-backend file
      --hicache-storage-prefetch-policy best_effort
    );;
esac

# ============================== Launch + pre-check + health ==============================
SERVER_LOG="$LOG_DIR/server.log"
echo ">>> launching SGLang cache_mode=${CACHE_MODE}"
echo "${SERVER_CMD[*]}" | tee "$SERVER_LOG"

if pgrep sglang >/dev/null 2>&1; then
  echo "ERROR: sglang already running. 'pkill -9 sglang && sleep 10' first." >&2
  pgrep -a sglang >&2
  exit 1
fi
if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
  echo "ERROR: port ${PORT} already serves /health 200" >&2
  exit 1
fi

"${SERVER_CMD[@]}" 2>&1 | tee -a "$SERVER_LOG" &
SERVER_BG_PID=$!

deadline=$(( $(date +%s) + WAIT_FOR_SERVER_SEC ))
echo ">>> wait /health up to ${WAIT_FOR_SERVER_SEC}s..."
while [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; do
  if ! kill -0 "$SERVER_BG_PID" 2>/dev/null; then
    echo "ERROR: server pipeline died. tail of $SERVER_LOG:" >&2
    tail -n 50 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERROR: /health timeout, see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
echo ">>> server ready (pid=$SERVER_BG_PID)"

# ============================== Warmup ==============================
echo ">>> warmup"
# random-range-ratio 0.8 matches InferenceX (input lengths uniform in
# [0.8*N, N]); previous 1.0 was a fixed length that didn't reflect the
# distribution InferenceX uses.
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --model "$MODEL_PATH" --dataset-name random \
  --random-input 1024 --random-output 128 --random-range-ratio 0.8 \
  --max-concurrency 4 --num-prompt 8 --output-file /dev/null \
  2>&1 | tee "$LOG_DIR/warmup.log"

# ============================== GSM8K precheck ==============================
if [ "$GSM8K_PRECHECK" = "true" ]; then
  GSM8K_SCRIPT=""
  for c in /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
           "$HOME/work-space/sglang/benchmark/gsm8k/bench_sglang.py"; do
    [ -f "$c" ] && GSM8K_SCRIPT="$c" && break
  done
  if [ -n "$GSM8K_SCRIPT" ]; then
    echo ">>> GSM8K precheck (${GSM8K_NUM_QUESTIONS} q, parallel=${GSM8K_PARALLEL})"
    ( cd "$LOG_DIR" && python3 "$GSM8K_SCRIPT" \
        --host "$HOST" --port "$PORT" \
        --num-questions "$GSM8K_NUM_QUESTIONS" \
        --parallel "$GSM8K_PARALLEL" \
        --result-file "$LOG_DIR/Accuracy_GSM8K.jsonl" \
        2>&1 | tee "$LOG_DIR/Accuracy_GSM8K.log" ) || true
    ACC=$(grep -oP '^Accuracy:\s+\K[0-9.]+' "$LOG_DIR/Accuracy_GSM8K.log" | tail -1 || true)
    echo ">>> GSM8K accuracy: ${ACC:-NA}"
    python3 - "$LOG_DIR/bench_meta.json" "${ACC:-null}" <<'PY'
import json, sys
p, acc = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["gsm8k_precheck_accuracy"] = float(acc) if acc != "null" else None
json.dump(d, open(p, "w"), indent=2)
PY
  else
    echo ">>> WARNING: bench_sglang.py not found, skipping GSM8K"
  fi
fi

# ============================== Bench multiturn ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
[ -z "$BENCH_SCRIPT" ] && { echo "ERROR: bench_multiturn.py not found" >&2; exit 1; }

# Flush radix tree + OS page cache so cascade starts cold.
curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
sleep 2
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
sleep 2

# Optional cache_monitor sidecar (per-tier hit rate per round → cache_tiers.csv).
CACHE_MONITOR_SCRIPT="$(dirname "$(readlink -f "$0")")/cache_monitor.py"
CACHE_MONITOR_PID=""
if [ -f "$CACHE_MONITOR_SCRIPT" ]; then
  python3 "$CACHE_MONITOR_SCRIPT" \
      --url "http://${HOST}:${PORT}/metrics" \
      --interval 10 \
      --num-clients "$NUM_CLIENTS" \
      --num-rounds "$NUM_ROUNDS" \
      --csv "$LOG_DIR/cache_tiers.csv" \
      > "$LOG_DIR/cache_monitor.log" 2>&1 &
  CACHE_MONITOR_PID=$!
fi

echo ">>> bench multiturn N=${NUM_CLIENTS} R=${REQUEST_LENGTH} rounds=${NUM_ROUNDS}"
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

if [ -n "$CACHE_MONITOR_PID" ] && kill -0 "$CACHE_MONITOR_PID" 2>/dev/null; then
  kill "$CACHE_MONITOR_PID" 2>/dev/null || true
  wait "$CACHE_MONITOR_PID" 2>/dev/null || true
fi

# ============================== Per-mode round summary ==============================
# Joins this mode's bench_multiturn.jsonl (per-round TTFT + hit rate)
# with cache_tiers.csv (per-round L2 fill + L3 prefetch BW p50/p99) into
# a single round_summary.csv. Tolerates missing cache_tiers.csv (older
# runs) and missing/partial bench_multiturn.jsonl (failed runs).
SUMMARIZER="$(dirname "$(readlink -f "$0")")/summarize_cascade.py"
if [ -f "$SUMMARIZER" ]; then
  python3 "$SUMMARIZER" per_mode "$LOG_DIR" || true
fi

# ============================== Cleanup ==============================
echo ">>> stopping server"
pkill -9 sglang 2>/dev/null || true
sleep 10
[ -n "$HICACHE_FILE_STORE_DIR" ] && [ -d "$HICACHE_FILE_STORE_DIR" ] && rm -rf "$HICACHE_FILE_STORE_DIR"

echo ">>> done. results in: $LOG_DIR"
