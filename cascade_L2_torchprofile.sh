#!/usr/bin/env bash
# cascade_L2_torchprofile.sh — capture a torch.profiler trace at the
# moment HiCache L1 (GPU radix cache) first overflows into L2 (host
# pinned pool).
#
# Two modes:
#
# AUTO (recommended; takes --L1-size + --L2-size):
#   ./cascade_L2_torchprofile.sh --tag X --model M --L1-size 40 --L2-size 80
#
#   The script:
#     1) computes weights/rank, HBM/rank, KV growth/round via
#        compute_profile_params.py
#     2) derives --mem-fraction-static, --hicache-size, --num-rounds so
#        L1 fills in WARMUP_ROUNDS = floor(L1 / KV_per_round) rounds
#     3) launches its own SGLang server by invoking cascade_dsr1_lite.sh
#        in the background
#     4) waits for /health, then watches /metrics until the bench has
#        completed WARMUP_ROUNDS rounds
#     5) fires `python3 -m sglang.profiler` to capture the next prefill
#        step (= first L1-evicting round, profile target)
#     6) waits for the cascade to finish and tears everything down
#
#   This makes B200 vs MI355X comparisons fair: same L1 / L2 footprint,
#   same workload growth, profile target locked to the same physical
#   cache state on both platforms.
#
# MANUAL (legacy; --rounds-warmup):
#   Assumes you've already launched a server externally with matching
#   --mem-fraction-static / --hicache-size and want to drive only the
#   profiler portion. Backward-compatible with the old invocation.
#
# Cross-platform note:
#   B200  (CUDA): --cpu --gpu  → torch.profiler with CPU+CUDA backend
#   MI355X (ROCm): --cpu --gpu  → torch.profiler with HIP via roctracer
#   On AMD, --rpd auto-enables for richer rocmProfileData layer.

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
HOST="localhost"
PORT=30000
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TP_SIZE=8
DOCKER="untagged-docker"

# AUTO mode knobs
L1_SIZE=""             # GB per rank; trigger AUTO mode when set
L2_SIZE=""             # GB per rank; must be >= L1_SIZE
KV_BYTES_PER_TOKEN=$((34 * 1024))   # DSR1-0528 default: 34 KB
BUFFER_GB=12
CACHE_MODE="L3_file"   # AUTO mode passes this to cascade_dsr1_lite.sh
WAIT_FOR_HEALTH_SEC=1500

# MANUAL mode knobs (also serve as overrides in AUTO mode)
ROUNDS_WARMUP=""
ROUNDS_PROFILE=1
NUM_ROUNDS_OVERRIDE=""

# Common cascade workload knobs (shared by both modes)
NUM_CLIENTS=300
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32

# Profiler knobs
NUM_PROFILE_STEPS=5
TAG=""
OUTPUT_DIR=""
USE_RPD="auto"

while [[ $# -gt 0 ]]; do
  case $1 in
    --host)              HOST="$2"; shift 2;;
    --port)              PORT="$2"; shift 2;;
    --model)             MODEL_PATH="$2"; shift 2;;
    --tp)                TP_SIZE="$2"; shift 2;;
    --docker)            DOCKER="$2"; shift 2;;
    --L1-size)           L1_SIZE="$2"; shift 2;;
    --L2-size)           L2_SIZE="$2"; shift 2;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN="$2"; shift 2;;
    --buffer-gb)         BUFFER_GB="$2"; shift 2;;
    --cache-mode)        CACHE_MODE="$2"; shift 2;;
    --rounds-warmup)     ROUNDS_WARMUP="$2"; shift 2;;
    --rounds-profile)    ROUNDS_PROFILE="$2"; shift 2;;
    --num-rounds)        NUM_ROUNDS_OVERRIDE="$2"; shift 2;;
    --num-clients)       NUM_CLIENTS="$2"; shift 2;;
    --request-length)    REQUEST_LENGTH="$2"; shift 2;;
    --output-length)     OUTPUT_LENGTH="$2"; shift 2;;
    --max-parallel)      MAX_PARALLEL="$2"; shift 2;;
    --request-rate)      REQUEST_RATE="$2"; shift 2;;
    --num-profile-steps) NUM_PROFILE_STEPS="$2"; shift 2;;
    --tag)               TAG="$2"; shift 2;;
    --output-dir)        OUTPUT_DIR="$2"; shift 2;;
    --rpd)               USE_RPD="true"; shift 1;;
    --no-rpd)            USE_RPD="false"; shift 1;;
    -h|--help)
      cat <<EOF
Usage: $0 --tag TAG [opts]

AUTO mode (recommended for fair B200 vs MI355X comparison):
  $0 --tag X --model PATH --L1-size 40 --L2-size 80 [--docker IMG]

  All these are auto-derived from --L1-size and --L2-size:
    --mem-fraction-static (= (weights/rank + L1 + buffer) / HBM/rank)
    --hicache-size        (= L2-size)
    --num-rounds          (= warmup + profile + 1 margin)
    profile target round  (= first round where L1 evicts)

  Constraints: L2-size >= L1-size (HiCache hierarchy invariant);
               (weights/rank + L1 + buffer) / HBM/rank <= 0.92.

MANUAL mode (legacy; server must already be running):
  $0 --tag X --rounds-warmup 7 --rounds-profile 1 [other knobs]

Required: --tag TAG
AUTO-mode required: --model PATH --L1-size N --L2-size N

Common opts:
  --host/--port                server location (default localhost:30000)
  --tp N                       (default 8) TP size for AUTO-mode launch
  --docker NAME                container tag for results dir naming
  --kv-bytes-per-token N       (default 34*1024 for DSR1-0528 MLA fp8)
  --buffer-gb N                (default 12) per-rank HBM headroom in AUTO
  --cache-mode MODE            (default L3_file) for AUTO-mode launch
  --rounds-profile M           (default 1) rounds the profiler covers
  --num-rounds N               override auto-derived total rounds
  --num-clients N              (default 300)
  --request-length N           (default 4096)
  --num-profile-steps K        (default 5) torch.profiler --num-steps
  --output-dir DIR             default ~/SGLang-benchmarks/profiles/<tag>/<ts>
  --rpd / --no-rpd             force ROCm rpd profiler on/off (auto-detect)
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[ -z "$TAG" ] && { echo "ERROR: --tag required" >&2; exit 1; }

# ============================== Mode arbitration ==============================
# AUTO when --L1-size given. Manual when --rounds-warmup given. Reject
# half-specified states (e.g. --L1-size without --L2-size) early so the
# user gets a clear message instead of a silent default.
AUTO_MODE=false
if [ -n "$L1_SIZE" ] || [ -n "$L2_SIZE" ]; then
  if [ -z "$L1_SIZE" ] || [ -z "$L2_SIZE" ]; then
    echo "ERROR: AUTO mode requires BOTH --L1-size and --L2-size" >&2
    exit 1
  fi
  AUTO_MODE=true
fi
if [ "$AUTO_MODE" = false ] && [ -z "$ROUNDS_WARMUP" ]; then
  echo "ERROR: must pass either --L1-size + --L2-size (AUTO) or --rounds-warmup (MANUAL)" >&2
  exit 1
fi

# ============================== Detect platform ==============================
VENDOR="unknown"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  VENDOR="nvidia"
fi
if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1; then
  if [ "$VENDOR" = "unknown" ] || [ -n "${ROCM_PATH:-}" ]; then
    VENDOR="amd"
  fi
fi
case "$VENDOR" in
  nvidia) echo ">>> platform: NVIDIA";;
  amd)    echo ">>> platform: AMD ROCm";;
  *)      echo ">>> platform: unknown (no nvidia-smi or rocm-smi found, continuing anyway)";;
esac
[ "$USE_RPD" = "auto" ] && {
  if [ "$VENDOR" = "amd" ]; then USE_RPD="true"; else USE_RPD="false"; fi
}

# ============================== Output dir ==============================
TS=$(date +%Y%m%d_%H%M%S)
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$HOME/SGLang-benchmarks/profiles/${TAG}/${TS}"
mkdir -p "$OUTPUT_DIR"
echo ">>> profile output dir: $OUTPUT_DIR"

# ============================== AUTO mode: derive params ==============================
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CASCADE_PID=""
CASCADE_LOG=""

if [ "$AUTO_MODE" = true ]; then
  HELPER="$SCRIPT_DIR/compute_profile_params.py"
  [ -f "$HELPER" ] || { echo "ERROR: $HELPER not found" >&2; exit 1; }

  echo ">>> AUTO mode: computing params from --L1-size=${L1_SIZE} --L2-size=${L2_SIZE}"
  PARAMS=$(python3 "$HELPER" \
    --model "$MODEL_PATH" \
    --tp "$TP_SIZE" \
    --L1-size "$L1_SIZE" \
    --L2-size "$L2_SIZE" \
    --num-clients "$NUM_CLIENTS" \
    --request-length "$REQUEST_LENGTH" \
    --kv-bytes-per-token "$KV_BYTES_PER_TOKEN" \
    --buffer-gb "$BUFFER_GB" \
    --rounds-profile "$ROUNDS_PROFILE" 2>&1) || {
      echo "$PARAMS" >&2
      exit 1
    }
  echo "$PARAMS" | grep -E '^(WARN|ERROR)' >&2 || true
  eval "$(echo "$PARAMS" | grep -E '^[A-Z_]+=')"
  echo "    weights/rank=${WEIGHTS_GB_PER_RANK}GB  HBM/rank=${HBM_GB_PER_RANK}GB"
  echo "    KV/round/rank=${KV_GB_PER_ROUND_PER_RANK}GB  → WARMUP=${WARMUP_ROUNDS} rounds"
  echo "    derived: mem-fraction-static=${MEM_FRACTION_STATIC}  num-rounds=${NUM_ROUNDS}"
  echo "    profile target: round ${PROFILE_TARGET_ROUND_1IDX} (1-indexed)"
  ROUNDS_WARMUP="$WARMUP_ROUNDS"
  if [ -n "$NUM_ROUNDS_OVERRIDE" ]; then
    NUM_ROUNDS="$NUM_ROUNDS_OVERRIDE"
  fi
  TOTAL_ROUNDS="$NUM_ROUNDS"

  # Save derived params for post-mortem.
  echo "$PARAMS" > "$OUTPUT_DIR/profile_params.txt"
fi

[ "$AUTO_MODE" = false ] && TOTAL_ROUNDS=$(( ROUNDS_WARMUP + ROUNDS_PROFILE ))

# ============================== Locate bench script ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py" \
         "$HOME/PR/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
[ -z "$BENCH_SCRIPT" ] && { echo "ERROR: bench_multiturn.py not found" >&2; exit 1; }

# ============================== Helpers ==============================
metric_baseline_count() {
  curl -s "http://${HOST}:${PORT}/metrics" \
    | awk '
        /^sglang:num_requests_total[^_]/        { f += $NF }
        /^sglang:num_aborted_requests_total/    { a += $NF }
        END { printf "%d\n", f + a }'
}

# Cleanup runs both bench/cascade subprocess and any orphan loggers.
BENCH_PID=""
cleanup() {
  for pid in "$BENCH_PID" "$CASCADE_PID"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo ">>> killing pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  # In AUTO mode the server is owned by cascade_dsr1_lite.sh; if that
  # process died abnormally there may still be sglang workers around.
  [ "$AUTO_MODE" = true ] && pkill -9 sglang 2>/dev/null || true
}
trap cleanup EXIT

# ============================== AUTO: launch cascade_dsr1_lite.sh ==============================
if [ "$AUTO_MODE" = true ]; then
  CASCADE_LITE="$SCRIPT_DIR/cascade_dsr1_lite.sh"
  [ -f "$CASCADE_LITE" ] || { echo "ERROR: $CASCADE_LITE not found" >&2; exit 1; }

  if pgrep -f sglang.launch_server >/dev/null 2>&1; then
    echo "ERROR: sglang already running. 'pkill -9 sglang && sleep 10' first." >&2
    exit 1
  fi
  if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
    echo "ERROR: port ${PORT} already serves /health 200" >&2
    exit 1
  fi

  CASCADE_LOG="$OUTPUT_DIR/cascade_dsr1_lite.log"
  echo ">>> launching cascade_dsr1_lite.sh in background; log: $CASCADE_LOG"
  "$CASCADE_LITE" \
    --tag "${TAG}_L1${L1_SIZE}_L2${L2_SIZE}" \
    --docker "$DOCKER" \
    --model "$MODEL_PATH" \
    --tp "$TP_SIZE" \
    --port "$PORT" \
    --cache-mode "$CACHE_MODE" \
    --hicache-size "$L2_SIZE" \
    --mem-fraction-static "$MEM_FRACTION_STATIC" \
    --num-clients "$NUM_CLIENTS" \
    --num-rounds "$TOTAL_ROUNDS" \
    --request-length "$REQUEST_LENGTH" \
    --max-parallel "$MAX_PARALLEL" \
    --request-rate "$REQUEST_RATE" \
    --no-gsm8k-precheck \
    > "$CASCADE_LOG" 2>&1 &
  CASCADE_PID=$!

  # Phase 1: wait for server health.
  echo ">>> waiting up to ${WAIT_FOR_HEALTH_SEC}s for /health=200..."
  deadline=$(( $(date +%s) + WAIT_FOR_HEALTH_SEC ))
  while true; do
    if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
      echo ">>> server up"
      break
    fi
    if ! kill -0 "$CASCADE_PID" 2>/dev/null; then
      echo "ERROR: cascade_dsr1_lite.sh died before server came up; tail of $CASCADE_LOG:" >&2
      tail -n 40 "$CASCADE_LOG" >&2 || true
      exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "ERROR: /health timeout" >&2
      exit 1
    fi
    sleep 5
  done

  # Phase 2: wait for the bench start marker. cascade_dsr1_lite.sh runs
  # an 8-prompt warmup + flush before bench, so we lock onto the line
  # printed right before bench_multiturn fires.
  echo ">>> waiting for bench_multiturn to start"
  deadline=$(( $(date +%s) + 600 ))
  while true; do
    if [ -f "$CASCADE_LOG" ] && grep -q '^>>> bench multiturn' "$CASCADE_LOG" 2>/dev/null; then
      break
    fi
    if ! kill -0 "$CASCADE_PID" 2>/dev/null; then
      echo "ERROR: cascade_dsr1_lite.sh died before bench started" >&2
      tail -n 60 "$CASCADE_LOG" >&2 || true
      exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "ERROR: bench start timeout" >&2
      exit 1
    fi
    sleep 2
  done
  echo ">>> bench_multiturn started"

  # Phase 3: take baseline NOW so warmup math counts only cascade reqs.
  BASELINE_DONE=$(metric_baseline_count)
  TARGET_FOR_PROFILE=$(( BASELINE_DONE + ROUNDS_WARMUP * NUM_CLIENTS ))
  echo ">>> baseline num_requests_done=${BASELINE_DONE}; trigger profiler at ${TARGET_FOR_PROFILE}"

else
  # ============================== MANUAL mode pre-flight ==============================
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; then
    echo "ERROR: $HOST:$PORT /health not 200; start server first" >&2
    exit 1
  fi
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/metrics" 2>/dev/null)" != "200" ]; then
    echo "ERROR: /metrics not reachable; relaunch with --enable-metrics --enable-cache-report" >&2
    exit 1
  fi
  echo ">>> flushing /flush_cache + drop_caches"
  curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
  sleep 2
  sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  sleep 2

  BASELINE_DONE=$(metric_baseline_count)
  TARGET_FOR_PROFILE=$(( BASELINE_DONE + ROUNDS_WARMUP * NUM_CLIENTS ))
  echo ">>> baseline num_requests_done=${BASELINE_DONE}; trigger profiler at ${TARGET_FOR_PROFILE}"

  BENCH_LOG="$OUTPUT_DIR/bench_multiturn.log"
  BENCH_JSONL="$OUTPUT_DIR/bench_multiturn.jsonl"
  echo ">>> launching bench_multiturn (total rounds=${TOTAL_ROUNDS})"
  python3 "$BENCH_SCRIPT" \
    --host "$HOST" --port "$PORT" \
    --model-path "$MODEL_PATH" \
    --num-clients "$NUM_CLIENTS" \
    --num-rounds "$TOTAL_ROUNDS" \
    --request-length "$REQUEST_LENGTH" \
    --output-length "$OUTPUT_LENGTH" \
    --max-parallel "$MAX_PARALLEL" \
    --request-rate "$REQUEST_RATE" \
    --ready-queue-policy random \
    --log-file "$BENCH_JSONL" \
    --tag "${TAG}-warmup${ROUNDS_WARMUP}-profile${ROUNDS_PROFILE}" \
    --disable-random-sample \
    --disable-auto-run \
    --enable-round-barrier \
    > "$BENCH_LOG" 2>&1 &
  BENCH_PID=$!
fi

# ============================== Wait until warmup rounds done ==============================
echo ">>> waiting for warmup completion (${ROUNDS_WARMUP} rounds × ${NUM_CLIENTS} reqs)"
LAST_LOG=0
DRIVER_PID="$BENCH_PID"
[ "$AUTO_MODE" = true ] && DRIVER_PID="$CASCADE_PID"
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  done_now=$(metric_baseline_count)
  reqs_done=$(( done_now - BASELINE_DONE ))
  if [ "$done_now" -ge "$TARGET_FOR_PROFILE" ]; then
    echo ">>> warmup complete: reqs_done=${reqs_done} (${done_now} - ${BASELINE_DONE})"
    break
  fi
  now=$(date +%s)
  if [ $(( now - LAST_LOG )) -ge 5 ]; then
    echo "    progress: ${reqs_done}/$(( ROUNDS_WARMUP * NUM_CLIENTS )) warmup reqs"
    LAST_LOG=$now
  fi
  sleep 1
done
if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
  echo "ERROR: cascade driver died before warmup completed" >&2
  [ -n "${BENCH_LOG:-}" ] && tail -n 40 "$BENCH_LOG" >&2 || true
  [ -n "${CASCADE_LOG:-}" ] && tail -n 40 "$CASCADE_LOG" >&2 || true
  exit 1
fi

# ============================== Build profiler activities ==============================
PROFILER_ARGS=(
  --url "http://${HOST}:${PORT}"
  --num-steps "$NUM_PROFILE_STEPS"
  --output-dir "$OUTPUT_DIR"
  --profile-prefix "${TAG}_${VENDOR}_R${PROFILE_TARGET_ROUND_1IDX:-X}"
  --profile-by-stage
  --cpu --gpu
)
if [ "$USE_RPD" = "true" ]; then
  echo ">>> enabling --rpd (ROCm rocmProfileData)"
  PROFILER_ARGS+=(--rpd)
fi

export SGLANG_TORCH_PROFILER_DIR="$OUTPUT_DIR"

echo ">>> calling sglang.profiler (blocks until ${NUM_PROFILE_STEPS} prefill+decode steps captured)"
echo "    args: ${PROFILER_ARGS[*]}"
python3 -m sglang.profiler "${PROFILER_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/profiler.log"

echo ">>> profiler returned; waiting for cascade to finish remaining rounds"
wait "$DRIVER_PID" 2>/dev/null || true
trap - EXIT
[ "$AUTO_MODE" = true ] && pkill -9 sglang 2>/dev/null && sleep 5 || true

# ============================== Summary ==============================
echo ""
echo ">>> done. artifacts in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" 2>/dev/null | grep -E '\.(json|gz|pickle|rpd|trace|txt|log)' || true

cat <<EOF

Triage the trace with:
  python3 ~/PR/sglang/.claude/skills/sglang-torch-profiler-analysis/scripts/analyze_sglang_torch_profile.py \\
    --input $OUTPUT_DIR

Look for:
  - "Memcpy HtoD" gpu kernels  → L2 host-pinned -> device transfer
  - hiradix_cache.py:load_back / prefetch_from_storage frames (CPU side)
  - File backend syscalls (pread/io_uring) for L3
EOF
