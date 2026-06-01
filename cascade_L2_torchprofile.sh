#!/usr/bin/env bash
# cascade_L2_torchprofile.sh — capture a torch.profiler trace at the
# moment HiCache L1 (GPU radix cache) first overflows into L2 (host
# pinned pool).
#
# Usage:
#   ./cascade_L2_torchprofile.sh --tag X --model M --L1-size 40 --L2-size 80
#
# What it does:
#   1) computes weights/rank, HBM/rank, KV growth/round via
#      compute_profile_params.py
#   2) derives --mem-fraction-static, --hicache-size, --num-rounds so
#      L1 is 100% full by end of WARMUP_ROUNDS = ceil(L1 / KV_per_round)
#   3) launches its own SGLang server by invoking cascade_dsr1_lite.sh
#      in the background
#   4) waits for /health, then watches /metrics until the bench has
#      completed WARMUP_ROUNDS rounds
#   5) fires `python3 -m sglang.profiler` to capture the next prefill
#      step (= first L1-evicting round, profile target)
#   6) waits for the cascade to finish and tears everything down
#
# This makes B200 vs MI355X comparisons fair: same L1 / L2 footprint,
# same workload growth, profile target locked to the same physical
# cache state on both platforms.
#
# Cross-platform note:
#   B200  (CUDA):  --cpu --gpu  → torch.profiler with CPU+CUDA backend
#   MI355X (ROCm): --cpu --gpu  → torch.profiler with HIP via roctracer

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
HOST="localhost"
PORT=30000
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TP_SIZE=8
DOCKER=${DOCKER:-untagged-docker}

# Cache sizing (per rank, GB). Both required.
L1_SIZE=""
L2_SIZE=""
KV_BYTES_PER_TOKEN=$((34 * 1024))   # DSR1-0528 default: 34 KB
BUFFER_GB=12
CACHE_MODE="L2"        # script name says "L2" — keep server hierarchy =
                       # L1+L2 by default so L3 file backend doesn't kick
                       # in mid-run. Pass --cache-mode L3_file to opt in
                       # to the 3-tier run.
# --cache-modes "none L3_file": profile several modes sequentially. Each
# mode gets its own server + its own profile-<tag>_<mode> output dir
# (the single-mode profiler flow below is run once per mode via re-exec).
# Mutually-exclusive-ish with --cache-mode: when set, --cache-mode is
# ignored and each listed mode is run in turn.
CACHE_MODES=""
WAIT_FOR_HEALTH_SEC=1500

# Knobs the helper derives but the user can still override.
ROUNDS_PROFILE=1
NUM_ROUNDS_OVERRIDE=""

# Cascade workload knobs.
NUM_CLIENTS=300
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32

# Profiler knobs.
NUM_PROFILE_STEPS=20
PROFILE_START_ROUND_1IDX=""
TAG=""
OUTPUT_DIR=""

ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case $1 in
    --host)              HOST="$2"; shift 2;;
    --port)              PORT="$2"; shift 2;;
    --model)             MODEL_PATH="$2"; shift 2;;
    --tp)                TP_SIZE="$2"; shift 2;;
    --docker)
      # Support both:
      #   DOCKER=img ./script ...
      #   ./script --docker img ...
      # If users write `DOCKER=img ./script --docker "$DOCKER"`, the shell
      # expands "$DOCKER" before the temporary env assignment is visible, so
      # the argument can be empty. In that case keep the environment/default.
      [ -n "${2:-}" ] && DOCKER="$2"
      shift 2
      ;;
    --L1-size)           L1_SIZE="$2"; shift 2;;
    --L2-size)           L2_SIZE="$2"; shift 2;;
    --kv-bytes-per-token) KV_BYTES_PER_TOKEN="$2"; shift 2;;
    --buffer-gb)         BUFFER_GB="$2"; shift 2;;
    --cache-mode)        CACHE_MODE="$2"; shift 2;;
    --cache-modes)       CACHE_MODES="$2"; shift 2;;
    --rounds-profile)    ROUNDS_PROFILE="$2"; shift 2;;
    --num-rounds)        NUM_ROUNDS_OVERRIDE="$2"; shift 2;;
    --num-clients)       NUM_CLIENTS="$2"; shift 2;;
    --request-length)    REQUEST_LENGTH="$2"; shift 2;;
    --output-length)     OUTPUT_LENGTH="$2"; shift 2;;
    --max-parallel)      MAX_PARALLEL="$2"; shift 2;;
    --request-rate)      REQUEST_RATE="$2"; shift 2;;
    --num-profile-steps) NUM_PROFILE_STEPS="$2"; shift 2;;
    --profile-start-round) PROFILE_START_ROUND_1IDX="$2"; shift 2;;
    --tag)               TAG="$2"; shift 2;;
    --output-dir)        OUTPUT_DIR="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $0 --tag TAG --model PATH --L1-size N --L2-size N [opts]

All these are auto-derived from --L1-size and --L2-size:
  --mem-fraction-static (= (weights/rank + L1 + buffer) / HBM/rank)
  --hicache-size        (= L2-size)
  --num-rounds          (= warmup + profile + 1 margin)
  profile target round  (= first round where L1 is 100% full)

Constraints: L2-size >= L1-size (HiCache hierarchy invariant);
             (weights/rank + L1 + buffer) / HBM/rank <= 0.92.

Required: --tag TAG --model PATH --L1-size N --L2-size N

Common opts:
  --host/--port                server location (default localhost:30000)
  --tp N                       (default 8) TP size for server launch
  --docker NAME                container tag for results dir naming
  --kv-bytes-per-token N       (default 34*1024 for DSR1-0528 MLA fp8)
  --buffer-gb N                (default 12) per-rank HBM headroom
  --cache-mode MODE            (default L2; one of none|L1|L2|L3_file)
  --cache-modes 'M1 M2 ...'    profile several modes in turn; each runs its
                               own server and writes to profile-<tag>_<mode>.
                               When set, --cache-mode is ignored.
  --rounds-profile M           (default 1) rounds the profiler covers
  --num-rounds N               override auto-derived total rounds
  --num-clients N              (default 300)
  --request-length N           (default 4096)
  --num-profile-steps K        (default 20) torch.profiler --num-steps
  --profile-start-round N      start profiling at 1-based round N and stop
                               when that round completes (next round boundary).
                               Requires total rounds > N so the server stays
                               alive long enough to flush profile artifacts.
                               When omitted, use legacy --num-profile-steps.
  --output-dir DIR             default ~/SGLang-benchmarks/results/<docker>/<model>/profile-<tag>
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[ -z "$TAG" ] && { echo "ERROR: --tag required" >&2; exit 1; }
if [ -z "$L1_SIZE" ] || [ -z "$L2_SIZE" ]; then
  echo "ERROR: both --L1-size and --L2-size are required" >&2
  exit 1
fi

# ============================== Multi-mode chain dispatcher ==============================
# --cache-modes "none L3_file" → re-exec self once per mode. Each child is a
# full single-mode profiling run (its own server + profiler capture) writing
# to profile-<tag>_<mode>. The profiler flow below is single-mode by design,
# so multi-mode is handled purely by looping here. Mirrors the chain pattern
# in cascade_dsr1_lite.sh.
if [ -n "$CACHE_MODES" ]; then
  # Forward every original arg except the ones we re-derive per mode
  # (--cache-modes, --cache-mode, --tag); each take a value, so skip 2.
  FORWARD_ARGS=()
  i=0
  while [ "$i" -lt "${#ORIG_ARGS[@]}" ]; do
    case "${ORIG_ARGS[$i]}" in
      --cache-modes|--cache-mode|--tag) i=$((i + 2)) ;;
      *) FORWARD_ARGS+=("${ORIG_ARGS[$i]}"); i=$((i + 1)) ;;
    esac
  done
  for MODE in $CACHE_MODES; do
    case "$MODE" in
      none|L1|L2|L3_file) ;;
      *) echo "ERROR: bad mode in --cache-modes: '$MODE' (none|L1|L2|L3_file)" >&2; exit 1;;
    esac
    echo ""
    echo ">>> ============================================================"
    echo ">>> profile chain: cache_mode=${MODE}  tag=${TAG}_${MODE}  (t=$(date +%H:%M:%S))"
    echo ">>> ============================================================"
    if ! "$0" --cache-mode "$MODE" --tag "${TAG}_${MODE}" "${FORWARD_ARGS[@]}"; then
      rc=$?
      echo ">>> profile chain: cache_mode=${MODE} FAILED (exit ${rc}); continuing to next mode" >&2
    fi
    pkill -9 sglang 2>/dev/null || true
    sleep 10
  done
  echo ">>> profile chain done (${CACHE_MODES})  (t=$(date +%H:%M:%S))"
  exit 0
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

if [ "$VENDOR" = "nvidia" ]; then
  if ! python3 -c 'import distro' >/dev/null 2>&1; then
    echo ">>> NVIDIA environment missing python package 'distro'; installing"
    python3 -m pip install distro --break-system-packages
  fi
fi

# ============================== Output dir ==============================
MODEL_NAME=$(basename "${MODEL_PATH%/}")
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
WORKLOAD_SLUG="L1-${L1_SIZE}GB_L2-${L2_SIZE}GB-Client${NUM_CLIENTS}-ReqLen${REQUEST_LENGTH}"
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}/profile-${TAG}"
mkdir -p "$OUTPUT_DIR"
echo ">>> profile output dir: $OUTPUT_DIR"

# ============================== Derive params ==============================
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CASCADE_PID=""
CASCADE_LOG=""

HELPER="$SCRIPT_DIR/compute_profile_params.py"
[ -f "$HELPER" ] || { echo "ERROR: $HELPER not found" >&2; exit 1; }

echo ">>> computing params from --L1-size=${L1_SIZE} --L2-size=${L2_SIZE}"
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
# Regex needs digits too: PROFILE_TARGET_ROUND_1IDX has a `1` in it.
eval "$(echo "$PARAMS" | grep -E '^[A-Z0-9_]+=')"
echo "    weights/rank=${WEIGHTS_GB_PER_RANK}GB  HBM/rank=${HBM_GB_PER_RANK}GB"
echo "    KV/round/rank=${KV_GB_PER_ROUND_PER_RANK}GB  → WARMUP=${WARMUP_ROUNDS} rounds"
echo "    derived: mem-fraction-static=${MEM_FRACTION_STATIC}  num-rounds=${NUM_ROUNDS}"
echo "    profile target: round ${PROFILE_TARGET_ROUND_1IDX} (1-indexed)"
ROUNDS_WARMUP="$WARMUP_ROUNDS"
[ -n "$NUM_ROUNDS_OVERRIDE" ] && NUM_ROUNDS="$NUM_ROUNDS_OVERRIDE"

PROFILE_MODE="steps"
ROUNDS_BEFORE_PROFILE="$ROUNDS_WARMUP"
PROFILE_ROUND_1IDX="${PROFILE_TARGET_ROUND_1IDX:-$((ROUNDS_WARMUP + 1))}"
if [ -n "$PROFILE_START_ROUND_1IDX" ]; then
  if ! [[ "$PROFILE_START_ROUND_1IDX" =~ ^[0-9]+$ ]] || [ "$PROFILE_START_ROUND_1IDX" -lt 1 ]; then
    echo "ERROR: --profile-start-round must be a positive 1-based round number" >&2
    exit 1
  fi
  PROFILE_MODE="round"
  PROFILE_ROUND_1IDX="$PROFILE_START_ROUND_1IDX"
  ROUNDS_BEFORE_PROFILE=$(( PROFILE_ROUND_1IDX - 1 ))
  if [ -z "$NUM_ROUNDS_OVERRIDE" ] && [ "$NUM_ROUNDS" -lt $(( PROFILE_ROUND_1IDX + 1 )) ]; then
    NUM_ROUNDS=$(( PROFILE_ROUND_1IDX + 1 ))
    echo "    adjusted num-rounds=${NUM_ROUNDS} to leave one margin round after profile"
  elif [ "$NUM_ROUNDS" -le "$PROFILE_ROUND_1IDX" ]; then
    echo "ERROR: --num-rounds (${NUM_ROUNDS}) must be > --profile-start-round (${PROFILE_ROUND_1IDX})" >&2
    exit 1
  fi
  echo "    round-bounded profile: start round ${PROFILE_ROUND_1IDX}; stop at next round boundary"
fi
TOTAL_ROUNDS="$NUM_ROUNDS"

# Save derived params for post-mortem.
echo "$PARAMS" > "$OUTPUT_DIR/profile_params.txt"

# ============================== Helpers ==============================
metric_baseline_count() {
  curl -s "http://${HOST}:${PORT}/metrics" \
    | awk '
        /^sglang:num_requests_total[^_]/        { f += $NF }
        /^sglang:num_aborted_requests_total/    { a += $NF }
        END { printf "%d\n", f + a }'
}

profile_api() {
  local action="$1"
  local profile_dir="${2:-}"
  local profile_prefix="${3:-}"

  python3 - "$HOST" "$PORT" "$action" "$profile_dir" "$profile_prefix" <<'PY'
import json
import pathlib
import sys
import urllib.request

host, port, action, profile_dir, profile_prefix = sys.argv[1:6]
base_url = f"http://{host}:{port}"

def post(path, payload=None):
    data = b"" if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = resp.read().decode("utf-8", errors="replace").strip()
    if body:
        print(body)

if action == "start":
    out = pathlib.Path(profile_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(base_url + "/server_info", timeout=60) as resp:
            server_info = resp.read().decode("utf-8", errors="replace")
        (out / "server_args.json").write_text(server_info)
    except Exception as exc:
        print(f"WARN: failed to dump server_args.json: {exc}", file=sys.stderr)
    post(
        "/start_profile",
        {
            "output_dir": str(out),
            "activities": ["CPU", "GPU"],
            "profile_prefix": profile_prefix,
            "profile_by_stage": False,
        },
    )
elif action == "stop":
    post("/stop_profile")
else:
    raise SystemExit(f"unknown profile action: {action}")
PY
}

list_profiler_dirs() {
  # torch.profiler creates one timestamp-named child dir under OUTPUT_DIR.
  # Rename it after capture so artifacts are stable and meaningful.
  find "$OUTPUT_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
    | awk '/^[0-9]+([.][0-9]+)?$/ { print }' || true
}

normalize_profiler_artifacts() {
  local profile_dirs="$1"
  local src_dir
  local src_dir_path
  local target_dir_path
  local moved=0

  target_dir_path="$OUTPUT_DIR/$WORKLOAD_SLUG"
  mkdir -p "$target_dir_path"

  if [ -z "$profile_dirs" ]; then
    echo ">>> no timestamp profiler directory found under $OUTPUT_DIR"
    rename_trace_files "$target_dir_path"
    return 0
  fi

  while IFS= read -r src_dir; do
    [ -n "$src_dir" ] || continue
    src_dir_path="$OUTPUT_DIR/$src_dir"
    [ -d "$src_dir_path" ] || continue
    echo ">>> merging profiler dir: $src_dir -> $WORKLOAD_SLUG"
    shopt -s nullglob dotglob
    mv "$src_dir_path"/* "$target_dir_path"/
    shopt -u nullglob dotglob
    rmdir "$src_dir_path" 2>/dev/null || true
    moved=1
  done <<< "$profile_dirs"

  rename_trace_files "$target_dir_path"
  if [ "$moved" -eq 0 ]; then
    echo ">>> profiler artifacts already normalized under $target_dir_path"
  fi
}

rename_trace_files() {
  local profile_dir="$1"
  local trace_file
  local filename
  local new_name

  for trace_file in "$profile_dir"/*-TP-*.trace.json.gz; do
    [ -f "$trace_file" ] || continue
    filename=$(basename "$trace_file")
    new_name=$(sed -E "s/-[0-9]+([.][0-9]+)?-TP-/-${WORKLOAD_SLUG}-TP-/" <<< "$filename")
    if [ "$new_name" != "$filename" ]; then
      mv "$trace_file" "$profile_dir/$new_name"
      echo ">>> renamed trace: $filename -> $new_name"
    fi
  done
}

cleanup() {
  if [ -n "$CASCADE_PID" ] && kill -0 "$CASCADE_PID" 2>/dev/null; then
    echo ">>> killing cascade pid=$CASCADE_PID"
    kill "$CASCADE_PID" 2>/dev/null || true
    wait "$CASCADE_PID" 2>/dev/null || true
  fi
  # Server is owned by cascade_dsr1_lite.sh; if that died abnormally
  # there may still be sglang workers around.
  pkill -9 sglang 2>/dev/null || true
}
trap cleanup EXIT

# ============================== Launch cascade_dsr1_lite.sh ==============================
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
BENCH_OUTPUT_DIR="$OUTPUT_DIR/bench"
echo ">>> launching cascade_dsr1_lite.sh in background; log: $CASCADE_LOG (also streaming below)"
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
  --output-dir "$BENCH_OUTPUT_DIR" \
  2>&1 | tee "$CASCADE_LOG" &
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

# Phase 3: take baseline NOW so round math counts only cascade reqs.
BASELINE_DONE=$(metric_baseline_count)
TARGET_FOR_PROFILE=$(( BASELINE_DONE + ROUNDS_BEFORE_PROFILE * NUM_CLIENTS ))
STOP_FOR_PROFILE=$(( BASELINE_DONE + PROFILE_ROUND_1IDX * NUM_CLIENTS ))
echo ">>> baseline num_requests_done=${BASELINE_DONE}; trigger profiler at ${TARGET_FOR_PROFILE}"

# ============================== Wait until pre-profile rounds done ==============================
if [ "$PROFILE_MODE" = "round" ]; then
  echo ">>> waiting for round $(( PROFILE_ROUND_1IDX - 1 )) completion (${ROUNDS_BEFORE_PROFILE} rounds × ${NUM_CLIENTS} reqs)"
else
  echo ">>> waiting for warmup completion (${ROUNDS_WARMUP} rounds × ${NUM_CLIENTS} reqs)"
fi
LAST_LOG=0
while kill -0 "$CASCADE_PID" 2>/dev/null; do
  done_now=$(metric_baseline_count)
  reqs_done=$(( done_now - BASELINE_DONE ))
  if [ "$done_now" -ge "$TARGET_FOR_PROFILE" ]; then
    echo ">>> profile start boundary reached: reqs_done=${reqs_done} (${done_now} - ${BASELINE_DONE})"
    break
  fi
  now=$(date +%s)
  if [ $(( now - LAST_LOG )) -ge 5 ]; then
    echo "    progress: ${reqs_done}/$(( ROUNDS_BEFORE_PROFILE * NUM_CLIENTS )) pre-profile reqs"
    LAST_LOG=$now
  fi
  sleep 1
done
if ! kill -0 "$CASCADE_PID" 2>/dev/null; then
  echo "ERROR: cascade driver died before warmup completed" >&2
  tail -n 40 "$CASCADE_LOG" >&2 || true
  exit 1
fi

# ============================== Build profiler activities ==============================
export SGLANG_TORCH_PROFILER_DIR="$OUTPUT_DIR"

if [ "$PROFILE_MODE" = "round" ]; then
  PROFILE_RUN_DIR="$OUTPUT_DIR/$(python3 - <<'PY'
import time
print(time.time())
PY
)"
  PROFILE_PREFIX="${TAG}_${VENDOR}_R${PROFILE_ROUND_1IDX}_round"
  echo ">>> starting round-bounded profiler at round ${PROFILE_ROUND_1IDX}; dir: ${PROFILE_RUN_DIR}"
  profile_api start "$PROFILE_RUN_DIR" "$PROFILE_PREFIX" 2>&1 | tee "$OUTPUT_DIR/profiler.log"

  echo ">>> waiting to stop profiler at next round boundary (${PROFILE_ROUND_1IDX} rounds × ${NUM_CLIENTS} reqs)"
  LAST_LOG=0
  while kill -0 "$CASCADE_PID" 2>/dev/null; do
    done_now=$(metric_baseline_count)
    reqs_done=$(( done_now - BASELINE_DONE ))
    if [ "$done_now" -ge "$STOP_FOR_PROFILE" ]; then
      echo ">>> profile stop boundary reached: reqs_done=${reqs_done} (${done_now} - ${BASELINE_DONE})"
      break
    fi
    now=$(date +%s)
    if [ $(( now - LAST_LOG )) -ge 5 ]; then
      echo "    progress: ${reqs_done}/$(( PROFILE_ROUND_1IDX * NUM_CLIENTS )) profiled-round reqs"
      LAST_LOG=$now
    fi
    sleep 1
  done
  if ! kill -0 "$CASCADE_PID" 2>/dev/null; then
    echo "ERROR: cascade driver died before profile stop boundary" >&2
    tail -n 40 "$CASCADE_LOG" >&2 || true
    exit 1
  fi
  echo ">>> stopping round-bounded profiler"
  profile_api stop 2>&1 | tee -a "$OUTPUT_DIR/profiler.log"
  normalize_profiler_artifacts "$(list_profiler_dirs)"
else
  PROFILER_ARGS=(
    --url "http://${HOST}:${PORT}"
    --num-steps "$NUM_PROFILE_STEPS"
    --output-dir "$OUTPUT_DIR"
    --profile-prefix "${TAG}_${VENDOR}_R${PROFILE_TARGET_ROUND_1IDX:-X}"
    --profile-by-stage
    --cpu --gpu
  )
  echo ">>> calling sglang.profiler (blocks until ${NUM_PROFILE_STEPS} prefill+decode steps captured)"
  echo "    args: ${PROFILER_ARGS[*]}"
  python3 -m sglang.profiler "${PROFILER_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/profiler.log"
  PROFILE_DIRS_AFTER=$(list_profiler_dirs)
  normalize_profiler_artifacts "$PROFILE_DIRS_AFTER"
fi

echo ">>> profiler returned; waiting for cascade to finish remaining rounds"
wait "$CASCADE_PID" 2>/dev/null || true
trap - EXIT
pkill -9 sglang 2>/dev/null && sleep 5 || true
# Some torch.profiler backends flush traces asynchronously after the profiler
# command returns. Run normalization again after cascade/server teardown so
# timestamp dirs recreated during the flush are folded into the workload dir.
normalize_profiler_artifacts "$(list_profiler_dirs)"

# ============================== Summary ==============================
echo ""
echo ">>> done. artifacts in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR" 2>/dev/null | grep -E '\.(json|gz|pickle|trace|txt|log)' || true

cat <<EOF

Triage the trace with:
  python3 ~/PR/sglang/.claude/skills/sglang-torch-profiler-analysis/scripts/analyze_sglang_torch_profile.py \\
    --input $OUTPUT_DIR

Look for:
  - "Memcpy HtoD" gpu kernels  → L2 host-pinned -> device transfer
  - hiradix_cache.py:load_back / prefetch_from_storage frames (CPU side)
  - File backend syscalls (pread/io_uring) for L3 (cache-mode=L3_file only)
EOF
