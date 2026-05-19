#!/usr/bin/env bash
# cascade_L2_torchprofile.sh — torch.profiler capture against an *already
# running* SGLang server, sandwiched between a user-controlled number of
# warmup rounds and the profile-target round(s).
#
# Why this shape:
# - Cache state at round R depends on having continuously run rounds 0..R-1
#   with the same seed. Two separate bench_multiturn invocations don't
#   share session state, so we run ONE bench_multiturn covering
#   warmup + profile rounds, and watch /metrics from the side to fire
#   `python3 -m sglang.profiler` at the moment round (warmup+1) starts.
# - SGLang's HTTP-level profiler is the cross-platform path:
#     B200 (CUDA):   --cpu --gpu  → torch.profiler with CPU+CUDA backend
#     MI355X (ROCm): --cpu --gpu  → torch.profiler with HIP via roctracer
#                    (optionally --rpd for richer rocmProfileData layer)
#   Both produce Chrome-trace JSON in $SGLANG_TORCH_PROFILER_DIR.
#
# Prereqs:
# - SGLang server already running at $HOST:$PORT, launched with
#   --enable-metrics --enable-cache-report (and whatever HiCache flags).
# - The bench_multiturn.py + sglang.profiler must come from the SAME
#   SGLang version/install as the running server.

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
HOST="localhost"
PORT=30000
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
ROUNDS_WARMUP=7        # # of rounds to fill cache before profiling
ROUNDS_PROFILE=1       # # of rounds covered by the profiler
NUM_CLIENTS=300
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
NUM_PROFILE_STEPS=5    # forward steps profiler captures per stage
TAG=""
OUTPUT_DIR=""
USE_RPD="auto"         # auto|true|false (ROCm-only extra activity)
WAIT_FOR_HEALTH_SEC=60

while [[ $# -gt 0 ]]; do
  case $1 in
    --host)              HOST="$2"; shift 2;;
    --port)              PORT="$2"; shift 2;;
    --model)             MODEL_PATH="$2"; shift 2;;
    --rounds-warmup)     ROUNDS_WARMUP="$2"; shift 2;;
    --rounds-profile)    ROUNDS_PROFILE="$2"; shift 2;;
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

Run torch.profiler against a running SGLang server. The profile target is
the round immediately AFTER --rounds-warmup (so to profile round 8 of a
typical cascade, pass --rounds-warmup 7).

Required: --tag TAG
Common opts:
  --host/--port                server location (default localhost:30000)
  --model PATH                 must match server-launched model
  --rounds-warmup N            (default 7)
  --rounds-profile M           (default 1) rounds to keep profiler running
  --num-clients N              (default 300, must match cascade run)
  --request-length / --output-length / --max-parallel / --request-rate
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
TOTAL_ROUNDS=$(( ROUNDS_WARMUP + ROUNDS_PROFILE ))

# ============================== Locate bench script ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py" \
         "$HOME/PR/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
[ -z "$BENCH_SCRIPT" ] && { echo "ERROR: bench_multiturn.py not found" >&2; exit 1; }

# ============================== Detect platform ==============================
# ROCm and CUDA both expose the GPU via PyTorch's CUDA API in the
# profiler, but rpd gives a richer source of truth on AMD. Auto-on
# when rocm-smi exists *and* nvidia-smi doesn't.
VENDOR="unknown"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  VENDOR="nvidia"
fi
if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1; then
  # If both, prefer whichever PyTorch was built against (heuristic: ROCm
  # binaries set ROCM_PATH; CUDA-only environments don't).
  if [ "$VENDOR" = "unknown" ] || [ -n "${ROCM_PATH:-}" ]; then
    VENDOR="amd"
  fi
fi
case "$VENDOR" in
  nvidia) echo ">>> platform: NVIDIA";;
  amd)    echo ">>> platform: AMD ROCm";;
  *)      echo ">>> platform: unknown (no nvidia-smi or rocm-smi found, continuing anyway)";;
esac

if [ "$USE_RPD" = "auto" ]; then
  if [ "$VENDOR" = "amd" ]; then USE_RPD="true"; else USE_RPD="false"; fi
fi

# ============================== Output dir ==============================
TS=$(date +%Y%m%d_%H%M%S)
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$HOME/SGLang-benchmarks/profiles/${TAG}/${TS}"
mkdir -p "$OUTPUT_DIR"
echo ">>> profile output dir: $OUTPUT_DIR"

# ============================== Pre-flight ==============================
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; then
  echo "ERROR: $HOST:$PORT /health not 200; start server first" >&2
  exit 1
fi
# /metrics + /start_profile must be enabled (server started with
# --enable-metrics; profiler endpoint is on by default in v0.5+).
if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/metrics" 2>/dev/null)" != "200" ]; then
  echo "ERROR: /metrics not reachable; relaunch with --enable-metrics --enable-cache-report" >&2
  exit 1
fi

# Fresh slate: flush radix tree + drop OS page cache so the warmup
# rounds you asked for are the *only* warmup, not server startup
# residue.
echo ">>> flushing /flush_cache + drop_caches"
curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
sleep 2
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
sleep 2

# ============================== Helper: count requests via /metrics ==============================
metric_baseline_count() {
  curl -s "http://${HOST}:${PORT}/metrics" \
    | awk '
        /^sglang:num_requests_total[^_]/        { f += $NF }
        /^sglang:num_aborted_requests_total/    { a += $NF }
        END { printf "%d\n", f + a }'
}

BASELINE_DONE=$(metric_baseline_count)
TARGET_FOR_PROFILE=$(( BASELINE_DONE + ROUNDS_WARMUP * NUM_CLIENTS ))
echo ">>> baseline num_requests_done=${BASELINE_DONE}; trigger profiler at ${TARGET_FOR_PROFILE}"

# ============================== Launch bench (background) ==============================
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

cleanup() {
  if kill -0 "$BENCH_PID" 2>/dev/null; then
    echo ">>> killing bench_multiturn (pid=$BENCH_PID)"
    kill "$BENCH_PID" 2>/dev/null || true
    wait "$BENCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# ============================== Wait until warmup rounds done ==============================
echo ">>> waiting for warmup completion (${ROUNDS_WARMUP} rounds × ${NUM_CLIENTS} reqs)"
LAST_LOG=0
while kill -0 "$BENCH_PID" 2>/dev/null; do
  done_now=$(metric_baseline_count)
  reqs_done=$(( done_now - BASELINE_DONE ))
  if [ "$done_now" -ge "$TARGET_FOR_PROFILE" ]; then
    echo ">>> warmup complete: reqs_done=${reqs_done} (${done_now} - ${BASELINE_DONE})"
    break
  fi
  # Throttle log to once every 5s to keep stdout readable.
  now=$(date +%s)
  if [ $(( now - LAST_LOG )) -ge 5 ]; then
    echo "    progress: ${reqs_done}/$(( ROUNDS_WARMUP * NUM_CLIENTS )) warmup reqs"
    LAST_LOG=$now
  fi
  sleep 1
done

if ! kill -0 "$BENCH_PID" 2>/dev/null; then
  echo "ERROR: bench_multiturn died before warmup completed; tail of $BENCH_LOG:" >&2
  tail -n 40 "$BENCH_LOG" >&2 || true
  exit 1
fi

# ============================== Build profiler activities ==============================
PROFILER_ARGS=(
  --url "http://${HOST}:${PORT}"
  --num-steps "$NUM_PROFILE_STEPS"
  --output-dir "$OUTPUT_DIR"
  --profile-prefix "${TAG}_${VENDOR}"
  --profile-by-stage
  --cpu --gpu
)
if [ "$USE_RPD" = "true" ]; then
  echo ">>> enabling --rpd (ROCm rocmProfileData)"
  PROFILER_ARGS+=(--rpd)
fi

export SGLANG_TORCH_PROFILER_DIR="$OUTPUT_DIR"

echo ">>> calling sglang.profiler (this blocks until ${NUM_PROFILE_STEPS} prefill+decode steps captured)"
echo "    args: ${PROFILER_ARGS[*]}"
python3 -m sglang.profiler "${PROFILER_ARGS[@]}" 2>&1 | tee "$OUTPUT_DIR/profiler.log"

echo ">>> profiler call returned; waiting for bench to finish remaining rounds"
wait "$BENCH_PID" 2>/dev/null || true
trap - EXIT

# ============================== Summary ==============================
echo ""
echo ">>> done. artifacts in: $OUTPUT_DIR"
echo "    - profiler.log          (sglang.profiler stdout)"
echo "    - bench_multiturn.log   (continuous bench output)"
echo "    - bench_multiturn.jsonl (per-round summary)"
ls -la "$OUTPUT_DIR" | grep -E '\.(json|gz|pickle|rpd|trace)' || true

cat <<EOF

Next: triage the trace with the SGLang skill (recommended).
  python3 ~/PR/sglang/.claude/skills/sglang-torch-profiler-analysis/scripts/analyze_sglang_torch_profile.py \\
    --input $OUTPUT_DIR

Or load the .trace.json.gz in chrome://tracing or https://ui.perfetto.dev.
Look for:
  - "Memcpy HtoD" gpu kernels  → L2 host-pinned -> device transfer
  - hiradix_cache.py:load_back / prefetch_from_storage frames (CPU side)
  - File backend syscalls (pread/io_uring) for L3
EOF
