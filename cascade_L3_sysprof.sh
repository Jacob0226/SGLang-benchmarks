#!/usr/bin/env bash
# cascade_L3_sysprof.sh — system-level profile for HiCache L3 (storage
# backend = file). Captures the three L3 bottleneck candidates in
# parallel, time-aligned, while a user-specified slice of a cascade run
# is in flight.
#
# Bottleneck candidates being profiled (in order of likelihood):
#   1) Disk read bandwidth        → iostat -xm (Linux block layer; same
#                                   on B200 host and MI355X host)
#   2) PCIe H2D bandwidth         → vendor-specific:
#                                     NVIDIA: nvidia-smi pcie --query-pcie
#                                     AMD:    rocprofv3 hip-trace slice
#                                             (real-time PCIe BW counter
#                                             not exposed on ROCm; we
#                                             instead capture all HIP
#                                             memcpy events for the slice
#                                             and post-process)
#   3) Filesystem syscall stalls  → strace summary on backend pwrite/pread
#                                   (optional, --enable-strace)
#
# Cross-platform note: iostat is the universal signal. On both vendors,
# the L3 file backend goes through the kernel block layer, so iostat tells
# you whether the disk is saturated. PCIe sampling is vendor-specific and
# strictly secondary — only matters if iostat shows disk has slack but L3
# is still slow.
#
# Like cascade_L2_torchprofile.sh, this expects an already-running server
# and runs ONE bench_multiturn covering warmup + profile rounds, with the
# loggers active only during the profile slice.

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
HOST="localhost"
PORT=30000
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TP_SIZE=8
DOCKER="untagged-docker"

# AUTO mode knobs (mirror cascade_L2_torchprofile.sh)
L1_SIZE=""             # GB per rank; setting this triggers AUTO mode
L2_SIZE=""             # GB per rank; must be >= L1_SIZE
KV_BYTES_PER_TOKEN=$((34 * 1024))   # DSR1-0528 default
BUFFER_GB=12
CACHE_MODE="L3_file"
WAIT_FOR_HEALTH_SEC=1500

# MANUAL mode + shared knobs
ROUNDS_WARMUP=""
ROUNDS_PROFILE=2       # default 2 so the slice has time for loggers
NUM_ROUNDS_OVERRIDE=""
NUM_CLIENTS=300
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32

TAG=""
OUTPUT_DIR=""
SAMPLE_INTERVAL_MS=200
ENABLE_STRACE=false
HICACHE_FILE_DIR=${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-}

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
    --tag)               TAG="$2"; shift 2;;
    --output-dir)        OUTPUT_DIR="$2"; shift 2;;
    --interval-ms)       SAMPLE_INTERVAL_MS="$2"; shift 2;;
    --hicache-file-dir)  HICACHE_FILE_DIR="$2"; shift 2;;
    --enable-strace)     ENABLE_STRACE=true; shift 1;;
    -h|--help)
      cat <<EOF
Usage: $0 --tag TAG [opts]

AUTO mode (recommended; fair B200 vs MI355X):
  $0 --tag X --model PATH --L1-size 40 --L2-size 80 [--docker IMG]
  Auto-derives mem-fraction-static / hicache-size / num-rounds via
  compute_profile_params.py and launches its own server through
  cascade_dsr1_lite.sh. Loggers run for ROUNDS_PROFILE rounds starting
  the moment L1 first overflows (= floor(L1/KV_per_round) + 1).

MANUAL mode (legacy; server must already be running):
  $0 --tag X --rounds-warmup 7 --rounds-profile 2

Required: --tag TAG
AUTO-mode required: --model PATH --L1-size N --L2-size N

Common opts:
  --host/--port              server location (default localhost:30000)
  --tp N                     (default 8) for AUTO-mode launch
  --docker NAME              container tag for results dir naming
  --kv-bytes-per-token N     (default 34*1024 for DSR1-0528 MLA fp8)
  --buffer-gb N              (default 12) per-rank HBM headroom in AUTO
  --cache-mode MODE          (default L3_file) for AUTO-mode launch
  --rounds-profile M         (default 2) rounds during which loggers run
  --num-rounds N             override auto-derived total rounds
  --num-clients N            (default 300)
  --hicache-file-dir DIR     L3 backend dir; auto-picked from
                             SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR env
                             (or first match under /tmp/cascade_dsr1_l3_*)
  --interval-ms N            PCIe / nvsmi poll period (default 200)
  --enable-strace            attach strace to sglang scheduler workers
                             during slice; high overhead, off by default
  --output-dir DIR           default ~/SGLang-benchmarks/profiles/<tag>_l3sys/<ts>
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[ -z "$TAG" ] && { echo "ERROR: --tag required" >&2; exit 1; }

# ============================== Mode arbitration ==============================
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
[ "$AUTO_MODE" = false ] && TOTAL_ROUNDS=$(( ROUNDS_WARMUP + ROUNDS_PROFILE ))

# ============================== Locate bench script ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py" \
         "$HOME/PR/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
[ -z "$BENCH_SCRIPT" ] && { echo "ERROR: bench_multiturn.py not found" >&2; exit 1; }

# ============================== Auto-find L3 dir ==============================
if [ -z "$HICACHE_FILE_DIR" ]; then
  # cascade_dsr1_lite.sh writes to /tmp/cascade_dsr1_l3_<TAG>_<SIZE>.
  # Pick the most recently mtime'd one (likely the active server's).
  HICACHE_FILE_DIR=$(ls -td /tmp/cascade_dsr1_l3_* 2>/dev/null | head -1 || true)
fi
if [ -z "$HICACHE_FILE_DIR" ] || [ ! -d "$HICACHE_FILE_DIR" ]; then
  echo "WARNING: no L3 backing dir found; iostat will sample ALL block devs" >&2
  L3_DEV=""
else
  # Translate dir → block device. df --output=source returns e.g.
  # /dev/nvme0n1p2; iostat wants the parent device name (nvme0n1).
  L3_SRC=$(df --output=source "$HICACHE_FILE_DIR" | tail -1)
  L3_BASE=$(basename "$L3_SRC")
  # Strip partition suffix: nvme0n1p2 -> nvme0n1; sda1 -> sda
  L3_DEV=$(echo "$L3_BASE" | sed -E 's/p?[0-9]+$//')
  echo ">>> L3 dir: $HICACHE_FILE_DIR  on /dev/$L3_BASE  (parent: /dev/$L3_DEV)"
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
echo ">>> platform: $VENDOR"

# ============================== Output dir ==============================
TS=$(date +%Y%m%d_%H%M%S)
[ -z "$OUTPUT_DIR" ] && OUTPUT_DIR="$HOME/SGLang-benchmarks/profiles/${TAG}_l3sys/${TS}"
mkdir -p "$OUTPUT_DIR"
echo ">>> output dir: $OUTPUT_DIR"

# ============================== Helpers ==============================
metric_baseline_count() {
  curl -s "http://${HOST}:${PORT}/metrics" \
    | awk '
        /^sglang:num_requests_total[^_]/        { f += $NF }
        /^sglang:num_aborted_requests_total/    { a += $NF }
        END { printf "%d\n", f + a }'
}

# ============================== AUTO mode: derive params + launch ==============================
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
CASCADE_PID=""
CASCADE_LOG=""
BENCH_PID=""

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
    --rounds-profile "$ROUNDS_PROFILE" 2>&1) || { echo "$PARAMS" >&2; exit 1; }
  echo "$PARAMS" | grep -E '^(WARN|ERROR)' >&2 || true
  eval "$(echo "$PARAMS" | grep -E '^[A-Z_]+=')"
  echo "    weights/rank=${WEIGHTS_GB_PER_RANK}GB  HBM/rank=${HBM_GB_PER_RANK}GB"
  echo "    KV/round/rank=${KV_GB_PER_ROUND_PER_RANK}GB  → WARMUP=${WARMUP_ROUNDS} rounds"
  echo "    derived: mem-fraction-static=${MEM_FRACTION_STATIC}  num-rounds=${NUM_ROUNDS}"
  echo "    profile target: round ${PROFILE_TARGET_ROUND_1IDX} (1-indexed)"
  ROUNDS_WARMUP="$WARMUP_ROUNDS"
  [ -n "$NUM_ROUNDS_OVERRIDE" ] && NUM_ROUNDS="$NUM_ROUNDS_OVERRIDE"
  TOTAL_ROUNDS="$NUM_ROUNDS"
  echo "$PARAMS" > "$OUTPUT_DIR/profile_params.txt"

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

  echo ">>> waiting up to ${WAIT_FOR_HEALTH_SEC}s for /health=200..."
  deadline=$(( $(date +%s) + WAIT_FOR_HEALTH_SEC ))
  while true; do
    if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
      echo ">>> server up"; break
    fi
    if ! kill -0 "$CASCADE_PID" 2>/dev/null; then
      echo "ERROR: cascade_dsr1_lite.sh died before server came up; tail of $CASCADE_LOG:" >&2
      tail -n 40 "$CASCADE_LOG" >&2 || true
      exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "ERROR: /health timeout" >&2; exit 1
    fi
    sleep 5
  done

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
      echo "ERROR: bench start timeout" >&2; exit 1
    fi
    sleep 2
  done
  echo ">>> bench_multiturn started"

  BASELINE_DONE=$(metric_baseline_count)
  TARGET_FOR_PROFILE=$(( BASELINE_DONE + ROUNDS_WARMUP * NUM_CLIENTS ))
  echo ">>> baseline num_requests_done=${BASELINE_DONE}; loggers start at ${TARGET_FOR_PROFILE}"

else
  # ============================== MANUAL mode pre-flight ==============================
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; then
    echo "ERROR: $HOST:$PORT /health not 200; start server first" >&2
    exit 1
  fi
  if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/metrics" 2>/dev/null)" != "200" ]; then
    echo "ERROR: /metrics not reachable; relaunch with --enable-metrics" >&2
    exit 1
  fi

  echo ">>> flushing /flush_cache + drop_caches"
  curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
  sleep 2; sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  sleep 2

  BASELINE_DONE=$(metric_baseline_count)
  TARGET_FOR_PROFILE=$(( BASELINE_DONE + ROUNDS_WARMUP * NUM_CLIENTS ))
  echo ">>> baseline num_requests_done=${BASELINE_DONE}; loggers start at ${TARGET_FOR_PROFILE}"

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
    --tag "${TAG}-warmup${ROUNDS_WARMUP}-l3sys" \
    --disable-random-sample \
    --disable-auto-run \
    --enable-round-barrier \
    > "$BENCH_LOG" 2>&1 &
  BENCH_PID=$!
fi

LOGGER_PIDS=()
ROCPROF_TMPDIR=""

cleanup() {
  for pid in "${LOGGER_PIDS[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  if [ -n "$ROCPROF_TMPDIR" ] && [ -d "$ROCPROF_TMPDIR" ]; then
    pkill -P $$ rocprofv3 2>/dev/null || true
  fi
  for pid in "$BENCH_PID" "$CASCADE_PID"; do
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo ">>> killing pid=$pid"
      kill "$pid" 2>/dev/null || true
      wait "$pid" 2>/dev/null || true
    fi
  done
  [ "$AUTO_MODE" = true ] && pkill -9 sglang 2>/dev/null || true
}
trap cleanup EXIT

# ============================== Wait until warmup done ==============================
echo ">>> waiting for warmup to finish (${ROUNDS_WARMUP} rounds)"
LAST_LOG=0
DRIVER_PID="$BENCH_PID"
[ "$AUTO_MODE" = true ] && DRIVER_PID="$CASCADE_PID"
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  done_now=$(metric_baseline_count)
  reqs_done=$(( done_now - BASELINE_DONE ))
  if [ "$done_now" -ge "$TARGET_FOR_PROFILE" ]; then
    echo ">>> warmup complete: reqs_done=${reqs_done}"
    break
  fi
  now=$(date +%s)
  if [ $(( now - LAST_LOG )) -ge 5 ]; then
    echo "    progress: ${reqs_done}/$(( ROUNDS_WARMUP * NUM_CLIENTS )) warmup reqs"
    LAST_LOG=$now
  fi
  sleep 1
done
if ! kill -0 "$BENCH_PID" 2>/dev/null; then
  echo "ERROR: bench died before warmup; tail of $BENCH_LOG:" >&2
  tail -n 40 "$BENCH_LOG" >&2 || true
  exit 1
fi

# ============================== Start loggers ==============================
SLICE_START=$(date +%s.%N)
echo ">>> starting L3 system loggers @ $(date -Iseconds)"
echo "$SLICE_START" > "$OUTPUT_DIR/slice_start.epoch"

# 1) iostat — universal disk-layer bandwidth, latency, queue depth.
#    -x: extended stats; -m: MB/s; -t: timestamp each block.
IOSTAT_INTERVAL=1
IOSTAT_LOG="$OUTPUT_DIR/iostat.log"
if [ -n "$L3_DEV" ]; then
  iostat -xmt $IOSTAT_INTERVAL "$L3_DEV" > "$IOSTAT_LOG" 2>&1 &
else
  iostat -xmt $IOSTAT_INTERVAL > "$IOSTAT_LOG" 2>&1 &
fi
LOGGER_PIDS+=($!)
echo "    iostat -> $IOSTAT_LOG (pid=${LOGGER_PIDS[-1]})"

# 2) Vendor PCIe / GPU bandwidth sampler.
case "$VENDOR" in
  nvidia)
    NVSMI_LOG="$OUTPUT_DIR/nvsmi_pcie.csv"
    PCIE_LOG="$OUTPUT_DIR/pcie_throughput.csv"
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,pcie.link.gen.current,pcie.link.width.current \
      --format=csv -lms "$SAMPLE_INTERVAL_MS" > "$NVSMI_LOG" 2>&1 &
    LOGGER_PIDS+=($!)
    nvidia-smi --query-gpu=timestamp,index,pcie.tx.bytes,pcie.rx.bytes,nvlink.tx.bytes,nvlink.rx.bytes \
      --format=csv -lms "$SAMPLE_INTERVAL_MS" > "$PCIE_LOG" 2>&1 &
    LOGGER_PIDS+=($!)
    echo "    nvidia-smi gpu/pcie -> $NVSMI_LOG / $PCIE_LOG"
    ;;
  amd)
    # rocm-smi has no real-time PCIe Rx counter. Fallback options:
    #   (a) rocm-smi --showbw          (NVLink-equivalent xGMI BW only)
    #   (b) rocprofv3 hip-trace slice  (capture HIP memcpy events)
    # We do both: (a) as a poll, (b) for the slice duration.
    ROCMSMI_LOG="$OUTPUT_DIR/rocmsmi.csv"
    (
      while sleep "$(awk -v ms="$SAMPLE_INTERVAL_MS" 'BEGIN{printf "%.3f", ms/1000}')"; do
        ts=$(date -Iseconds)
        rocm-smi --showuse --showmemuse --csv 2>/dev/null \
          | awk -v ts="$ts" 'NR==1{print "ts,"$0; next} NF{print ts","$0}'
      done
    ) > "$ROCMSMI_LOG" 2>&1 &
    LOGGER_PIDS+=($!)
    echo "    rocm-smi -> $ROCMSMI_LOG"

    if command -v rocprofv3 >/dev/null 2>&1; then
      ROCPROF_TMPDIR="$OUTPUT_DIR/rocprof"
      mkdir -p "$ROCPROF_TMPDIR"
      # Attach to the running scheduler tp_rank=0 process. Heuristic:
      # the parent SGLang process advertises 'launch_server' in its
      # cmdline; we attach to its first child (typically tp_rank=0).
      SGL_PID=$(pgrep -f 'sglang.launch_server' | head -1 || true)
      if [ -n "$SGL_PID" ]; then
        echo "    rocprofv3 attach to pid=$SGL_PID -> $ROCPROF_TMPDIR"
        rocprofv3 --hip-trace --kernel-trace \
          --output-format csv \
          -d "$ROCPROF_TMPDIR" \
          --runtime-trace --hip-runtime-trace \
          -p "$SGL_PID" 2>&1 > "$OUTPUT_DIR/rocprof.log" &
        LOGGER_PIDS+=($!)
      else
        echo "    WARN: no sglang process to attach rocprofv3 to" >&2
      fi
    else
      echo "    WARN: rocprofv3 not on PATH; skipping HIP trace" >&2
    fi
    ;;
  *)
    echo "    WARN: no GPU vendor tool; PCIe sampling skipped" >&2
    ;;
esac

# 3) Optional strace (high overhead, default off).
if [ "$ENABLE_STRACE" = "true" ]; then
  SGL_PID=$(pgrep -f 'sglang.launch_server' | head -1 || true)
  if [ -n "$SGL_PID" ]; then
    STRACE_LOG="$OUTPUT_DIR/strace.log"
    strace -p "$SGL_PID" -f -ttt -T -e trace=pread64,pwrite64,read,write,io_uring_enter,io_uring_setup \
      -o "$STRACE_LOG" 2>&1 &
    LOGGER_PIDS+=($!)
    echo "    strace pid=$SGL_PID -> $STRACE_LOG"
  else
    echo "    WARN: ENABLE_STRACE=true but no sglang process found" >&2
  fi
fi

# ============================== Wait for profile slice ==============================
echo ">>> loggers running; waiting for ${ROUNDS_PROFILE} profile rounds to complete"
TARGET_FOR_DONE=$(( BASELINE_DONE + TOTAL_ROUNDS * NUM_CLIENTS ))
while kill -0 "$DRIVER_PID" 2>/dev/null; do
  done_now=$(metric_baseline_count)
  if [ "$done_now" -ge "$TARGET_FOR_DONE" ]; then
    break
  fi
  sleep 1
done

SLICE_END=$(date +%s.%N)
echo "$SLICE_END" > "$OUTPUT_DIR/slice_end.epoch"
SLICE_DUR=$(awk -v a="$SLICE_START" -v b="$SLICE_END" 'BEGIN{printf "%.2f", b-a}')
echo ">>> slice complete (${SLICE_DUR}s); stopping loggers"

# Stop loggers cleanly so files flush.
for pid in "${LOGGER_PIDS[@]}"; do
  kill "$pid" 2>/dev/null || true
done
wait "${LOGGER_PIDS[@]}" 2>/dev/null || true
LOGGER_PIDS=()

wait "$DRIVER_PID" 2>/dev/null || true
trap - EXIT
[ "$AUTO_MODE" = true ] && pkill -9 sglang 2>/dev/null && sleep 5 || true

# ============================== Summary ==============================
echo ""
echo ">>> done. artifacts in: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"

cat <<EOF

How to read the artifacts
=========================

iostat.log  (universal)
  - 'rMB/s' on /dev/$L3_DEV is L3 read bandwidth.
    DSR1-0528 fp8 KV at page_size=64 ≈ 1.1 MB/page; if rMB/s ≪ disk
    spec, L3 file backend is NOT disk-bound → the issue is upstream
    (memcpy / scheduler).
  - '%util' = how busy the device is. >90% on the slice = disk-bound.
  - 'await'   = avg I/O completion latency in ms. NVMe should be <1ms;
    >5ms means queue is stacking.

EOF

case "$VENDOR" in
  nvidia)
    cat <<EOF
nvsmi_pcie.csv / pcie_throughput.csv  (NVIDIA)
  - pcie.rx.bytes is HOST -> device (this is the L2->L1 transfer);
    Δrx_bytes / Δt = H2D PCIe BW. B200 PCIe Gen5 x16 ≈ 64 GB/s peak.
  - If iostat says disk has slack but rx_bytes is ~peak → PCIe-bound,
    not disk-bound.
  - utilization.gpu near 100% during the slice = compute-bound, L3
    transfer is being hidden by attention; further L3 speedups won't
    help TTFT.
EOF
    ;;
  amd)
    cat <<EOF
rocmsmi.csv  (AMD)
  - GPU utilization & memory use only; AMD has no real-time PCIe BW
    counter exposed via rocm-smi.

rocprof/  (AMD, if rocprofv3 was found)
  - Open the agent_*.csv (or hip_api_trace_*.csv) files. Look for
    hipMemcpyAsync H2D rows.
  - Sum (bytes / duration) for hipMemcpyAsync H2D rows in the slice
    window → effective PCIe H2D BW. MI355X PCIe Gen5 x16 same ≈64 GB/s.
  - kernel_trace will show attention / MLA kernels overlapped with
    those memcpies. 'wall_duration' vs 'gpu_duration' tells you whether
    the host stage was hidden under the GPU work.

If rocprofv3 wasn't available: install rocm-developer-tools or
rocprofiler-systems, or fall back to ROCProfiler-Compute (omniperf).
EOF
    ;;
esac

cat <<EOF
Decision tree
=============
  iostat %util > 90 ?
    yes → disk-bound. Move L3 to faster storage (NVMe Gen5, 3FS, RDMA).
    no  → check H2D BW:
            near vendor PCIe peak ?
              yes → PCIe-bound. Enable zero-copy backend (see notes),
                    or shrink page_size to overlap better.
              no  → upstream (scheduler / Python loop / kernel launch);
                    rerun cascade_L2_torchprofile.sh and check
                    hiradix_cache.py:prefetch_from_storage frame time.

EOF
