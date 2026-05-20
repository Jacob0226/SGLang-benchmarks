#!/usr/bin/env bash
# cascade_NoCache_torchprofile.sh — baseline prefill profile with NO prefix cache.
#
# Companion to cascade_L2_torchprofile.sh + cascade_L3_sysprof.sh:
#   - L2 profile : measures HiCache L1→L2 evict / load_back / prefetch overhead
#   - L3 profile : measures L3 file backend disk + PCIe + syscall overhead
#   - NoCache    : measures pure cold-miss prefill (this script) — the
#                  baseline that L2 / L3 overhead is added on top of
#
# Workflow (mirrors GLM.sh):
#   1. compute_profile_params.py → mem-fraction-static (same as cascade runs,
#      so KV pool size + cuda graph capture state are directly comparable)
#   2. start server with --disable-radix-cache (no prefix cache at all)
#   3. warmup (8 random prompts to stabilize scheduler + capture cuda graph)
#   4. GSM8K precheck (correctness anchor; default mode only)
#   5. for n in 1..NUM_INPUT_LENS:
#          input_len = REQUEST_LENGTH × n
#          sglang.bench_serving --random-input $input_len --random-output 1
#                               --max-concurrency $NUM_CLIENTS
#                               --num-prompt $((NUM_CLIENTS × PROMPT_MULTIPLIER))
#                               --profile --profile-num-steps 5 [--profile-by-stage]
#                               --profile-prefix in${input_len}_out1_conc${NUM_CLIENTS}_p${num_prompts}
#   6. pkill sglang; sleep
#   7. repeat steps 2-6 with --disable-cuda-graph (no-cuda-graph mode)
#
# Each input_len corresponds to a HiCache cascade round's prompt size:
#   4096  = round 1's prompt (initial)
#   8192  = round 2's prompt (round 1 + 1 turn history)
#   ...
#   4096*N = round N's prompt
#
# So prof_in${L*N}_out1_* is the no-cache equivalent of HiCache cascade
# round N's prefill — same prompt length, same effective batch (capped by
# chunked-prefill-size), but every request is a cold miss instead of
# partial prefix hit.
#
# Comparison with cascade_L2_torchprofile.sh trace at round N gives the
# decomposition:
#   total_prefill_cost = no_cache_prefill_cost + L1_to_L2_overhead
# and similarly for L3.

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TAG=""
DOCKER="untagged-docker"
TP_SIZE=8
HOST="localhost"
PORT=30000

# Workload: match cascade FairCompare_Small_v1 by default. NUM_CLIENTS is
# the effective server-side concurrent batch (NOT --max-parallel, which is
# a misleading client-side semaphore — server actually sees ≈ NUM_CLIENTS
# concurrent due to how bench_multiturn drains the per-round backlog).
NUM_CLIENTS=100
REQUEST_LENGTH=4096           # base input length (= 1 round of HiCache prompt)
NUM_INPUT_LENS=8              # sweep 1L, 2L, ..., NL
OUTPUT_LENGTH=1               # prefill profile, decode is single token
PROMPT_MULTIPLIER=2           # num_prompt = NUM_CLIENTS × this
                              # (2 rounds worth → enough for 5 profile steps)

# Sizing: still passed even though cache is off, because mem_fraction_static
# affects KV pool size which affects max_running_requests / cuda graph
# capture. Use SAME values as the cascade_L{2,3} profiles for fair compare.
L1_SIZE=20
L2_SIZE=50

# Server constants (must match cascade_dsr1_lite.sh).
PAGE_SIZE=64
CONTEXT_LENGTH=65536
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
KV_CACHE_DTYPE=fp8_e4m3
ATTENTION_BACKEND=""          # auto-detect by vendor

# Correctness anchor.
GSM8K_NUM_QUESTIONS=1200
GSM8K_PARALLEL=1200
GSM8K_PRECHECK="true"

# Profiler knobs.
WAIT_FOR_SERVER_SEC=1500
NUM_PROFILE_STEPS=5
PROF_COMBINED="false"         # true = single combined trace; false = --profile-by-stage

ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case $1 in
    --tag)                 TAG="$2"; shift 2;;
    --docker)              DOCKER="$2"; shift 2;;
    --model)               MODEL_PATH="$2"; shift 2;;
    --tp)                  TP_SIZE="$2"; shift 2;;
    --host)                HOST="$2"; shift 2;;
    --port)                PORT="$2"; shift 2;;
    --num-clients)         NUM_CLIENTS="$2"; shift 2;;
    --request-length)      REQUEST_LENGTH="$2"; shift 2;;
    --num-input-lens)      NUM_INPUT_LENS="$2"; shift 2;;
    --prompt-multiplier)   PROMPT_MULTIPLIER="$2"; shift 2;;
    --L1-size)             L1_SIZE="$2"; shift 2;;
    --L2-size)             L2_SIZE="$2"; shift 2;;
    --page-size)           PAGE_SIZE="$2"; shift 2;;
    --context-length)      CONTEXT_LENGTH="$2"; shift 2;;
    --chunked-prefill-size) CHUNKED_PREFILL_SIZE="$2"; shift 2;;
    --max-prefill-tokens)  MAX_PREFILL_TOKENS="$2"; shift 2;;
    --attention-backend)   ATTENTION_BACKEND="$2"; shift 2;;
    --gsm8k-num-questions) GSM8K_NUM_QUESTIONS="$2"; shift 2;;
    --no-gsm8k-precheck)   GSM8K_PRECHECK="false"; shift 1;;
    --prof-combined)       PROF_COMBINED="true"; shift 1;;
    --num-profile-steps)   NUM_PROFILE_STEPS="$2"; shift 2;;
    --wait-for-server-sec) WAIT_FOR_SERVER_SEC="$2"; shift 2;;
    -h|--help)
      cat <<EOF
Usage: $0 --tag TAG --docker DOCKER [opts]

Required:
  --tag TAG                  identifier for the result subdir
  --docker NAME              container tag for results dir naming

Common opts (defaults match cascade FairCompare_Small_v1 setup):
  --model PATH               default: DeepSeek-R1-0528 path under /data
  --tp N                     (default 8)
  --num-clients N            (default 100) effective server concurrency
  --request-length L         (default 4096) base prompt length (= 1 HiCache round)
  --num-input-lens N         (default 8) sweep input_len = L*1, L*2, ..., L*N
  --prompt-multiplier M      (default 2) num-prompt = NUM_CLIENTS * M
  --output-length             fixed at 1 (prefill profile)
  --L1-size N                (default 20) only used to derive mem-fraction-static
  --L2-size N                (default 50) same
  --num-profile-steps K      (default 5) torch.profiler steps per input_len
  --prof-combined            single combined trace (default: --profile-by-stage)
  --no-gsm8k-precheck        skip GSM8K correctness check
  --gsm8k-num-questions N    (default 1200)

Output:
  ~/SGLang-benchmarks/results/<docker>/<model>-NoCache-<tag>/
    ├── server.log, warmup.log, Accuracy_GSM8K.{log,jsonl}, cmdline.txt
    ├── in<L>_out1_conc<N>_p<NP>-*.trace.json.gz   (per input_len)
    └── no-cuda-graph/
        └── (same artifacts, with --disable-cuda-graph)

Each in<input_len>_*.trace.json.gz is the no-cache equivalent of cascade
round N's prefill at prompt length = REQUEST_LENGTH * N.
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[ -z "$TAG" ] && { echo "ERROR: --tag required" >&2; exit 1; }

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

# Auto-pick attention backend if not specified.
if [ -z "$ATTENTION_BACKEND" ]; then
  case "$VENDOR" in
    nvidia) ATTENTION_BACKEND="trtllm_mla";;
    amd)    ATTENTION_BACKEND="aiter";;
    *)      ATTENTION_BACKEND="aiter";;
  esac
fi

# ============================== Output dir ==============================
MODEL_NAME=$(basename "${MODEL_PATH%/}")
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-NoCache-${TAG}"
mkdir -p "$BASE_LOG_DIR"
echo ">>> base log dir: $BASE_LOG_DIR"

# Save invocation for postmortem (matches the convention in cascade_dsr1_lite.sh).
{
  printf '%s' "$0"
  for a in "${ORIG_ARGS[@]}"; do
    printf ' %q' "$a"
  done
  printf '\n'
} > "$BASE_LOG_DIR/cmdline.txt"

# ============================== Derive mem_fraction_static ==============================
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
HELPER="$SCRIPT_DIR/compute_profile_params.py"
[ -f "$HELPER" ] || { echo "ERROR: $HELPER not found" >&2; exit 1; }

echo ">>> deriving mem_fraction_static via $HELPER"
PARAMS=$(python3 "$HELPER" \
  --model "$MODEL_PATH" \
  --tp "$TP_SIZE" \
  --L1-size "$L1_SIZE" \
  --L2-size "$L2_SIZE" \
  --num-clients "$NUM_CLIENTS" \
  --request-length "$REQUEST_LENGTH" 2>&1) || { echo "$PARAMS" >&2; exit 1; }
echo "$PARAMS" | grep -E '^(WARN|ERROR|INFO)' >&2 || true
# Whitelist: only the vars we actually consume. Helper also emits NUM_ROUNDS=
# and others that would silently shadow our CLI values.
ALLOW='^(MEM_FRACTION_STATIC|WEIGHTS_GB_PER_RANK|HBM_GB_PER_RANK|USABLE_HBM_GB_PER_RANK|FRAMEWORK_RESERVE_GB)='
eval "$(echo "$PARAMS" | grep -E "$ALLOW")"
echo "    weights=${WEIGHTS_GB_PER_RANK}GB  HBM=${HBM_GB_PER_RANK}GB  mem_fraction_static=${MEM_FRACTION_STATIC}"

# ============================== Env vars (vendor-specific) ==============================
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1
if [ "$VENDOR" = "amd" ]; then
  # Match cascade_dsr1_lite.sh env on AMD: disable aiter FP8 prefill kernel
  # (PR #18528) + force aiter path + INT4 quick-reduce off (we want
  # bf16 allreduce here, NOT INT4, for cleaner baseline).
  export SGLANG_USE_AITER=1
  export ROCM_QUICK_REDUCE_QUANTIZATION=NONE
  export SGLANG_AITER_FP8_PREFILL_ATTN=0
fi

# ============================== Helpers ==============================
trap 'pkill -9 sglang 2>/dev/null || true; sleep 5' EXIT

start_server() {
  local logfile="$1"
  shift
  local extra_args=("$@")

  echo ">>> starting SGLang server (NoCache: --disable-radix-cache); extra: ${extra_args[*]:-none}"
  local cmd=(
    python3 -u -m sglang.launch_server
      --model-path "$MODEL_PATH"
      --tp "$TP_SIZE"
      --host "$HOST" --port "$PORT"
      --mem-fraction-static "$MEM_FRACTION_STATIC"
      --watchdog-timeout 2400
      --enable-metrics
      --trust-remote-code
      --kv-cache-dtype "$KV_CACHE_DTYPE"
      --page-size "$PAGE_SIZE"
      --context-length "$CONTEXT_LENGTH"
      --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
      --max-prefill-tokens "$MAX_PREFILL_TOKENS"
      --attention-backend "$ATTENTION_BACKEND"
      --disable-radix-cache
  )
  if [ "$VENDOR" = "nvidia" ]; then
    cmd+=(
      --moe-runner-backend flashinfer_trtllm
      --enable-flashinfer-allreduce-fusion
    )
  fi
  cmd+=("${extra_args[@]}")

  {
    echo ">>> server cmd:"
    printf '%q ' "${cmd[@]}"
    echo
    echo "---"
  } > "$logfile"

  "${cmd[@]}" >> "$logfile" 2>&1 &
  SERVER_BG_PID=$!

  echo ">>> wait /health up to ${WAIT_FOR_SERVER_SEC}s..."
  local deadline=$(( $(date +%s) + WAIT_FOR_SERVER_SEC ))
  while [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; do
    if ! kill -0 "$SERVER_BG_PID" 2>/dev/null; then
      echo "ERROR: server died; tail of $logfile:" >&2
      tail -n 50 "$logfile" >&2
      exit 1
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "ERROR: /health timeout" >&2
      exit 1
    fi
    sleep 5
  done
  echo ">>> server up (pid=$SERVER_BG_PID)"
}

warmup() {
  local log_dir="$1"
  local logfile="$log_dir/warmup.log"
  echo ">>> warmup"
  python3 -m sglang.bench_serving \
    --backend sglang --host "$HOST" --port "$PORT" \
    --model "$MODEL_PATH" --dataset-name random \
    --random-input 1024 --random-output 128 --random-range-ratio 1.0 \
    --max-concurrency 4 --num-prompt 8 --output-file /dev/null \
    2>&1 | tee "$logfile" >/dev/null
}

gsm8k_precheck() {
  local log_dir="$1"
  if [ "$GSM8K_PRECHECK" != "true" ]; then
    echo ">>> GSM8K precheck skipped"
    return
  fi
  local gsm_script=""
  for c in /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
           "$HOME/work-space/sglang/benchmark/gsm8k/bench_sglang.py"; do
    [ -f "$c" ] && gsm_script="$c" && break
  done
  if [ -z "$gsm_script" ]; then
    echo ">>> WARN: bench_sglang.py not found, skipping GSM8K"
    return
  fi
  local logfile="$log_dir/Accuracy_GSM8K.log"
  echo ">>> GSM8K precheck (${GSM8K_NUM_QUESTIONS} q, parallel=${GSM8K_PARALLEL})"
  ( cd "$log_dir" && python3 "$gsm_script" \
      --host "$HOST" --port "$PORT" \
      --num-questions "$GSM8K_NUM_QUESTIONS" \
      --parallel "$GSM8K_PARALLEL" \
      --result-file "$log_dir/Accuracy_GSM8K.jsonl" \
      2>&1 | tee "$logfile" >/dev/null ) || true
  local acc
  acc=$(grep -oP '^Accuracy:\s+\K[0-9.]+' "$logfile" | tail -1 || true)
  echo "    GSM8K accuracy: ${acc:-NA}"
}

profile_one() {
  local log_dir="$1"
  local input_len="$2"
  local num_prompts=$((NUM_CLIENTS * PROMPT_MULTIPLIER))
  local stem="in${input_len}_out${OUTPUT_LENGTH}_conc${NUM_CLIENTS}_p${num_prompts}"
  local logfile="$log_dir/prof_${stem}.log"

  # Idempotency: skip if a trace with this stem already exists.
  if compgen -G "$log_dir/${stem}-*.trace.json.gz" >/dev/null; then
    echo "    skip: $stem trace already exists"
    return
  fi

  export SGLANG_TORCH_PROFILER_DIR="$log_dir"

  local prof_args=(
    --profile
    --profile-num-steps "$NUM_PROFILE_STEPS"
    --profile-prefix "$stem"
  )
  if [ "$PROF_COMBINED" != "true" ]; then
    prof_args+=(--profile-by-stage)
  fi

  echo ">>> profile in=${input_len} conc=${NUM_CLIENTS} prompts=${num_prompts}"
  python3 -m sglang.bench_serving \
    --backend sglang --host "$HOST" --port "$PORT" \
    --model "$MODEL_PATH" --dataset-name random \
    --random-input "$input_len" \
    --random-output "$OUTPUT_LENGTH" \
    --random-range-ratio 1.0 \
    --max-concurrency "$NUM_CLIENTS" \
    --num-prompt "$num_prompts" \
    --output-file /dev/null \
    "${prof_args[@]}" \
    2>&1 | tee "$logfile" >/dev/null
}

# ============================== Main loop ==============================
PROF_MODES=("default" "no-cuda-graph")

for PROF_MODE in "${PROF_MODES[@]}"; do
  if [ "$PROF_MODE" = "no-cuda-graph" ]; then
    LOG_DIR="$BASE_LOG_DIR/no-cuda-graph"
    EXTRA_SERVER_ARGS=(--disable-cuda-graph)
  else
    LOG_DIR="$BASE_LOG_DIR"
    EXTRA_SERVER_ARGS=()
  fi
  mkdir -p "$LOG_DIR"
  SERVER_LOG="$LOG_DIR/server.log"

  echo ""
  echo ">>> ============================================================"
  echo ">>> profiling mode: $PROF_MODE   (t=$(date +%H:%M:%S))"
  echo ">>> ============================================================"

  start_server "$SERVER_LOG" "${EXTRA_SERVER_ARGS[@]}"
  warmup "$LOG_DIR"

  if [ "$PROF_MODE" = "default" ]; then
    gsm8k_precheck "$LOG_DIR"
  fi

  # Sweep input lengths: REQUEST_LENGTH × 1, ×2, ..., × NUM_INPUT_LENS.
  for ((n = 1; n <= NUM_INPUT_LENS; n++)); do
    input_len=$((REQUEST_LENGTH * n))
    if [ "$input_len" -gt "$CONTEXT_LENGTH" ]; then
      echo ">>> skip in=${input_len} (> --context-length ${CONTEXT_LENGTH})"
      continue
    fi
    if [ "$input_len" -gt "$MAX_PREFILL_TOKENS" ]; then
      # Each request still has to fit in one chunked-prefill window; SGLang
      # will OOM the request scheduler if input > max-prefill-tokens.
      echo ">>> skip in=${input_len} (> --max-prefill-tokens ${MAX_PREFILL_TOKENS})"
      continue
    fi
    profile_one "$LOG_DIR" "$input_len"
  done

  echo ">>> stopping server"
  pkill -9 sglang 2>/dev/null || true
  sleep 10
done

trap - EXIT

echo ""
echo ">>> done. artifacts in: $BASE_LOG_DIR"
echo ">>> traces:"
find "$BASE_LOG_DIR" -maxdepth 3 -name "*.trace.json.gz" -printf '    %P\n' 2>/dev/null | sort
