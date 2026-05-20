#!/usr/bin/env bash
# repro_pr25742_acc_matrix.sh
#
# Reproduce sglang issue #25742 GSM8K accuracy matrix on AMD MI355X.
#
# Launch logic mirrors the MI355X branch of GLM.sh's start_server() exactly
# (no --match-pr extras, no MTP, no lm-eval). Accuracy is measured with the
# same bench_sglang.py command as GLM.sh's accuracy_test().
#
# Run inside an MI355X sglang docker container, e.g.:
#   docker exec jacchang_GLM5_FP4 bash \
#       /home/jacchang/SGLang-benchmarks/repro_pr25742_acc_matrix.sh \
#       <TP> <log_dir>
#
# Args:
#   $1: tensor parallel size (e.g. 2 or 8)
#   $2: log directory (will be created)

set -euo pipefail
set -x

# Match GLM.sh line 19: bench_sglang.py opens 1200 concurrent sockets at
# --parallel 1200, which trips the default 1024 fd cap and produces
# "URLError: [Errno 24] Too many open files" for a handful of requests.
ulimit -n 65535 || true

TP="${1:?usage: $0 <tp> <log_dir>}"
LOG_DIR="${2:?usage: $0 <tp> <log_dir>}"

MODEL_PATH="/data/huggingface/hub/amd/GLM-5.1-MXFP4"
HOST="localhost"
PORT="8552"
TOKENIZER_WORKER_NUM=$((TP * 2))

mkdir -p "$LOG_DIR"
SERVER_LOG="$LOG_DIR/server.log"
GSM8K_LOG="$LOG_DIR/Accuracy_GSM8K.log"

# InferenceMax tuning carried over from GLM.sh (lines ~133-135).
export SAFETENSORS_FAST_GPU=1
export SGLANG_ROCM_FUSED_DECODE_MLA=0

# Older images (e.g. v0.5.10rc0-rocm720-mi35x-20260415) install aiter and
# sglang as editable packages and rely on /etc/bash.bashrc to set
# PYTHONPATH, which non-interactive `docker exec bash -c` does NOT source.
# Without these PYTHONPATH entries, Python treats /sgl-workspace/{aiter,sglang}
# as empty namespace packages and the real packages (one level deeper) are
# never found:
#   - aiter:   `from aiter import dynamic_per_tensor_quant` -> ImportError
#   - sglang:  `from sglang.test.test_utils import ...`     -> ModuleNotFoundError
# Setting PYTHONPATH explicitly fixes both. Harmless on newer images where
# the same paths exist and the editable finder works either way.
if [ -d /sgl-workspace/aiter/aiter ]; then
    export PYTHONPATH="/sgl-workspace/aiter:${PYTHONPATH:-}"
fi
if [ -d /sgl-workspace/sglang/python/sglang ]; then
    export PYTHONPATH="/sgl-workspace/sglang/python:${PYTHONPATH:-}"
fi

# Stop any leftover sglang processes first (specific match — won't kill
# this script or wrappers whose cmdline merely contains "sglang-rocm").
cleanup_sglang() {
    pkill -9 -f "sglang\.launch_server"   || true
    pkill -9 -f "sglang::scheduler"       || true
    pkill -9 -f "sglang::detokenizer"     || true
    pkill -9 -f "sglang::tokenizer"       || true
    pkill -9 -f "_inductor/compile_worker" || true
    sleep 3
}
cleanup_sglang

# ---- Launch (mirrors GLM.sh MI355X branch, no --match-pr / MTP) ----
python3 -m sglang.launch_server \
    --model "$MODEL_PATH" \
    --tp "$TP" \
    --host "$HOST" --port "$PORT" \
    --trust-remote-code \
    --tool-call-parser glm47 \
    --reasoning-parser glm45 \
    --watchdog-timeout 1200 \
    --mem-fraction-static 0.85 \
    --kv-cache-dtype fp8_e4m3 \
    --disable-radix-cache \
    --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 8}' \
    --nsa-prefill-backend tilelang \
    --nsa-decode-backend tilelang \
    --tokenizer-worker-num "$TOKENIZER_WORKER_NUM" \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo ">>> launched sglang server pid=$SERVER_PID  (TP=$TP, log=$SERVER_LOG)"

# ---- Wait for /health, polling every 5s, with progress every 30s ----
start_ts=$(date +%s)
last_progress=0
while true; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "http://$HOST:$PORT/health" || echo "000")
    if [ "$code" = "200" ]; then
        echo ">>> server healthy after $(( $(date +%s) - start_ts ))s"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server pid=$SERVER_PID died before /health=200" >&2
        tail -30 "$SERVER_LOG" >&2 || true
        exit 1
    fi
    now=$(date +%s)
    if (( now - last_progress >= 30 )); then
        echo "... waiting (elapsed $(( now - start_ts ))s, last code=$code)"
        last_progress=$now
    fi
    sleep 5
done

# ---- GSM8K (user's GLM.sh accuracy_test() command) ----
echo ">>> running GSM8K bench_sglang.py ..."
{
    echo ">>>Executing command:"
    echo "python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py --port $PORT --num-questions 1200 --parallel 1200"
    echo "---"
} > "$GSM8K_LOG"
python3 /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
    --port "$PORT" \
    --num-questions 1200 \
    --parallel 1200 2>&1 | tee -a "$GSM8K_LOG"

# ---- Cleanup ----
echo ">>> killing server ..."
kill -9 "$SERVER_PID" 2>/dev/null || true
cleanup_sglang
echo "DONE TP=$TP log=$LOG_DIR"
grep -E "Accuracy:|Invalid:|Latency:" "$GSM8K_LOG" || true
