#!/usr/bin/env bash
# GLM-5.3-Flash NVFP4 SPEED-Bench golden AL collector for B200 / SGLang.
#
# InferenceX has no golden AL curve for GLM-5.3-Flash, so --golden-al in
# ix_agentx_glm53flash.sh has nothing official to point at. This measures one
# the way upstream's benchmarks/single_node/speedbench/ collectors do:
# SPEED-Bench qualitative, one category (coding), temperature 1.0, top_p 0.95,
# output_len 4096, EOS honoured, real verification, one server boot per MTP
# level, AL = SGLang's avg_spec_accept_length (bonus token included). Server
# flags are the B200 AgentX recipe's low-concurrency ones, minus HiCache,
# which only matters for prefix reuse.
#
# GLM-5.3-Flash's chat template always opens the generation prompt with
# <think>, so the default template is thinking_on and no kwargs are passed.
#
# The verify method changes AL by ~0.4 on this model, so it is a curve axis,
# like Kimi DSpark's per-sampler curves upstream:
#   --verify target   SGLang's CUDA default (draft as a point mass)
#   --verify rs       --speculative-use-rejection-sampling
#
# Run INSIDE the B200 container. Needs SPEED-Bench's qualitative.jsonl in
# --speedbench-dir. NeMo Skills' speed-bench/prepare.py --config qualitative
# stops at the gated cais/hle, so prepare the coding rows alone with
#   python3 speedbench_prepare_category.py --prepare <NeMo prepare.py> \
#       --category coding --output_dir /raid/home/jacchang/speedbench
#
# Results land in golden_al/glm5.3flash_mtp.yaml's shape; the driver simulates
# that file (target) by default.
#
#   ./speedbench_glm53flash_b200_sglang.sh --verify target --gpus 0,1,2,3 --port 29100
#   ./speedbench_glm53flash_b200_sglang.sh --verify rs --gpus 4,5,6,7 --port 29200 --mtp-list "3 4 5"
set -uo pipefail

usage() { sed -n '2,31p' "$0"; exit 1; }

CKPT=/raid/home/jacchang/models/nvidia/GLM-5.3-Flash-NVFP4
MODEL_ID=nvidia/GLM-5.3-Flash-NVFP4
SPEEDBENCH_DIR=/raid/home/jacchang/speedbench
OUT_ROOT=""
VERIFY=""
GPUS=""
PORT=29100
MTP_LIST="3 4 5"
CATEGORY=coding
OUTPUT_LEN=4096
# AL is a per-draft accept/reject property, so batch the pass like upstream's
# SGLang collector does rather than running it at conc 1.
CONCURRENCY=32
TEMPERATURE=1.0
TOP_P=0.95

while [[ $# -gt 0 ]]; do
    case $1 in
        --verify)     VERIFY="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --port)       PORT="$2"; shift 2 ;;
        --mtp-list)   MTP_LIST="$2"; shift 2 ;;
        --ckpt)       CKPT="$2"; shift 2 ;;
        --speedbench-dir) SPEEDBENCH_DIR="$2"; shift 2 ;;
        --out)        OUT_ROOT="$2"; shift 2 ;;
        --conc)       CONCURRENCY="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

case "$VERIFY" in
    target) VERIFY_ARGS=() ;;
    rs)     VERIFY_ARGS=(--speculative-use-rejection-sampling) ;;
    *) echo "ERROR: --verify must be 'target' or 'rs'" >&2; exit 1 ;;
esac
[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
[ -d "$CKPT" ] || { echo "ERROR: checkpoint not found at $CKPT" >&2; exit 1; }
: "${OUT_ROOT:=$SPEEDBENCH_DIR/results/$VERIFY}"
mkdir -p "$OUT_ROOT"
LOG="$OUT_ROOT/collector.log"

# SGLang's speed-bench loader reads prompts from "turns" (first user turn) and
# filters on "category"; prepare.py writes "messages".
CATEGORY_JSONL="$SPEEDBENCH_DIR/qualitative_${CATEGORY}.jsonl"
if [ ! -s "$CATEGORY_JSONL" ]; then
    python3 - "$SPEEDBENCH_DIR/qualitative.jsonl" "$CATEGORY_JSONL" "$CATEGORY" <<'EOF' || exit 1
import json, sys
src, dst, category = sys.argv[1:]
with open(src) as f, open(dst + ".tmp", "w") as out:
    for line in f:
        row = json.loads(line)
        if row.get("category") == category:
            row["turns"] = [m["content"] for m in row["messages"] if m["role"] == "user"]
            out.write(json.dumps(row) + "\n")
import os
os.replace(dst + ".tmp", dst)
EOF
fi
NUM_PROMPTS=$(wc -l < "$CATEGORY_JSONL")
[ "$NUM_PROMPTS" -gt 0 ] || { echo "ERROR: no '$CATEGORY' rows in $CATEGORY_JSONL" >&2; exit 1; }

export PYTHONNOUSERSITE=1 TORCH_CUDA_ARCH_LIST=10.0

SERVER_PID=""
gpu_mem_used_max_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits ${GPUS:+-i "$GPUS"} | sort -rn | head -1
}
# The server runs under setsid, so its process group is its own and can be
# signalled without touching this script or a collector on the other GPUs.
stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    kill -TERM -- "-$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 2; done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null
    wait "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do
        [ "$(gpu_mem_used_max_mib)" -le 2048 ] && break
        sleep 3
    done
    SERVER_PID=""
}
trap stop_server EXIT

declare -A AL_RESULT

run_cell() {
    local mtp="$1"
    local cell="$OUT_ROOT/mtp${mtp}"
    mkdir -p "$cell"
    local result="$cell/bench.jsonl"
    if [ -s "$result" ]; then
        echo ">>> mtp=$mtp already measured, reusing $result" | tee -a "$LOG"
    else
        local max_running=$((2 * CONCURRENCY))
        local cmd=(
            python3 -m sglang.launch_server
            --model-path "$CKPT" --served-model-name "$MODEL_ID"
            --host 0.0.0.0 --port "$PORT" --trust-remote-code
            --tp 4 --ep-size 1
            --quantization modelopt_fp4
            --kv-cache-dtype fp8_e4m3
            --dsa-prefill-backend trtllm --dsa-decode-backend trtllm
            --moe-runner-backend flashinfer_trtllm
            --mamba-radix-cache-strategy extra_buffer
            --tool-call-parser glm47 --reasoning-parser glm45
            --chunked-prefill-size 8192
            --prefill-max-requests 1
            --mem-fraction-static 0.75
            --max-running-requests "$max_running"
            --cuda-graph-max-bs-decode "$max_running"
            --model-loader-extra-config '{"enable_multithread_load": true, "num_threads": 32}'
            --speculative-algorithm EAGLE
            --speculative-num-steps "$mtp"
            --speculative-eagle-topk 1
            --speculative-num-draft-tokens "$((mtp + 1))"
            "${VERIFY_ARGS[@]}"
            --watchdog-timeout 1800
            --enable-metrics
        )
        printf '%q ' "${cmd[@]}" > "$cell/sglang_command.txt"; echo >> "$cell/sglang_command.txt"
        env | grep -E '^SGLANG_SIMULATE_ACC_' && { echo "ERROR: SGLANG_SIMULATE_ACC_* is set; this must measure real AL" >&2; exit 1; }

        echo ">>> $(date -Is) mtp=$mtp verify=$VERIFY gpus=${GPUS:-all} port=$PORT: starting server" | tee -a "$LOG"
        setsid "${cmd[@]}" > "$cell/server.log" 2>&1 &
        SERVER_PID=$!
        local ok=0
        for _ in $(seq 1 360); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then ok=1; break; fi
            sleep 5
        done
        if [ "$ok" != 1 ]; then
            echo ">>> mtp=$mtp server failed to start, see $cell/server.log" | tee -a "$LOG"
            AL_RESULT[$mtp]=N/A
            stop_server
            return
        fi

        echo ">>> $(date -Is) mtp=$mtp: SPEED-Bench $CATEGORY, $NUM_PROMPTS prompts, conc $CONCURRENCY" | tee -a "$LOG"
        python3 -m sglang.bench_serving \
            --backend sglang --port "$PORT" --model "$CKPT" \
            --dataset-name speed-bench --dataset-path "$CATEGORY_JSONL" \
            --speed-bench-output-len "$OUTPUT_LEN" \
            --num-prompts "$NUM_PROMPTS" \
            --max-concurrency "$CONCURRENCY" \
            --temperature "$TEMPERATURE" --top-p "$TOP_P" \
            --disable-ignore-eos \
            --output-file "$result" --output-details \
            > "$cell/bench.log" 2>&1
        echo ">>> $(date -Is) mtp=$mtp: bench exit=$?" | tee -a "$LOG"
        stop_server
    fi

    local al
    al=$(python3 -c '
import json, sys
with open(sys.argv[1]) as f:
    r = json.loads(f.readlines()[-1])
al = r.get("accept_length")
done = r.get("completed") or 0
out = r.get("total_output_tokens") or 0
print(f"{al:.2f}" if al else "N/A")
print("  completed=%s mean_output_len=%.0f" % (done, out / max(done, 1)), file=sys.stderr)
' "$result" 2>>"$LOG" || echo N/A)
    echo ">>> mtp=$mtp verify=$VERIFY AL=$al" | tee -a "$LOG"
    AL_RESULT[$mtp]="$al"
}

for mtp in $MTP_LIST; do run_cell "$mtp"; done

YAML="$OUT_ROOT/glm5.3flash_mtp_${VERIFY}.yaml"
{
    echo "# LOCAL, not an InferenceX-committed curve. Acceptance Length (AL) measured with SPEED-Bench."
    echo "# dataset: $CATEGORY ($NUM_PROMPTS prompts) | temperature: $TEMPERATURE | top_p: $TOP_P | output_len: $OUTPUT_LEN | conc: $CONCURRENCY"
    echo "# thinking_on: chat template default (generation prompt opens <think>)"
    echo "# verify: $VERIFY $( [ "$VERIFY" = rs ] && echo '(--speculative-use-rejection-sampling)' || echo '(SGLang CUDA default, target-only)')"
    echo "# Measured on $MODEL_ID ($(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) TP4, SGLang $(python3 -c 'import sglang; print(sglang.__version__)'), EAGLE topk 1), $(date -I)."
    echo "# Generated by SGLang-benchmarks/Agentic/speedbench_glm53flash_b200_sglang.sh"
    echo "#"
    echo "# key = num_speculative_tokens (MTP steps); value = golden AL"
    echo "glm-5.3-flash:"
    echo "  thinking_on:"
    for mtp in $MTP_LIST; do echo "    $mtp: ${AL_RESULT[$mtp]:-N/A}"; done
} > "$YAML"
echo "=== $(date -Is) done: $YAML" | tee -a "$LOG"
cat "$YAML" | tee -a "$LOG"
