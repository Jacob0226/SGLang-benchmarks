#!/usr/bin/env bash
# InferenceX AgentX sweep for GLM-5.3-Flash NVFP4 on B200 / SGLang, TP4.
#
# Runs recipe_glm53flash_fp4_b200_sglang_mtp.sh once per concurrency with the
# environment the run-sweep workflow would inject, so the client, the corpus,
# the result schema and the failure gates are InferenceX's rather than ours.
# The recipe itself is local: upstream has no GLM-5.3-Flash entry (its only
# GLM-5.3 config is the full model on MI355X TileRT), so the B200 SGLang shape
# is modelled on glm5.2-fp4-b200-sglang-agentic-mtp.
#
# Run this INSIDE the container (jacchang_GLM53-Flash-MTP,
# lmsysorg/sglang:v0.5.20-cu130). GPU selection comes from CUDA_VISIBLE_DEVICES
# or --gpus.
#
#   ./ix_agentx_glm53flash_b200.sh                      # TP4, conc 1 4 8 12 16
#   ./ix_agentx_glm53flash_b200.sh --smoke --conc 4     # plumbing check, ~20 min
#   ./ix_agentx_glm53flash_b200.sh --quick --conc 16    # A/B iteration, ~30 min
#   ./ix_agentx_glm53flash_b200.sh --mtp-steps 3 --conc 16    # MTP depth A/B
#   ./ix_agentx_glm53flash_b200.sh --hicache-size 200         # add a host tier
#   ./ix_agentx_glm53flash_b200.sh --dry-run            # print the env and exit
set -uo pipefail

usage() { sed -n '2,20p' "$0"; exit 1; }

# The sglang cu13 image ships lsof but not psmisc, so fuser is not available.
kill_port() {
    local port="$1" pids
    pids=$(lsof -t -i ":$port" -sTCP:LISTEN 2>/dev/null)
    [ -n "$pids" ] && kill -9 $pids 2>/dev/null
    return 0
}

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_HOME="${BENCH_HOME:-$(dirname "$HERE")}"
IX="${IX:-/home/jacchang/InferenceX}"
RECIPE="$HERE/recipe_glm53flash_fp4_b200_sglang_mtp.sh"

DOCKER="${DOCKER:-lmsysorg/sglang:v0.5.20-cu130}"
MODEL_ID="nvidia/GLM-5.3-Flash-NVFP4"
CKPT="${CKPT:-/data/huggingface/hub/nvidia/GLM-5.3-Flash-NVFP4}"

TAG=""
TP=4
EP=1
CONC_LIST="1 4 8 12 16"
DURATION="${DURATION:-3600}"
GPUS=""
PORT_BASE=28900
MEM_FRACTION_STATIC=0.85
CHUNKED_PREFILL_SIZE=16384
# steps 5 / 6 draft tokens is what the local fixed-length TP4 MTP run used and
# is the deepest EAGLE ladder GLM-5.3-Flash's single nextn head sustains here.
MTP_STEPS=5
SPEC_DECODING=mtp
ACC_MODE=real
GOLDEN_AL=""
HICACHE_SIZE=0
CONTEXT_LENGTH=""
MAMBA_FULL_MEMORY_RATIO=""
MAX_MAMBA_CACHE_SIZE=""
ENABLE_POWER=0
QUICK=0
SMOKE=0
FAST=0
DRY_RUN=0
declare -a EXTRA_ENV=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)        TAG="$2"; shift 2 ;;
        --docker)     DOCKER="$2"; shift 2 ;;
        --ckpt)       CKPT="$2"; shift 2 ;;
        --tp)         TP="$2"; shift 2 ;;
        --ep)         EP="$2"; shift 2 ;;
        --conc)       CONC_LIST="$2"; shift 2 ;;
        --duration)   DURATION="$2"; shift 2 ;;
        --gpus)       GPUS="$2"; shift 2 ;;
        --port)       PORT_BASE="$2"; shift 2 ;;
        --mem-fraction)      MEM_FRACTION_STATIC="$2"; shift 2 ;;
        --chunked-prefill)   CHUNKED_PREFILL_SIZE="$2"; shift 2 ;;
        --mtp-steps)  MTP_STEPS="$2"; shift 2 ;;
        --no-mtp)     SPEC_DECODING=none; shift ;;
        --acc)        ACC_MODE="$2"; shift 2 ;;
        --golden-al)  GOLDEN_AL="$2"; ACC_MODE=golden; shift 2 ;;
        --hicache-size)      HICACHE_SIZE="$2"; shift 2 ;;
        --context-length)    CONTEXT_LENGTH="$2"; shift 2 ;;
        --mamba-ratio)       MAMBA_FULL_MEMORY_RATIO="$2"; shift 2 ;;
        --max-mamba-cache-size) MAX_MAMBA_CACHE_SIZE="$2"; shift 2 ;;
        --power)      ENABLE_POWER=1; shift ;;
        --quick)      QUICK=1; shift ;;
        --smoke)      SMOKE=1; shift ;;
        --fast)       FAST=1; shift ;;
        --dry-run)    DRY_RUN=1; shift ;;
        --env)        EXTRA_ENV+=("$2"); shift 2 ;;
        -h|--help)    usage ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done

# ---- preflight -----------------------------------------------------------
[ -f "$RECIPE" ] || { echo "ERROR: recipe not found at $RECIPE" >&2; exit 1; }
[ -d "$CKPT" ]   || { echo "ERROR: checkpoint not found at $CKPT" >&2; exit 1; }
# benchmark_lib.sh's agentic half and the infx result package both moved in the
# 2026-09 tree; an older checkout fails deep inside install_agentic_deps.
for required in "$IX/benchmarks/benchmark_lib.sh" "$IX/benchmarks/runtime_settings.sh" \
                "$IX/utils/agentic-benchmark/requirements.txt" "$IX/utils/aiperf/pyproject.toml" \
                "$IX/infx/results/agentic/process_agentic_result.py"; do
    [ -e "$required" ] || {
        echo "ERROR: $required is missing. Update the InferenceX checkout at $IX" >&2
        echo "       (git pull && git submodule update --init utils/aiperf)." >&2
        exit 1
    }
done

[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
NGPU=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
if [ "$NGPU" -lt "$TP" ]; then
    echo "ERROR: TP=$TP needs $TP GPUs but only $NGPU are visible (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all})." >&2
    exit 1
fi

case "$ACC_MODE" in
    real) ;;
    golden)
        [ -n "$GOLDEN_AL" ] || {
            echo "ERROR: --acc golden needs --golden-al <AL>. golden_al_distribution/glm5.3_mtp.yaml" >&2
            echo "       carries only glm-5.3-fp8 at K=3 (2.99, itself marked PROVISIONAL and copied" >&2
            echo "       from GLM-5.2); there is no measured curve for GLM-5.3-Flash at any depth." >&2
            exit 1
        } ;;
    *) echo "ERROR: --acc must be 'real' or 'golden', got '$ACC_MODE'" >&2; exit 1 ;;
esac

# ---- InferenceX CI environment ------------------------------------------
# runtime_settings.sh owns the AIPERF_*/AGENTIC_* defaults that build_replay_cmd
# hard-requires (it uses check_env_vars, so an unset one aborts rather than
# falling back). Load it first, then override only what this lane changes.
# shellcheck source=/dev/null
source "$IX/benchmarks/runtime_settings.sh"

export INFMAX_CONTAINER_WORKSPACE="$IX"
export MODEL="$MODEL_ID"
export MODEL_PATH="$CKPT"
# GLM-5.3-Flash is not GLM-5.3, and upstream gives Flash variants their own
# prefix (DeepSeek-V4-Pro is dsv4, DeepSeek-V4.1-Flash is dsv41flash). The dot
# has to stay: resolve_trace_source globs glm5.3*, so glm5.3flash still selects
# the unfiltered 1M-context corpus, where a dotless glm53flash would fall
# through to the 256k variant without saying so. It also makes golden_length()
# look for glm5.3flash_mtp.yaml and fail loudly instead of quietly borrowing
# GLM-5.3's provisional curve.
export MODEL_PREFIX=glm5.3flash
export PRECISION=fp4
export FRAMEWORK=sglang
export RUNNER_TYPE=b200
export IMAGE="$DOCKER"
export SCENARIO_TYPE=agentic-coding
export THINKING_MODE=thinking_on
export SPEC_DECODING
export DISAGG=false
export IS_MULTINODE=false
export EVAL_ONLY=false
export TP EP_SIZE="$EP" PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1
export DP_ATTENTION=false
export REQUIRE_POWER=0
export ENABLE_AGENTX_POWER="$ENABLE_POWER"
# The workflow supplies this one, not runtime_settings.sh, and build_replay_cmd
# check_env_vars-aborts on an unset (as opposed to '0') value.
export AIPERF_EXPERIMENTAL_FAST=0
export GPU_MONITOR_INTERVAL=1
export MEM_FRACTION_STATIC CHUNKED_PREFILL_SIZE
export SPEC_NUM_STEPS="$MTP_STEPS" SPEC_NUM_DRAFT_TOKENS=$((MTP_STEPS + 1))
export ACC_MODE
[ -n "$GOLDEN_AL" ] && export GOLDEN_AL
[ -n "$MAMBA_FULL_MEMORY_RATIO" ] && export MAMBA_FULL_MEMORY_RATIO
[ -n "$MAX_MAMBA_CACHE_SIZE" ] && export MAX_MAMBA_CACHE_SIZE

# The client replays the corpus unfiltered when MAX_MODEL_LEN=0. Capping the
# server without capping the client turns the over-length traces into 4xxs that
# still occupy a lane, so the two move together.
if [ -n "$CONTEXT_LENGTH" ]; then
    export CONTEXT_LENGTH MAX_MODEL_LEN="$CONTEXT_LENGTH"
else
    export MAX_MODEL_LEN=0          # native 1,048,576
fi

# KV tier. GLM-5.3-Flash only has 11 full-attention layers, so TP4 holds ~12.4M
# tokens of fp8 KV in HBM and a host tier is opt-in rather than load-bearing.
if [ "$HICACHE_SIZE" -gt 0 ]; then
    export KV_OFFLOADING=dram
    export KV_OFFLOAD_BACKEND=hicache
    export KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'
    export HICACHE_SIZE
    # Ask InferenceX's own matrix logic for the node budget instead of
    # transcribing a number; b200-nscale is the runner whose 2,063,920 MiB
    # matches this host, and TP4 at dram-utilization 0.80 yields 865 GB.
    TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-$(cd "$IX" && python3 -c "
import yaml
from infx.matrix.generate import agentic_dram_offload_gb
print(agentic_dram_offload_gb(
    {'dram-utilization': 0.8},
    {'tp': $TP, 'ep': $EP, 'kv-offloading': 'dram'},
    'cluster:b200-nscale',
    yaml.safe_load(open('configs/runners.yaml')),
))")}"
else
    export KV_OFFLOADING=none
    # process_agentic_result rejects a result that names a backend it did not use.
    export KV_OFFLOAD_BACKEND=""
    export KV_OFFLOAD_BACKEND_METADATA=""
    TOTAL_CPU_DRAM_GB=0
fi
export TOTAL_CPU_DRAM_GB

# /raid is node-local NVMe; the venv rebuild and the ~100 GB WEKA corpus are
# both far happier there than on the NFS home.
export AIPERF_RUNTIME_DIR="${AIPERF_RUNTIME_DIR:-/raid/home/jacchang/ix-agentic-runtime}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/raid/home/jacchang/hf_hub_cache}"
mkdir -p "$AIPERF_RUNTIME_DIR" "$HF_HUB_CACHE"

for kv in "${EXTRA_ENV[@]:-}"; do [ -n "$kv" ] && export "${kv?}"; done

# ---- run-length modes ----------------------------------------------------
[ -n "$TAG" ] || TAG="TP${TP}_EP${EP}-mtp${MTP_STEPS}"
[ "$SPEC_DECODING" = "none" ] && TAG="TP${TP}_EP${EP}-nospec"
if [ "$FAST" = "1" ]; then
    # 1 warmup request per lane leaves the prefix caches nearly empty, which is
    # exactly what an agentic run is supposed to exercise. Useful only to prove
    # the pipe works end to end.
    export AIPERF_EXPERIMENTAL_FAST=1
    TAG="${TAG}-fast"
fi
if [ "$QUICK" = "1" ]; then
    # Half warmup, 10-minute window: conc 16 lands in ~30 min instead of ~100.
    # Under 900s the scenario marks submission_valid=false, so compare these
    # only against other --quick runs.
    export AIPERF_WARMUP_REQUESTS_PER_LANE=5
    export AIPERF_UNSAFE_OVERRIDE=true
    DURATION=600
    TAG="${TAG}-quick"
fi
if [ "$SMOKE" = "1" ]; then
    export AIPERF_WARMUP_REQUESTS_PER_LANE=1
    export AIPERF_UNSAFE_OVERRIDE=true
    DURATION=300
    TAG="${TAG}-smoke"
fi
export DURATION

# ---- result layout (GLM.sh's convention) ---------------------------------
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")"
DOCKER_FILENAME="$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')"
ROOT="$BENCH_HOME/results/$MODEL_NAME/$DOCKER_FILENAME/bench-Agentic-$TAG"
mkdir -p "$ROOT"
SWEEP_LOG="$ROOT/sweep.log"

{
    echo "=== $(date -Is) GLM-5.3-Flash NVFP4 AgentX ==="
    echo "tag=$TAG tp=$TP ep=$EP spec=$SPEC_DECODING steps=$MTP_STEPS acc=$ACC_MODE${GOLDEN_AL:+($GOLDEN_AL)}"
    echo "mem_fraction=$MEM_FRACTION_STATIC chunked_prefill=$CHUNKED_PREFILL_SIZE context=${CONTEXT_LENGTH:-native-1M}"
    echo "kv_offloading=$KV_OFFLOADING hicache_size=${HICACHE_SIZE} total_cpu_dram_gb=$TOTAL_CPU_DRAM_GB"
    echo "duration=$DURATION conc=($CONC_LIST) gpus=${CUDA_VISIBLE_DEVICES:-all}"
    echo "ix=$IX image=$DOCKER"
    echo "root=$ROOT"
} | tee -a "$SWEEP_LOG"

if [ "$DRY_RUN" = "1" ]; then
    echo "--- recipe env (dry run) ---"
    env | grep -E '^(MODEL|TP|EP_SIZE|PP_SIZE|DCP_SIZE|PCP_SIZE|DP_ATTENTION|SPEC_|ACC_MODE|GOLDEN_AL|KV_|HICACHE|TOTAL_CPU_DRAM_GB|MAX_MODEL_LEN|CONTEXT_LENGTH|MAMBA_|MAX_MAMBA|MEM_FRACTION|CHUNKED_|AIPERF_|AGENTIC_|INFMAX_|RUNNER_TYPE|PRECISION|FRAMEWORK|SCENARIO_|DURATION|ENABLE_AGENTX_POWER|REQUIRE_POWER|IS_MULTINODE|DISAGG|EVAL_ONLY|IMAGE|HF_HUB_CACHE)' | sort
    exit 0
fi

# ---- sweep ---------------------------------------------------------------
for CONC in $CONC_LIST; do
    export CONC
    export EXP_NAME="glm5.3flash_tp${TP}_conc${CONC}_kv${KV_OFFLOADING}_spec-${SPEC_DECODING}"
    export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-pp1-dcp1-pcp1-ep${EP}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_local-b200"
    export RESULT_DIR="$ROOT/$EXP_NAME"
    export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
    export PORT=$((PORT_BASE + CONC))
    export GPU_METRICS_CSV="$RESULT_DIR/gpu_metrics.csv"
    export SGLANG_TORCH_PROFILER_DIR="$RESULT_DIR"

    if [ -f "$RESULT_DIR/$RESULT_FILENAME.json" ]; then
        echo ">>> conc=$CONC already has a result, skipping." | tee -a "$SWEEP_LOG"
        continue
    fi
    mkdir -p "$RESULT_DIR"

    # A server that outlived the previous point keeps both the port and its
    # share of HBM, and booting on top of it silently halves the KV pool.
    kill_port "$PORT"
    for _ in $(seq 1 60); do
        busy=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
        [ "${busy:-0}" -le 1024 ] && break
        echo "    waiting for GPU reclaim (max used=${busy} MiB)" | tee -a "$SWEEP_LOG"
        sleep 15
    done

    echo ">>> $(date -Is) starting conc=$CONC port=$PORT -> $RESULT_DIR" | tee -a "$SWEEP_LOG"
    bash "$RECIPE" > "$RESULT_DIR/recipe.log" 2>&1
    rc=$?
    echo ">>> $(date -Is) conc=$CONC exit=$rc" | tee -a "$SWEEP_LOG"

    # The two numbers that decide whether the hybrid memory split is sane: the
    # full-attention token pool and the KDA state slot count. Both move with
    # --mamba-ratio, and neither appears in the result JSON.
    grep -hoE 'KV Cache is allocated.*|max_mamba_cache_size=[0-9]+|Memory pool end.*' \
        "$RESULT_DIR/server.log" 2>/dev/null | sort -u | sed 's/^/    /' | tee -a "$SWEEP_LOG"

    # The recipe leaves the server up when it exits non-zero mid-flight.
    pkill -9 -f 'sglang.launch_server' 2>/dev/null
    pkill -9 -f 'sglang::' 2>/dev/null
    kill_port "$PORT"
    sleep 30
done

echo "=== $(date -Is) sweep done: $ROOT ===" | tee -a "$SWEEP_LOG"
echo "Summarize with: $HERE/ix_agentx_summarize.py --hw b200 $ROOT"
