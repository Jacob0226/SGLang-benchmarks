#!/usr/bin/env bash
# InferenceX AgentX replica for GLM-5.2 MXFP4 on MI355X / SGLang.
#
# Runs SemiAnalysisAI/InferenceX's own recipe
#   benchmarks/single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh
# once per concurrency, with the same env the run-sweep workflow injects
# (configs/amd-master.yaml :: glm5.2-fp4-mi355x-sglang-agentic-mtp).
#
# Run this INSIDE the container. GPU selection comes from HIP_VISIBLE_DEVICES.
#
# Results land in GLM.sh's layout: results/<model>/<docker-image>/<leaf-tag>,
# with the leaf tag composed the way GLM.sh composes it.
#
# The tag defaults to TP<tp>_EP<ep>, so the arm alone picks the leaf.
#
#   ./ix_agentx_glm52.sh --conc "1 2 4 8 10 12 16"   # bench-Agentic-TP4_EP4
#   ./ix_agentx_glm52.sh --arm tp4 --smoke --conc 4  # bench-Agentic-TP4_EP1
set -uo pipefail

IX=/workspace
TAG=""
DOCKER="${DOCKER:-rocm/sgl-dev:v0.5.18-rocm720-mi35x-20260824}"
ARM="tep4"                 # tep4 = tp4/ep4 (board arm) | tp4 = tp4/ep1 | tp8 = tp8/ep8
CONC_LIST=""
DURATION="${DURATION:-3600}"
FAST=0
SMOKE=0
DSA_PREFILL="tilelang"
DSA_DECODE="tilelang"
PORT_BASE=28800
declare -a EXTRA_ENV=()

usage() { sed -n '2,14p' "$0"; exit 1; }
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)      TAG="$2"; shift 2 ;;
        --docker)   DOCKER="$2"; shift 2 ;;
        --arm)      ARM="$2"; shift 2 ;;
        --conc)     CONC_LIST="$2"; shift 2 ;;
        --duration) DURATION="$2"; shift 2 ;;
        --dsa)          DSA_PREFILL="$2"; DSA_DECODE="$2"; shift 2 ;;
        --dsa-prefill)  DSA_PREFILL="$2"; shift 2 ;;
        --dsa-decode)   DSA_DECODE="$2"; shift 2 ;;
        --port)     PORT_BASE="$2"; shift 2 ;;
        --fast)     FAST=1; shift ;;
        --smoke)    SMOKE=1; shift ;;
        --env)      EXTRA_ENV+=("$2"); shift 2 ;;
        *) echo "unknown option: $1" >&2; usage ;;
    esac
done
case "$ARM" in
    tep4) TP=4; EP=4; DEFAULT_CONC="1 2 4 8 10 12 16" ;;
    tp4)  TP=4; EP=1; DEFAULT_CONC="1 2 4 8 10 12 16" ;;
    tp8)  TP=8; EP=8; DEFAULT_CONC="1 2 4" ;;
    *) echo "unknown --arm '$ARM' (tep4|tp4|tp8)" >&2; exit 1 ;;
esac
[ -n "$CONC_LIST" ] || CONC_LIST="$DEFAULT_CONC"
# Derive from the arm so the directory name can never disagree with what the
# server was actually launched with.
[ -n "$TAG" ] || TAG="TP${TP}_EP${EP}"

NGPU=$(python3 -c 'import torch;print(torch.cuda.device_count())')
if [ "$NGPU" -lt "$TP" ]; then
    echo "ERROR: arm '$ARM' needs TP=$TP but only $NGPU GPU(s) are visible (HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-all})." >&2
    exit 1
fi

# ---- recipe env: mirrors .github/workflows/benchmark-tmpl.yml -------------
export MODEL=amd/GLM-5.2-MXFP4
export MODEL_PATH=/data/huggingface/hub/amd/GLM-5.2-MXFP4
export MODEL_PREFIX=glm5.2
export PRECISION=fp4
export FRAMEWORK=sglang
export SCENARIO_TYPE=agentic-coding
export SPEC_DECODING=mtp
export DISAGG=false
export IS_MULTINODE=false
export TP EP_SIZE="$EP" PP_SIZE=1 DCP_SIZE=1 PCP_SIZE=1
export DP_ATTENTION=false
export KV_OFFLOADING=dram
export KV_OFFLOAD_BACKEND=hicache
# process_agentic_result.py rejects the run unless the metadata name matches.
export KV_OFFLOAD_BACKEND_METADATA='{"name":"hicache"}'
# MODELS.md caps agentic DRAM offload proportionally to the GPUs a config uses,
# so the budget changes with TP. Ask InferenceX's own matrix logic instead of
# transcribing a number: this is the function the run-sweep workflow calls, and
# it reproduces the 1199 GB (TP4) / 2399 GB (TP8) the board's runs recorded.
TOTAL_CPU_DRAM_GB="${TOTAL_CPU_DRAM_GB:-$(cd "$IX" && python3 -c "
import sys, yaml
sys.path.insert(0, 'utils/matrix_logic')
from generate_sweep_configs import agentic_dram_offload_gb
print(agentic_dram_offload_gb(
    {'dram-utilization': 0.8},
    {'tp': $TP, 'ep': $EP, 'kv-offloading': 'dram'},
    'cluster:mi355x-amds',
    yaml.safe_load(open('configs/runners.yaml')),
))")}"
export TOTAL_CPU_DRAM_GB
export INFMAX_CONTAINER_WORKSPACE="$IX"
export AIPERF_FAILED_REQUEST_THRESHOLD=0.10
export ENABLE_AGENTX_POWER=0
# One venv for the whole sweep: install_agentic_deps rebuilds it per process,
# but a warm uv cache turns that into seconds instead of minutes.
export AIPERF_RUNTIME_DIR=/data/ix-agentic-runtime

# ---- PR stack switches ---------------------------------------------------
export DSA_PREFILL_BACKEND="$DSA_PREFILL"
export DSA_DECODE_BACKEND="$DSA_DECODE"
# aiter's gluon fp8_mqa_logits kernel fails to compile (LLVM `Begin <= End`)
# when the indexer's 32 heads/rank meet BLOCK_M=2 and a >=2 GiB logits buffer,
# which is exactly TP4 + agentic long context. Shrinking the chunk budget keeps
# every chunk under the 2 GiB buffer-descriptor cap and avoids that path.
export SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION="${SGLANG_DSA_MQA_LOGITS_FREE_MEM_FRACTION:-0.04}"
for kv in "${EXTRA_ENV[@]:-}"; do [ -n "$kv" ] && export "${kv?}"; done

if [ "$FAST" = "1" ]; then
    export AIPERF_EXPERIMENTAL_FAST=1   # duration 1200s, 1 warmup req per lane
    TAG="${TAG}-fast"
fi
if [ "$SMOKE" = "1" ]; then
    # Plumbing check only: one warmup request per lane and a sub-900s window,
    # which AIPerf flags as submission_valid=false. Never compare these numbers.
    export AIPERF_WARMUP_REQUESTS_PER_LANE=1
    DURATION=300
    TAG="${TAG}-smoke"
fi

# Not $HOME: the container runs as root while the bind mount is the host user's.
BENCH_HOME="${BENCH_HOME:-/home/jacchang/SGLang-benchmarks}"
# GLM.sh's layout. The leaf mirrors its convention: mode first, then the
# scenario family, so agentic replay runs sort apart from the fixed-length ones.
MODEL_NAME="$(basename "$(dirname "$MODEL_PATH")")_$(basename "$MODEL_PATH")"
DOCKER_FILENAME="$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')"
ROOT="$BENCH_HOME/results/$MODEL_NAME/$DOCKER_FILENAME/bench-Agentic-$TAG"
mkdir -p "$ROOT"
SWEEP_LOG="$ROOT/sweep.log"
echo "=== $(date -Is) tag=$TAG arm=$ARM tp=$TP ep=$EP dsa=$DSA_PREFILL/$DSA_DECODE duration=$DURATION conc=($CONC_LIST) ===" | tee -a "$SWEEP_LOG"
env | grep -E '^SGLANG_|^DSA_' | sort | tee -a "$SWEEP_LOG"

for CONC in $CONC_LIST; do
    export CONC DURATION
    export EXP_NAME="glm5.2_tp${TP}_conc${CONC}_kvdram-hicache_spec-mtp"
    export RESULT_FILENAME="${EXP_NAME}_${PRECISION}_${FRAMEWORK}_tp${TP}-pp1-dcp1-pcp1-ep${EP}-dpa${DP_ATTENTION}_disagg-${DISAGG}_spec-${SPEC_DECODING}_conc${CONC}_local-mi355x"
    export RESULT_DIR="$ROOT/$EXP_NAME"
    export AGENTIC_OUTPUT_DIR="$RESULT_DIR"
    export PORT=$((PORT_BASE + CONC))
    export GPU_METRICS_CSV="$RESULT_DIR/gpu_metrics.csv"

    if [ -f "$RESULT_DIR/$RESULT_FILENAME.json" ]; then
        echo ">>> conc=$CONC already has a result, skipping." | tee -a "$SWEEP_LOG"
        continue
    fi
    mkdir -p "$RESULT_DIR"

    # aiter's JIT baton records the builder's pid in the lock file and only
    # breaks the lock when that pid is gone. A server killed mid-build leaves
    # the lock behind, and every rank of the next run blocks in
    # file_baton.wait() with the GPUs idle. Nothing is building right now, so
    # any surviving lock is stale by definition.
    find /sgl-workspace/aiter/aiter/jit/build -maxdepth 1 -name 'lock_*' -delete 2>/dev/null

    # A server that outlived the previous point keeps both the port and its
    # share of HBM. Booting on top of it silently halves the KV pool (seen as
    # max_total_num_tokens 321216 instead of ~2.48M), so refuse to start until
    # the port is free and our GPUs are actually idle.
    fuser -k "$PORT/tcp" 2>/dev/null
    for _ in $(seq 1 60); do
        busy=$(rocm-smi --showmemuse 2>/dev/null | awk -v want="${HIP_VISIBLE_DEVICES:-}" '
            match($0, /GPU\[[0-9]+\]/) { idx = substr($0, RSTART+4, RLENGTH-5) }
            /GPU Memory Allocated \(VRAM%\)/ {
                if (want != "") { keep=0; n=split(want,w,","); for(i=1;i<=n;i++) if (w[i]+0==idx+0) keep=1; if(!keep) next }
                if ($NF+0 > m) m = $NF+0
            } END { print m+0 }')
        [ "${busy:-0}" -le 10 ] && break
        echo "    waiting for GPU reclaim (vram%max=$busy)" | tee -a "$SWEEP_LOG"
        sleep 15
    done

    echo ">>> $(date -Is) starting conc=$CONC port=$PORT -> $RESULT_DIR" | tee -a "$SWEEP_LOG"
    ( cd "$IX" && bash benchmarks/single_node/agentic/glm5.2_fp4_mi355x_sglang_mtp.sh ) \
        > "$RESULT_DIR/recipe.log" 2>&1
    rc=$?
    echo ">>> $(date -Is) conc=$CONC exit=$rc" | tee -a "$SWEEP_LOG"

    # The recipe leaves the server running when it exits non-zero mid-flight.
    pkill -9 -f 'sglang.launch_server' 2>/dev/null
    pkill -9 -f 'sglang::' 2>/dev/null
    fuser -k "$PORT/tcp" 2>/dev/null
    sleep 30
done

echo "=== $(date -Is) sweep done: $ROOT ===" | tee -a "$SWEEP_LOG"
