#!/usr/bin/env bash
# AgentX on the 20261115 image: does --hybrid-cache-checkpoints-level explain the TTFT gap vs 1106?
# Same setup as the 11/16 "off" lane (agentx_onoff_v2_1116.sh): f32 KDA state, the image's
# sgl-amd-mono, BF16 KV + monokernel, TileLang DSA, ReplaySSM off, SGLang cbd4439f3d + sglang#49036.
#   l2: image default (level 2, soft cap of one checkpoint per path)
#   l3: --hybrid-cache-checkpoints-level 3 --mamba-max-states-per-path -1 (the 1106 behaviour:
#       a checkpoint every prefill chunk and every --mamba-track-interval tokens, no cap)
# Run inside jacchang_pre1115 (node 151).
# usage: agentx_ckpt_level_1119.sh <l2|l3> <gpus> <port_base> <mtp steps> "<conc list>" [duration_s]
ulimit -c 0
set -uo pipefail
MODE=$1 GPUS=$2 PORT_BASE=$3 STEPS=$4 CONC=$5 DUR=${6:-3600}
D=/shared_nfs/jacchang/agentic_1119_ckpt
IMG=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261115
SGL=/home/jacchang/PR/wt-sglang-img1115-49036
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export HSA_NO_SCRATCH_RECLAIM=1
export BENCH_HOME=$D/$MODE
mkdir -p $BENCH_HOME
export AIPERF_RUNTIME_DIR=/tmp/ix_runtime_1119ckpt_$PORT_BASE
export HF_HUB_CACHE=/shared_nfs/jacchang/hf_hub_cache_ix
log() { echo "[$(date -Is)] [$MODE] $*" | tee -a $D/driver.log; }
sha() { git -c safe.directory='*' -C "$1" log -1 --format=%h; }

ARGS=(--env "PYTHONPATH=$SGL/python" --server-arg "--kv-cache-dtype bfloat16" --server-arg --enable-amd-monokernel)
case $MODE in
    l2) ;;
    l3) ARGS+=(--server-arg "--hybrid-cache-checkpoints-level 3" --server-arg "--mamba-max-states-per-path -1") ;;
    *) echo "mode must be l2 or l3" >&2; exit 1 ;;
esac
TAG=TP4_EP1-mtp$STEPS-$(sha $SGL)-statefp32-mono$(sha /sgl-workspace/sgl-amd-mono)-ckpt$MODE-1119
log "run $TAG conc ($CONC) gpus $GPUS dur $DUR begin ${DRY:+(dry run)}"
cd /home/jacchang/SGLang-benchmarks/Agentic
mkdir -p /tmp/flydsl_agentic1119_$MODE
FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl_agentic1119_$MODE SGLANG_CACHE_DIR=/root/sglang_cache_agentic1119_$MODE \
    ./ix_agentx_glm53flash.sh --platform mi355x --docker $IMG --gpus $GPUS --port $PORT_BASE \
    --conc "$CONC" --mtp-steps $STEPS --dsa-backend tilelang --replayssm off --duration $DUR --tag $TAG \
    "${ARGS[@]}" ${DRY:+--dry-run} > $D/agentx_${TAG}${DRY:+-dry}.log 2>&1
log "run $TAG conc ($CONC) exit $?"
