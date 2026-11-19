#!/usr/bin/env bash
# InferenceX AgentX on rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261118, stock vs stock + PR stack.
#   stock: image SGLang 3a41a59919 + image sgl-amd-mono a04ceab, TileLang DSA, BF16 KV.
#          Run inside jacchang_stock1118.
#   pr:    SGLang wt-glm53-1119-prstack-img1118-v2 (13 PRs, no #44661) + sgl-amd-mono a04ceab
#          + #141, Triton DSA, #38764's env, aiter#8006 in the container's aiter.
#          Run inside jacchang_pr1118.
# Groups follow section 17.7.0: m1 = conc 1 MTP5 + monokernel (BF16 KV);
# m48 = conc 4 8 MTP3 + monokernel (BF16 KV); hi = conc 16 32 64 MTP5, no monokernel
# (stock BF16 KV, pr FP8 KV). STATE=bf16|fp32 picks the KDA state pool dtype.
# usage: STATE=bf16 agentx_img1118_1119.sh <stock|pr> <gpus> <port_base> <m1|m48|hi> [duration s]
ulimit -c 0
set -uo pipefail
LANE=$1 GPUS=$2 PORT_BASE=$3 GROUP=$4 DUR=${5:-3600}
STATE=${STATE:?STATE must be bf16 or fp32}
D=${OUT_DIR:-/shared_nfs/jacchang/agentx_1119_img1118}
IMG=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261118
SGL_PR=/home/jacchang/PR/wt-glm53-1119-prstack-img1118-v2
MONO_PR=/shared_nfs/jacchang/exp/sgl-amd-mono-1119-a04ceab-pr141
export HSA_DISABLE_COREDUMP_ON_EXCEPTION=1
export HSA_NO_SCRATCH_RECLAIM=1
export BENCH_HOME=$D/$LANE-state$STATE
mkdir -p $BENCH_HOME
export AIPERF_RUNTIME_DIR=/tmp/ix_runtime_1119img1118_$PORT_BASE
export HF_HUB_CACHE=/shared_nfs/jacchang/hf_hub_cache_ix
log() { echo "[$(date -Is)] [$LANE-$GROUP] $*" | tee -a $D/driver.log; }
sha() { git -c safe.directory='*' -C "$1" log -1 --format=%h; }

case $GROUP in
    m1)  CONC="1";        STEPS=5; MONO=1 ;;
    m48) CONC="4 8";      STEPS=3; MONO=1 ;;
    hi)  CONC="16 32 64"; STEPS=5; MONO=0 ;;
    *) echo "group must be m1, m48 or hi" >&2; exit 1 ;;
esac
case $STATE in
    bf16) ARGS=(--server-arg "--mamba-ssm-dtype bfloat16") ;;
    fp32) ARGS=(--server-arg "--mamba-ssm-dtype float32") ;;
    *) echo "STATE must be bf16 or fp32" >&2; exit 1 ;;
esac
case $LANE in
    stock)
        DSA=tilelang
        ARGS+=(--server-arg "--kv-cache-dtype bfloat16")
        TAG=TP4_EP1-mtp$STEPS-stock$(sha /sgl-workspace/sglang)-mono$(sha /sgl-workspace/sgl-amd-mono)-state$STATE-1119
        ;;
    pr)
        DSA=triton
        ARGS+=(--env "PYTHONPATH=$MONO_PR:$SGL_PR/python"
               --env "SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj")
        [ $MONO = 1 ] && ARGS+=(--server-arg "--kv-cache-dtype bfloat16")
        TAG=TP4_EP1-mtp$STEPS-prstack$(sha $SGL_PR)-mono$(sha $MONO_PR)-aiter8006-state$STATE-1119
        ;;
    *) echo "lane must be stock or pr" >&2; exit 1 ;;
esac
[ $MONO = 1 ] && ARGS+=(--server-arg --enable-amd-monokernel)
log "run $TAG conc ($CONC) gpus $GPUS dur $DUR begin ${DRY:+(dry run)}"
cd /home/jacchang/SGLang-benchmarks/Agentic
FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl_1119img1118_$LANE SGLANG_CACHE_DIR=/root/sglang_cache_1119img1118_$LANE \
    ./ix_agentx_glm53flash.sh --platform mi355x --docker $IMG --gpus $GPUS --port $PORT_BASE \
    --conc "$CONC" --mtp-steps $STEPS --dsa-backend $DSA --duration $DUR --tag $TAG "${ARGS[@]}" \
    ${DRY:+--dry-run} > $D/agentx_${TAG}_$GROUP${DRY:+-dry}.log 2>&1
log "run $TAG conc ($CONC) exit $?"
