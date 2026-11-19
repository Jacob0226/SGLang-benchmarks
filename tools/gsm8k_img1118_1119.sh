#!/usr/bin/env bash
# GSM8K gate for the 1118 PR stack before its AgentX: the AgentX conc 16-64 server (BF16 KDA
# state, FP8 KV, Triton DSA, ReplaySSM, MTP5, real acceptance), then sgl-eval GSM8K (1319,
# thinking, temperature 1.0, top-p 0.95). Run inside jacchang_pr1118 (node 152).
# usage: gsm8k_img1118_1119.sh <gpus> <port> <out_dir>
set -uo pipefail
ulimit -n 65536
GPUS=$1 PORT=$2 OUT=$3
SGL=/home/jacchang/PR/wt-glm53-1119-prstack-img1118-v2
MONO=/shared_nfs/jacchang/exp/sgl-amd-mono-1119-a04ceab-pr141
MODEL=/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4
mkdir -p $OUT
stamp() { echo "[$(date -Is)] [gsm8k-$PORT] $*" | tee -a $OUT/driver.log; }

if [ ! -x /root/sgl_eval_venv/bin/sgl-eval ]; then
    python3 -m venv /root/sgl_eval_venv && /root/sgl_eval_venv/bin/pip install -q \
        "git+https://github.com/sgl-project/sgl-eval.git@a231b7a439b235090ff7baa30778fa2b514309ae" \
        > $OUT/sgl_eval_install.log 2>&1
    mkdir -p /root/.cache && cp -r /home/jacchang/.cache/sgl_eval /root/.cache/
fi
[ -x /root/sgl_eval_venv/bin/sgl-eval ] || { stamp "sgl-eval install failed"; exit 1; }

stamp "launch $(git -c safe.directory='*' -C $SGL log -1 --format=%h) mono $(git -c safe.directory='*' -C $MONO log -1 --format=%h) gpus $GPUS"
t0=$(date +%s)
env SGLANG_OPT_GLM53_KDA_PTPC_MODULES=qkv_proj,f_a_proj,g_a_proj,o_proj \
    HSA_NO_SCRATCH_RECLAIM=1 HSA_DISABLE_COREDUMP_ON_EXCEPTION=1 HIP_VISIBLE_DEVICES=$GPUS \
    ROCM_QUICK_REDUCE_QUANTIZATION=INT4 AITER_QUICK_REDUCE_QUANTIZATION=INT4 \
    PYTHONPATH=$MONO:$SGL/python FLYDSL_RUNTIME_CACHE_DIR=/tmp/flydsl_gsm8k_$PORT \
    python3 -m sglang.launch_server --model-path $MODEL --served-model-name amd/GLM-5.3-Flash-Quark-MXFP4 \
    --port $PORT --trust-remote-code --tp 4 --ep-size 1 \
    --kv-cache-dtype fp8_e4m3 --dsa-prefill-backend triton --dsa-decode-backend triton \
    --moe-runner-backend aiter --mamba-radix-cache-strategy extra_buffer --reasoning-parser glm45 \
    --chunked-prefill-size 16384 --mem-fraction-static 0.85 --max-running-requests 64 \
    --cuda-graph-max-bs-decode 64 --mamba-full-memory-ratio 0.9 --mamba-ssm-dtype bfloat16 \
    --hybrid-cache-checkpoints-level 3 \
    --cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2560 \
    --speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 6 --enable-linear-replayssm-spec --watchdog-timeout 1800 \
    > $OUT/server.log 2>&1 &
pid=$!
ok=0
for _ in $(seq 1 720); do
    curl -sf "http://127.0.0.1:$PORT/health_generate" >/dev/null && { ok=1; break; }
    kill -0 $pid 2>/dev/null || break
    sleep 5
done
rc=1
if [ $ok = 1 ]; then
    stamp "ready after $(($(date +%s) - t0)) s"
    t1=$(date +%s)
    /root/sgl_eval_venv/bin/sgl-eval run gsm8k --base-url http://127.0.0.1:$PORT/v1 \
        --model amd/GLM-5.3-Flash-Quark-MXFP4 --num-threads 1319 --max-tokens 4096 \
        --temperature 1.0 --top-p 0.95 --seed 0 --thinking --out-dir $OUT/gsm8k > $OUT/gsm8k.log 2>&1
    rc=$?
    stamp "gsm8k exit $rc after $(($(date +%s) - t1)) s: $(grep -iE 'score|accuracy' $OUT/gsm8k.log | tail -2 | tr '\n' ' ')"
    grep -E "accept len" $OUT/server.log | tail -1 >> $OUT/driver.log
else
    stamp "server failed"
fi
kill -TERM $pid 2>/dev/null
for _ in $(seq 1 60); do kill -0 $pid 2>/dev/null || break; sleep 2; done
pkill -TERM -f "sglang.launch_server.*--port $PORT( |$)" 2>/dev/null
sleep 10
stamp "done"
exit $rc
