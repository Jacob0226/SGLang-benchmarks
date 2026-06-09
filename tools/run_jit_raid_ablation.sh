#!/usr/bin/env bash
# Ablation: JIT-kernel vs RAID0 contribution under wait_complete + FP8 KV cache.
# 2x2: kernel {AOT, JIT} x L3 {/tmp single drive, /raid RAID0}.
# wait_complete makes hit rate deterministic (theoretical max) AND puts the
# L3->L2 prefetch + L2->L1 transfer on the critical path, so both RAID0 (L3
# speed) and JIT (transfer kernel) effects become measurable.
#
# Run INSIDE the container (started with -v /raid:/raid, --init).
#   bash tools/run_jit_raid_ablation.sh
set -euo pipefail
cd "$(dirname "$0")/.."

DOCKER=${DOCKER:-rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260601}
MODEL=${MODEL:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
COMMON=(--docker "$DOCKER" --model "$MODEL" --cache-modes "L3_file"
        --aiter-fp8-prefill-attn 0 --L1-size 10 --hicache-size 20
        --num-rounds 8 --num-clients 30 --page-size 64)
L2=20

ks() { pkill -9 -f '[s]glang' 2>/dev/null || true; sleep 12; }

run() {  # $1 tag, $2 L3 base dir
  ks
  echo "==== RUN $1  (L3=$2, wait_complete) ===="
  L3_BASE_DIR="$2" HICACHE_PREFETCH_POLICY=wait_complete \
    ./cascade_dsr1_lite.sh --tag "$1" "${COMMON[@]}"
}

echo "########## AOT kernel ##########"
bash tools/apply_pr25154_jit.sh --restore
run abl_AOT_tmp  /tmp
run abl_AOT_raid /raid

echo "########## JIT kernel ##########"
bash tools/apply_pr25154_jit.sh
run abl_JIT_tmp  /tmp
run abl_JIT_raid /raid
ks

echo
echo "############### ABLATION SUMMARY (R4-R7 TTFT / hit) ###############"
for tag in abl_AOT_tmp abl_AOT_raid abl_JIT_tmp abl_JIT_raid; do
  f=$(ls -t results/*/DeepSeek-R1-0528/bench-${tag}/L3file_L2_size_${L2}/bench_multiturn.log 2>/dev/null | head -1)
  echo "----- $tag -----"
  grep -E "Round [4-7]:" "$f" 2>/dev/null || echo "  (no log)"
done
