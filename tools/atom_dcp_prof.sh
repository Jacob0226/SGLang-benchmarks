#!/usr/bin/env bash
# Profile ATOM GLM-5.2-MXFP4 with the InferenceX conc>=16 arm (TP4 + DCP4 + MTP),
# one concurrency per call, so two calls can share an 8-GPU box (GPU 0-3 / 4-7).
# Run INSIDE an atom-dev container whose HIP_VISIBLE_DEVICES selects 4 GPUs.
#
# Usage: atom_dcp_prof.sh <conc> <server_port> <engine_port>
#   IN_OUT="70000:16 1024:16" (default) — shapes profiled on the same server.
#
# Per-concurrency draft depth / forced acceptance length / graph sizes follow
# InferenceX glm5.2_fp4_mi355x_atom_mtp.sh @71d4712 (rocm/atom-dev:nightly_202609211553):
#   DCP arm: K4 / AL 3.33 below conc 48, K3 / AL 2.99 at conc >= 48,
#   max-num-seqs 2*conc, capture sizes [1,2,4,8] + 12..2*conc step 4.
set -euo pipefail
CONC=$1; SERVER_PORT=$2; ENGINE_PORT=$3

if (( CONC >= 48 )); then K=3; AL=2.99; else K=4; AL=3.33; fi
sizes='[1,2,4,8'
for ((s = 12; s <= CONC * 2; s += 4)); do sizes+=",$s"; done
sizes+=']'

export CONC ENGINE_PORT
export IN_OUT="${IN_OUT:-70000:16 1024:16}"
export PROF_MULT="${PROF_MULT:-1}"
export GPU_MEM_UTIL=0.95 BLOCK_SIZE=64 INDEX_CACHE_DTYPE=fp4 MAX_BATCHED_TOKENS=16384
export MAX_NUM_SEQS=$((2 * CONC)) CUDAGRAPH_SIZES="$sizes"

exec "$HOME/SGLang-benchmarks/BenchFixedLength/ATOM_GLM.sh" --prof \
    --model "$HOME/models/amd/GLM-5.2-MXFP4" --tp 4 --dcp 4 --mtp "$K" --acc-len "$AL" \
    --tag "TP4-DCP4-MTP${K}-c${CONC}" --docker rocm/atom-dev:nightly_202609211553 \
    --port "$SERVER_PORT"
