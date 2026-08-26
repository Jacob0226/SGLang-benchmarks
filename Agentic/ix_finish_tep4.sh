#!/usr/bin/env bash
# Finish the board's full TEP4 concurrency list under GLM.sh's result layout.
#
# conc 1/8/16 already ran under the old ix-agentx/prstack path, so move them
# into the tagged directory first; the driver then skips them and only fills in
# the gaps (2, 4, 10, 12).
set -u

OLD=/home/jacchang/SGLang-benchmarks/results/ix-agentx/prstack
NEW=/home/jacchang/SGLang-benchmarks/results/amd_GLM-5.2-MXFP4/rocm_sgl-dev-v0.5.18-rocm720-mi35x-20260824/bench-Agentic-TP4_EP4
CONC16_JSON="$OLD/glm5.2_tp4_conc16_kvdram-hicache_spec-mtp/glm5.2_tp4_conc16_kvdram-hicache_spec-mtp_fp4_sglang_tp4-pp1-dcp1-pcp1-ep4-dpafalse_disagg-false_spec-mtp_conc16_local-mi355x.json"

while [ ! -f "$CONC16_JSON" ]; do sleep 60; done
echo "$(date -Is) conc=16 done, migrating completed points into $NEW"

sleep 120
pkill -9 -f 'sglang.launch_server' 2>/dev/null
pkill -9 -f 'sglang::' 2>/dev/null
sleep 30

mkdir -p "$NEW"
for c in 1 8 16; do
    d="$OLD/glm5.2_tp4_conc${c}_kvdram-hicache_spec-mtp"
    [ -d "$d" ] && mv "$d" "$NEW/"
done
[ -f "$OLD/sweep.log" ] && mv "$OLD/sweep.log" "$NEW/sweep-conc1-8-16.log"

exec /home/jacchang/SGLang-benchmarks/Agentic/ix_agentx_glm52.sh \
    --arm tep4 --conc "1 2 4 8 10 12 16" --dsa triton \
    --env SGLANG_DSA_FP8_PROJ_GEMM=1 --env SGLANG_USE_MXFP4_MLA_BMM=1
