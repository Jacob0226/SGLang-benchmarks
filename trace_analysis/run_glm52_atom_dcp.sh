#!/usr/bin/env bash
# GLM-5.2 ATOM decode-step breakdown for the InferenceX conc>=16 arm
# (TP4 + DCP4 + MTP, rocm/atom-dev:nightly_202609211553), produced by
# tools/atom_dcp_prof.sh.
#
#   ./run_glm52_atom_dcp.sh                 # every conc x shape
#   ./run_glm52_atom_dcp.sh 32 70000        # one
#
# Only full-batch verify steps are candidates: --forward-match pins
# "bs=C tok=C*(K+1) d=C", so the ramp-down tail (bs=30/32 ...) is excluded and
# --pick median takes the steady-state step from what is left. The labels come
# from the matching eager_decode[ verify forward of the no-cuda-graph trace.
set -euo pipefail

BENCH=${BENCH:-$HOME/SGLang-benchmarks}
RES=$BENCH/results/amd_GLM-5.2-MXFP4/rocm_atom-dev-nightly_202609211553
OUT=$BENCH/Analysis/analysis_GLM5.2/ATOM_TP4_DCP4_MTP_0921

draft_tokens() { if (( $1 >= 48 )); then echo 3; else echo 4; fi; }

run_one() {
    local c=$1 in=$2 k; k=$(draft_tokens "$c")
    local dir=$RES/prof-TP4-DCP4-MTP${k}-c${c} cfg=in${in}_out16_conc${c}_p${c}
    local time=$dir/prof_${cfg}/${cfg}-AMD-TP-0.trace.json.gz
    local struct=$dir/no-cuda-graph/prof_${cfg}/${cfg}-AMD-TP-0-NoGraph.trace.json.gz
    local tok=$((c * (k + 1)))
    local match="bs=${c} tok=${tok} d=${c}"
    [ -f "$time" ] || { echo "[skip] missing $time"; return 0; }
    local args=(--time-trace "$time")
    if [ -f "$struct" ]; then
        args+=(--struct-trace "$struct" --struct-forward-match "eager_decode[bs=${c} tok=${tok} ")
    else
        echo "[warn] no struct trace for $cfg; kernels stay unlabeled"
    fi
    echo ">>> i${in} conc${c} (MTP${k}) match='$match'"
    python3 "$BENCH/trace_analysis/analyze/atom_trace.py" --phase decode --pick median \
        --forward-match "$match" "${args[@]}" \
        --out "$OUT/i${in}_conc${c}_decode" --tag "_ATOM_DCP4_MTP${k}"
}

if [ $# -eq 2 ]; then run_one "$1" "$2"; exit; fi
for in in 70000 1024; do for c in 32 64; do run_one "$c" "$in"; done; done
