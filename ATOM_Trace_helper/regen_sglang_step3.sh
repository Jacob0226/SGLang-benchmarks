#!/usr/bin/env bash
# Regenerate SGLANG single-forward step3 breakdowns (old + new PR30575) for
# prefill + decode, using trim_trace_to_forward.py (union CPU+GPU window) so
# analyze_trace.py reports ONE forward with correct call-sites.
set -euo pipefail
cd /home/jacchang/SGLang-benchmarks
BASE=results/amd_GLM-5.2-MXFP4/rocm_sgl-dev-v0.5.15.post1-rocm720-mi35x-20260714
OLD=$BASE/prof-TP4_6PR_PR30575old_INT4
NEW=$BASE/prof-TP4_6PR_PR30575new_INT4
PRE_SB=analysis_GLM5.2/SGLang_vs_ATOM_i8k_conc64_prefill/sidebyside
DEC_SB=analysis_GLM5.2/SGLang_vs_ATOM_i8k_conc64_decode/sidebyside
TMP=/tmp/regen_step3; mkdir -p "$TMP"

# args: <profdir> <phase(prefill|decode)> <tag> <outdir>
gen () {
  local pd=$1 phase=$2 tag=$3 outdir=$4
  local gname goff
  if [ "$phase" = prefill ]; then gname=EXTEND; else gname=DECODE; fi
  local gon="$pd/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-$gname.trace.json.gz"
  goff="$pd/no-cuda-graph/prof_in8192_out16_conc64_p64/in8192_out16_conc64_p64-AMD-TP-0-$gname-NoGraph.trace.json.gz"
  echo ">>> $tag $phase"
  python3 ATOM_Trace_helper/trim_trace_to_forward.py --in "$gon"  --out "$TMP/${tag}_${phase}_on.json.gz"  --phase "$phase" 2>&1 | grep INFO
  python3 ATOM_Trace_helper/trim_trace_to_forward.py --in "$goff" --out "$TMP/${tag}_${phase}_off.json.gz" --phase "$phase" 2>&1 | grep INFO
  python3 analyze_trace.py --graph-on "$TMP/${tag}_${phase}_on.json.gz" \
      --graph-off "$TMP/${tag}_${phase}_off.json.gz" \
      --out "$outdir" --tag "_$tag" 2>&1 | grep -iE "Step 3 written|ERROR|No nn"
}

gen "$OLD" prefill SGLANG_old "$PRE_SB"
gen "$NEW" prefill SGLANG_new "$PRE_SB"
gen "$OLD" decode  SGLANG_old "$DEC_SB"
gen "$NEW" decode  SGLANG_new "$DEC_SB"
echo "DONE"
