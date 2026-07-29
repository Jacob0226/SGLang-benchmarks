#!/usr/bin/env bash
# Regenerate the GLM-5.2 SGLang-vs-ATOM side-by-side workbooks (i8k conc64).
#
# Every step1/step3 is now restricted to ONE forward pass via --forward-match, so
# the call-order table at the top of the callorder workbook is on the same scale
# as the sglang_vs_atom_glm52 per-forward bucket summary at the bottom:
#
#   SGLANG  prefill : step[EXTEND bs=3 toks=16368] (old) / toks=16332 (new)
#   ATOM    prefill : prefill[bs=3 tok=16384 ctx=[8063, 7153, 1168]]  (same forward
#                     sglang_vs_atom_glm52 picked as median-of-55)
#   SGLANG  decode  : step[DECODE bs=64]
#   ATOM    decode  : decode[bs=64 tok=64 d=64]                        (median)
#
# Without --forward-match, step1 averages each kernel over the WHOLE trace, which
# blends the 7.6k-token and 16.4k-token prefill chunks (e.g. the TP all-reduce came
# out as 713 us/call instead of the real 861 us/call for the bs=3 forward).
set -euo pipefail

BENCH=${BENCH:-$HOME/SGLang-benchmarks}
RES=$BENCH/results/amd_GLM-5.2-MXFP4
SGL_TAG=rocm_sgl-dev-v0.5.15.post1-rocm720-mi35x-20260714
ATOM_TAG=rocm_atom-dev-nightly_202607221602

OLD=$RES/$SGL_TAG/prof-TP4_6PR_PR30575old_INT4
NEW=$RES/$SGL_TAG/prof-TP4_6PR_PR30575new_INT4
ATOM=$RES/$ATOM_TAG/prof-TP4_CIcfg

PRE=$BENCH/analysis_GLM5.2/SGLang_vs_ATOM_i8k_conc64_prefill
DEC=$BENCH/analysis_GLM5.2/SGLang_vs_ATOM_i8k_conc64_decode

# ATOM prefill forward pinned to the exact label sglang_vs_atom_glm52 reported, so both
# halves of the workbook describe the same forward.
ATOM_PRE_FWD=${ATOM_PRE_FWD:-"ctx=[8063, 7153, 1168]"}
ATOM_DEC_FWD=${ATOM_DEC_FWD:-"bs=64 tok=64 d=64"}

cd "$BENCH"

echo "################ PREFILL ################"
for v in old new; do
  case $v in old) D=$OLD;; new) D=$NEW;; esac
  echo "---- SGLANG_$v prefill ----"
  python3 trace_analysis/sglang/analyze_trace.py \
    --graph-on  "$D/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-EXTEND.trace.json.gz" \
    --graph-off "$D/no-cuda-graph/prof_in8192_out16_conc64_p64/in8192_out16_conc64_p64-AMD-TP-0-EXTEND-NoGraph.trace.json.gz" \
    --forward-match "bs=3" \
    --out "$PRE/sidebyside" --tag "_SGLANG_$v" >/dev/null
done

echo "---- ATOM prefill ----"
python3 trace_analysis/atom/analyze_atom_trace.py --phase prefill --pick median \
  --forward-match "$ATOM_PRE_FWD" --struct-forward-match "bs=3 tok=16384" \
  --time-trace   "$ATOM/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0.trace.json.gz" \
  --struct-trace "$ATOM/no-cuda-graph/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-NoGraph.trace.json.gz" \
  --out "$PRE/sidebyside" --tag _ATOM >/dev/null

echo "################ DECODE ################"
for v in old new; do
  case $v in old) D=$OLD;; new) D=$NEW;; esac
  echo "---- SGLANG_$v decode ----"
  python3 trace_analysis/sglang/analyze_trace.py \
    --graph-on  "$D/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-DECODE.trace.json.gz" \
    --graph-off "$D/no-cuda-graph/prof_in8192_out16_conc64_p64/in8192_out16_conc64_p64-AMD-TP-0-DECODE-NoGraph.trace.json.gz" \
    --forward-match "DECODE bs=64" \
    --out "$DEC/sidebyside" --tag "_SGLANG_$v" >/dev/null
done

# ATOM's no-cuda-graph trace only recorded prefill[...] forwards, so the decode
# module tree is necessarily borrowed from a prefill-shaped segment (bounded by
# consecutive model.layers.0.* annotations). Timings still come from the decode
# window of the graph-ON trace; only Section/LeafModule labels come from there.
echo "---- ATOM decode ----"
python3 trace_analysis/atom/analyze_atom_trace.py --phase decode --pick median \
  --forward-match "$ATOM_DEC_FWD" --struct-forward-match "model.layers.0" \
  --time-trace   "$ATOM/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0.trace.json.gz" \
  --struct-trace "$ATOM/no-cuda-graph/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-NoGraph.trace.json.gz" \
  --out "$DEC/sidebyside" --tag _ATOM >/dev/null

echo "################ CALL-ORDER WORKBOOKS ################"
python3 trace_analysis/compare/callorder_sidebyside.py \
  --out "$PRE/sidebyside/callorder_prefill_SGLang_vs_ATOM.xlsx" \
  --src SGLANG_old "$PRE/sidebyside/step3_layer_breakdown_SGLANG_old.xlsx" \
  --src ATOM       "$PRE/sidebyside/step3_layer_breakdown_ATOM.xlsx" \
  --summary-csv    "$PRE/cmp_glm52_prefill_OLDvsATOM.csv" \
  --title "GLM-5.2 prefill — call order, ONE forward per side (SGLANG step[EXTEND bs=3 toks=16368] | ATOM prefill[bs=3 tok=16384]); NOT aligned. Avg_us = one launch; LaunchCnt = launches of that call site in the forward (SGLANG: one per layer, so it equals the layers of the types in LayerType); Σ_ms = Avg_us x LaunchCnt; KernelΣ_ms/KernelCnt = that kernel name's totals in the same forward, repeated on every row sharing the name (do not sum). SGLANG_old_30575=711.1ms | ATOM=641.1ms"

python3 trace_analysis/compare/callorder_sidebyside.py \
  --out "$DEC/sidebyside/callorder_decode_SGLang_vs_ATOM.xlsx" \
  --src SGLANG_old "$DEC/sidebyside/step3_layer_breakdown_SGLANG_old.xlsx" \
  --src ATOM       "$DEC/sidebyside/step3_layer_breakdown_ATOM.xlsx" \
  --summary-csv    "$DEC/cmp_glm52_decode_OLDvsATOM.csv" \
  --title "GLM-5.2 decode — call order, ONE forward per side (SGLANG step[DECODE bs=64] | ATOM decode[bs=64 tok=64 d=64]); NOT aligned. Avg_us = one launch; LaunchCnt = launches of that call site in the forward (SGLANG: one per layer, so it equals the layers of the types in LayerType); Σ_ms = Avg_us x LaunchCnt; KernelΣ_ms/KernelCnt = that kernel name's totals in the same forward, repeated on every row sharing the name (do not sum). SGLANG_old_30575=27.8ms | ATOM=25.0ms"

echo "done."
