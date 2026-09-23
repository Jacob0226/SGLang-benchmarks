#!/usr/bin/env bash
# Full per-kernel inventory of ONE decode forward, both platforms, i8k conc4.
# This is the evidence base for "what would speed up MI355X decode", so it wants
# launch counts and call sites, not just totals.
# Output: SGLang-benchmarks/tmp/logs/decode_inventory.txt
OUT=/home/jacchang/SGLang-benchmarks/tmp/logs
mkdir -p "$OUT"
exec > "$OUT/decode_inventory.txt" 2>&1
cd /home/jacchang/SGLang-benchmarks || exit 1

S=analysis_GLM5.3/Docker0914_10PR_steps5_SGLang_i8k_conc4/sidebyside
python3 trace_analysis/diagnostics/dump_kernel_inventory.py \
    "$S/step3_layer_breakdown_MI355X_decode.xlsx" \
    "$S/step3_layer_breakdown_B200_decode.xlsx"
