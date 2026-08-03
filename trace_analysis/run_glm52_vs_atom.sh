#!/usr/bin/env bash
# GLM-5.2 SGLang-vs-ATOM side-by-side workbooks, i8k conc64, one run per variant.
#
#   ./run_glm52_vs_atom.sh                     # every variant
#   ./run_glm52_vs_atom.sh Docker0729_8PR      # just one
#   FORCE_ATOM=1 ./run_glm52_vs_atom.sh        # re-derive the ATOM reference
#
# The ATOM side is derived ONCE into a cache and copied into every variant, so
# the reference column is byte-identical across all of them. Deriving it per
# variant is what let the three old scripts disagree.
#
# ATOM's forward is selected by shape and then by median, NOT pinned to a label.
# An ATOM prefill is labelled with its context lengths -- prefill[bs=3 tok=16384
# ctx=[8063, 7153, 1168]] -- and those are unique to one profile, so a pinned
# label stops matching the moment the profile is regenerated. Matching on
# "bs=3 tok=16384" leaves 44 candidates whose middle 50% sit in 640.4-643.4 ms;
# --pick median takes 642.1 ms from that. Averaging instead would report 629.8 ms,
# because one candidate is a 128.5 ms fragment at a trace boundary.
#
# Every step1/step3 is restricted to ONE forward so the call-order table at the
# top of the workbook and the bucket summary at the bottom share a scale. Without
# it step1 averages each kernel over the WHOLE trace, blending the 7.6k-token and
# 16.4k-token prefill chunks (the TP all-reduce came out as 713 us/call instead
# of the real 861 us/call for the bs=3 forward).
set -euo pipefail

BENCH=${BENCH:-$HOME/SGLang-benchmarks}
RES=$BENCH/results/amd_GLM-5.2-MXFP4
ANA=$BENCH/analysis_GLM5.2

TAG_0714=rocm_sgl-dev-v0.5.15.post1-rocm720-mi35x-20260714
TAG_0729=rocm_sgl-dev-v0.5.16-rocm720-mi35x-20260729
ATOM_DIR=$RES/rocm_atom-dev-nightly_202607221602/prof-TP4_CIcfg

ATOM_CACHE=$ANA/.atom_reference_i8k_conc64
ATOM_TIME=$ATOM_DIR/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0.trace.json.gz
ATOM_STRUCT=$ATOM_DIR/no-cuda-graph/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-NoGraph.trace.json.gz

# variant -> sglang image tag : graph-ON profile : graph-OFF profile : output prefix : phases
#
# pf16 exists because the original 8PR prefill profile holds a single usable
# 16384-token forward, so --forward-pick median has nothing to choose from. Its
# graph-OFF trace is borrowed from the non-pf16 run: it only supplies the module
# tree, which the extra steps do not change.
variant_spec() {
    case $1 in
    6PR_old)            echo "$TAG_0714:prof-TP4_6PR_QRInt4_MoECSV:prof-TP4_6PR_QRInt4_MoECSV:6PR_old_SGLang_vs_ATOM_i8k_conc64:prefill decode" ;;
    6PR_new)            echo "$TAG_0714:prof-TP4_6PR-30575new_QRInt4_MoECSV:prof-TP4_6PR-30575new_QRInt4_MoECSV:6PR_new_SGLang_vs_ATOM_i8k_conc64:prefill decode" ;;
    Docker0714_8PR)     echo "$TAG_0714:prof-TP4_8PR_QRInt4_MoECSV:prof-TP4_8PR_QRInt4_MoECSV:Docker0714_8PR_SGLang_vs_ATOM_i8k_conc64:prefill decode" ;;
    Docker0729_8PR)     echo "$TAG_0729:prof-TP4_8PR_QRInt4_MoECSV:prof-TP4_8PR_QRInt4_MoECSV:Docker0729_8PR_SGLang_vs_ATOM_i8k_conc64:prefill decode" ;;
    Docker0729_8PR_pf16) echo "$TAG_0729:prof-TP4_8PR_QRInt4_MoECSV_pf16:prof-TP4_8PR_QRInt4_MoECSV:Docker0729_8PR_pf16_SGLang_vs_ATOM_i8k_conc64:prefill" ;;
    *) echo "unknown variant: $1" >&2; return 1 ;;
    esac
}

ALL_VARIANTS=(6PR_old 6PR_new Docker0714_8PR Docker0729_8PR Docker0729_8PR_pf16)

# Column semantics, repeated in every workbook so a stray file still explains
# itself. The Σ_ms warning matters: KernelΣ_ms repeats on every row sharing a
# kernel name, so summing that column double counts, while Σ_ms sums to the
# forward.
COLDOC="Avg_us = one launch; LaunchCnt = launches of that call site in the forward (SGLANG: one per layer, so it equals the layers of the types in LayerType); Σ_ms = Avg_us x LaunchCnt, so summing Σ_ms gives the forward; KernelΣ_ms/KernelCnt = that kernel name's totals in the same forward, repeated on every row sharing the name (do not sum)."

cd "$BENCH"

# --- ATOM reference, derived once ---------------------------------------------
build_atom_reference() {
    mkdir -p "$ATOM_CACHE"
    echo "======== ATOM reference (shared by every variant) ========"
    for f in "$ATOM_TIME" "$ATOM_STRUCT"; do
        [ -f "$f" ] || { echo "missing ATOM trace: $f" >&2; exit 1; }
    done

    echo "---- ATOM prefill (median of the bs=3 tok=16384 forwards) ----"
    python3 trace_analysis/analyze/atom_trace.py --phase prefill --pick median \
        --forward-match "bs=3 tok=16384" --struct-forward-match "bs=3 tok=16384" \
        --time-trace "$ATOM_TIME" --struct-trace "$ATOM_STRUCT" \
        --out "$ATOM_CACHE" --tag _ATOM_prefill 2>&1 | grep -E "candidate|using forward|ERROR" || true

    # ATOM's no-cuda-graph trace only recorded prefill[...] forwards, so the
    # decode module tree is necessarily borrowed from a prefill-shaped segment
    # (bounded by consecutive model.layers.0.* annotations). Timings still come
    # from the decode window of the graph-ON trace; only the Section/LeafModule
    # labels come from there.
    echo "---- ATOM decode ----"
    python3 trace_analysis/analyze/atom_trace.py --phase decode --pick median \
        --forward-match "bs=64 tok=64 d=64" --struct-forward-match "model.layers.0" \
        --time-trace "$ATOM_TIME" --struct-trace "$ATOM_STRUCT" \
        --out "$ATOM_CACHE" --tag _ATOM_decode 2>&1 | grep -E "candidate|using forward|ERROR" || true
}

atom_xlsx() { echo "$ATOM_CACHE/step3_layer_breakdown_ATOM_$1.xlsx"; }

# --- one variant ---------------------------------------------------------------
run_variant() {
    local name=$1 spec sgl_tag on_prof off_prof outbase phases
    spec=$(variant_spec "$name") || exit 1
    IFS=: read -r sgl_tag on_prof off_prof outbase phases <<<"$spec"

    local ON=$RES/$sgl_tag/$on_prof/prof_in8192_out16_conc64_p128
    local OFF=$RES/$sgl_tag/$off_prof/no-cuda-graph/prof_in8192_out16_conc64_p64

    echo "################ $name ################"
    for phase in $phases; do
        local UP OUT match
        case $phase in
        prefill) UP=EXTEND; match="EXTEND bs=3" ;;
        decode)  UP=DECODE; match="DECODE bs=64" ;;
        esac
        OUT=$ANA/${outbase}_${phase}

        local on_trace=$ON/in8192_out16_conc64_p128-AMD-TP-0-$UP.trace.json.gz
        local off_trace=$OFF/in8192_out16_conc64_p64-AMD-TP-0-$UP-NoGraph.trace.json.gz
        for f in "$on_trace" "$off_trace"; do
            [ -f "$f" ] || { echo "missing trace: $f" >&2; exit 1; }
        done

        mkdir -p "$OUT/sidebyside"
        cp -f "$(atom_xlsx "$phase")" "$OUT/sidebyside/step3_layer_breakdown_ATOM.xlsx"

        echo "---- $name $phase: SGLang ----"
        python3 trace_analysis/analyze/sglang_trace.py \
            --graph-on "$on_trace" --graph-off "$off_trace" \
            --forward-match "$match" --forward-pick median \
            --out "$OUT/sidebyside" --tag _SGLANG \
            > "$OUT/analyze_sglang_$phase.log" 2>&1 || {
                echo "analyze/sglang_trace.py failed, see $OUT/analyze_sglang_$phase.log" >&2; exit 1; }
        grep -E "restricted|ERROR" "$OUT/analyze_sglang_$phase.log" || true

        local sgl_xlsx=$OUT/sidebyside/step3_layer_breakdown_SGLANG.xlsx
        local atom_ref=$OUT/sidebyside/step3_layer_breakdown_ATOM.xlsx
        local csv=$OUT/cmp_glm52_${phase}_SGLangvsATOM.csv

        # Buckets read the same step3 workbooks as the table above, so there is
        # no second forward-segmentation policy and this step needs no traces.
        echo "---- $name $phase: buckets ----"
        python3 trace_analysis/compare/glm52_buckets.py --phase "$phase" \
            --out "$csv" \
            --src "SGLANG_$name" "$sgl_xlsx" \
            --src ATOM "$atom_ref"

        echo "---- $name $phase: call-order workbook ----"
        python3 trace_analysis/compare/side_by_side.py --align none \
            --out "$OUT/callorder_i8k_conc64_${name}_vs_ATOM_${phase}.xlsx" \
            --src "SGLANG_$name" "$sgl_xlsx" \
            --src ATOM "$atom_ref" \
            --summary-csv "$csv" \
            --title "GLM-5.2 $phase i8k conc64 — SGLang $name vs ATOM, call order, ONE forward per side, each the median of its own run; NOT aligned. $COLDOC"

        echo ">>> $name $phase done -> $OUT"
    done
}

VARIANTS=("$@")
[ ${#VARIANTS[@]} -gt 0 ] || VARIANTS=("${ALL_VARIANTS[@]}")

if [ -n "${FORCE_ATOM:-}" ] || [ ! -f "$(atom_xlsx prefill)" ] || [ ! -f "$(atom_xlsx decode)" ]; then
    build_atom_reference
else
    echo "reusing ATOM reference in $ATOM_CACHE (FORCE_ATOM=1 to rebuild)"
fi

for v in "${VARIANTS[@]}"; do
    run_variant "$v"
done

echo "done."
