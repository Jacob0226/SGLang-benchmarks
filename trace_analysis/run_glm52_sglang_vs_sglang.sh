#!/usr/bin/env bash
# MI355X vs B200, both running SGLang on GLM-5.2, i8k at conc4 and conc64.
# This is README Flow A: same stack on two GPUs, so module names match and
# side_by_side.py --align lcs can pair the rows.
#
# Three asymmetries between the two sides are baked into the data and cannot be
# fixed by re-running the analysis; they are why the numbers below need reading
# with care rather than as a clean apples-to-apples ratio:
#
#  1. Prefill chunk size. Both servers run chunked_prefill_size=16384, but the
#     MI355X tree carries PR32888 (exact chunk fill) and fills 16384 tokens,
#     while stock v0.5.16 on B200 stops at 16317. B200 therefore does 0.41%
#     less work per prefill forward.
#  2. Profile depth. MI355X captured 5 steps per stage, B200 only 2, so the
#     B200 side has exactly one bs=3 prefill forward and --forward-pick median
#     degenerates to "the only one".
#  3. conc64 decode structure trace. B200's graph-OFF conc64 decode ran at
#     bs=40, not bs=64. It only supplies the nn.Module tree, which batch size
#     does not change, so it is matched separately from the timed forward.
set -euo pipefail

BENCH=${BENCH:-$HOME/SGLang-benchmarks}
RES=$BENCH/results
ANA=$BENCH/analysis_GLM5.2

AMD=$RES/amd_GLM-5.2-MXFP4/rocm_sgl-dev-v0.5.16-rocm720-mi35x-20260729/prof-TP4_8PR_QRInt4_MoECSV
NV=$RES/nvidia_GLM-5.2-NVFP4/lmsysorg_sglang-v0.5.16-cu130-runtime/prof-TP4

cd "$BENCH"

# analyze <label> <graph-on> <graph-off> <forward-match> <outdir> <tag>
analyze() {
    echo "---- $1 ----"
    python3 trace_analysis/analyze/sglang_trace.py \
        --graph-on "$2" --graph-off "$3" \
        --forward-match "$4" --forward-pick median \
        --out "$5" --tag "$6" 2>&1 | grep -E "restricted|coverage|ERROR|Error" || true
}

for CONC in 4 64; do
    case $CONC in
    4) AMD_ON_P=8; AMD_OFF_P=4; NV_ON_P=8; NV_OFF_P=4 ;;
    64) AMD_ON_P=128; AMD_OFF_P=64; NV_ON_P=128; NV_OFF_P=64 ;;
    esac

    OUT=$ANA/Docker0729_8PR_SGLang_i8k_conc${CONC}
    SBS=$OUT/sidebyside
    mkdir -p "$SBS"

    AMD_ON=$AMD/prof_in8192_out16_conc${CONC}_p${AMD_ON_P}/in8192_out16_conc${CONC}_p${AMD_ON_P}-AMD-TP-0
    AMD_OFF=$AMD/no-cuda-graph/prof_in8192_out16_conc${CONC}_p${AMD_OFF_P}/in8192_out16_conc${CONC}_p${AMD_OFF_P}-AMD-TP-0
    NV_ON=$NV/prof_in8192_out16_conc${CONC}_p${NV_ON_P}/in8192_out16_conc${CONC}_p${NV_ON_P}-NV-TP-0
    NV_OFF=$NV/no-cuda-graph/prof_in8192_out16_conc${CONC}_p${NV_OFF_P}/in8192_out16_conc${CONC}_p${NV_OFF_P}-NV-TP-0

    echo "======== conc${CONC} PREFILL ========"
    analyze "MI355X prefill" "${AMD_ON}-EXTEND.trace.json.gz" "${AMD_OFF}-EXTEND-NoGraph.trace.json.gz" \
        "EXTEND bs=3" "$SBS" "_MI355X_prefill"
    analyze "B200 prefill" "${NV_ON}-EXTEND.trace.json.gz" "${NV_OFF}-EXTEND-NoGraph.trace.json.gz" \
        "EXTEND bs=3" "$SBS" "_B200_prefill"

    # "DECODE bs=" and not "DECODE bs=$CONC": the same match is applied to the
    # graph-OFF trace, and B200's conc64 no-graph run decoded at bs=40. Each
    # trace holds exactly one decode batch size, so the looser match still picks
    # the intended forward on both sides.
    echo "======== conc${CONC} DECODE ========"
    analyze "MI355X decode" "${AMD_ON}-DECODE.trace.json.gz" "${AMD_OFF}-DECODE-NoGraph.trace.json.gz" \
        "DECODE bs=" "$SBS" "_MI355X_decode"
    analyze "B200 decode" "${NV_ON}-DECODE.trace.json.gz" "${NV_OFF}-DECODE-NoGraph.trace.json.gz" \
        "DECODE bs=" "$SBS" "_B200_decode"

    for PHASE in prefill decode; do
        echo "---- conc${CONC} $PHASE buckets (step3: decoder layers) ----"
        python3 trace_analysis/compare/glm52_buckets.py --phase "$PHASE" \
            --out "$OUT/cmp_glm52_${PHASE}_MI355XvsB200.csv" \
            --src MI355X "$SBS/step3_layer_breakdown_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step3_layer_breakdown_B200_${PHASE}.xlsx"

        # step1 scope as well, because step3 depends on matching kernels between
        # the timed graph-ON forward and the graph-OFF module tree by name. That
        # breaks for conc64 decode: B200's no-graph run was bs=40 and cuBLAS
        # picks shape-specialised nvjet tiles, so the bs=64 dense GEMMs find no
        # call site and step3 reports 1.11 ms against MI355X's 5.73 ms. step1
        # needs no attribution and puts the same bucket at 6.05 vs 5.83 ms.
        # Where the two scopes agree (everything except that one cell) either
        # can be quoted; where they disagree, step1 is the ground truth.
        echo "---- conc${CONC} $PHASE buckets (step1: whole forward) ----"
        python3 trace_analysis/compare/glm52_buckets.py --phase "$PHASE" \
            --out "$OUT/cmp_glm52_${PHASE}_MI355XvsB200_step1scope.csv" \
            --src MI355X "$SBS/step1_kernel_stats_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step1_kernel_stats_B200_${PHASE}.xlsx"

        # Call order, each side in its own, which is the layout that survives
        # GLM-5.2's heterogeneous layers. Σ_ms here is per row (Avg x that row's
        # own LaunchCnt), so the Indexer call sites that run on 21 of 78 layers
        # and the MoE shared expert that runs on 1 of 75 are each counted their
        # own number of times. The --align lcs workbook cannot do that: it keeps
        # one LayerCount per section, taken from the section's first row, and
        # its Subtotal sums per-call averages -- which inflated MI355X's conc4
        # decode to 14.94 ms against a real 10.84 ms, and flipped the sign of
        # the B200 comparison. Use this one for magnitudes.
        echo "---- conc${CONC} $PHASE call-order workbook ----"
        python3 trace_analysis/compare/side_by_side.py --align none \
            --out "$OUT/callorder_i8k_conc${CONC}_MI355X_vs_B200_${PHASE}.xlsx" \
            --src MI355X "$SBS/step3_layer_breakdown_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step3_layer_breakdown_B200_${PHASE}.xlsx" \
            --summary-csv "$OUT/cmp_glm52_${PHASE}_MI355XvsB200.csv" \
            --title "GLM-5.2 ${PHASE} i8k conc${CONC} — MI355X (MXFP4, aiter/triton DSA, 8PR @ d1616fcdc6) vs B200 (NVFP4 modelopt, trtllm DSA + flashinfer_trtllm MoE, stock v0.5.16). Both TP=4, page_size=64, fp8_e4m3 KV. ONE forward per side, call order, NOT aligned. Avg_us = one launch; LaunchCnt = launches of that call site in the forward (one per layer, so it equals the layers of the types in LayerType — the Indexer runs on 21 of 78, the MoE shared expert on 1 of 75); Σ_ms = Avg_us x LaunchCnt, so summing the Σ_ms column gives the forward; KernelΣ_ms/KernelCnt = that kernel name's totals in the same forward, repeated on every row sharing the name (do not sum)."

        # named to match analysis_GLM5.2/i1k_conc64/compare_i1k_conc64_MI355X_vs_B200.xlsx
        echo "---- conc${CONC} $PHASE lcs workbook ----"
        python3 trace_analysis/compare/side_by_side.py --align lcs \
            --out "$OUT/compare_i8k_conc${CONC}_MI355X_vs_B200_${PHASE}.xlsx" \
            --src MI355X "$SBS/step3_layer_breakdown_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step3_layer_breakdown_B200_${PHASE}.xlsx" \
            --trace MI355X "${AMD_ON}-$([ "$PHASE" = prefill ] && echo EXTEND || echo DECODE).trace.json.gz" \
            --trace B200 "${NV_ON}-$([ "$PHASE" = prefill ] && echo EXTEND || echo DECODE).trace.json.gz" \
            --title "GLM-5.2 ${PHASE} i8k conc${CONC} — MI355X (MXFP4, aiter/triton DSA, 8PR @ d1616fcdc6) vs B200 (NVFP4 modelopt, trtllm DSA + flashinfer_trtllm MoE, stock v0.5.16). Both TP=4, page_size=64, fp8_e4m3 KV, chunked_prefill_size=16384. ONE forward per side, the median of that side's captured forwards. MI355X prefill fills 16384 tokens (PR32888 exact chunk fill), B200 fills 16317."
    done

    # Bucket tables compare Sigma_kernel, which on its own cannot say whether a
    # forward is slow because its kernels are slow or because they do not
    # overlap. At conc64 decode the two sides' Sigma_kernel are within 2% while
    # wall time differs by 25%, so this measurement is what the decode
    # conclusion actually rests on.
    {
        for SIDE in MI355X B200; do
            [ "$SIDE" = MI355X ] && BASE=$AMD_ON || BASE=$NV_ON
            echo "######## conc${CONC} PREFILL bs=3 -- $SIDE ########"
            python3 trace_analysis/diagnostics/forward_overlap.py \
                "${BASE}-EXTEND.trace.json.gz" --match "EXTEND bs=3"
            echo "######## conc${CONC} DECODE -- $SIDE ########"
            python3 trace_analysis/diagnostics/forward_overlap.py \
                "${BASE}-DECODE.trace.json.gz" --match "DECODE bs="
        done
    } > "$OUT/forward_wall_overlap.txt" 2>&1

    # all-reduce is prefill's largest single gap, and a collective's duration on
    # one rank is communication plus however long that rank waits for the
    # slowest one. Splitting the two says whether to look at the interconnect or
    # at load balance, so it needs all four TP ranks rather than just TP-0.
    {
        for SIDE in MI355X B200; do
            if [ "$SIDE" = MI355X ]; then
                DIR=$AMD/prof_in8192_out16_conc${CONC}_p${AMD_ON_P}; G="*-AMD-TP-*-EXTEND.trace.json.gz"
            else
                DIR=$NV/prof_in8192_out16_conc${CONC}_p${NV_ON_P}; G="*-NV-TP-*-EXTEND.trace.json.gz"
            fi
            echo "######## conc${CONC} PREFILL bs=3 all-reduce skew -- $SIDE ########"
            # shellcheck disable=SC2086
            python3 trace_analysis/diagnostics/comm_skew_split.py \
                --traces $DIR/$G --stack sglang --phase prefill --match "EXTEND bs=3"
        done
    } > "$OUT/allreduce_skew_prefill.txt" 2>&1

    echo ">>> conc${CONC} done -> $OUT"
done
