#!/usr/bin/env bash
# MI355X vs B200, both running SGLang on GLM-5.3-Flash, i8k at conc4 and conc64.
# README Flow A: same stack on two GPUs, so module names match and
# side_by_side.py --align lcs can pair the rows.
#
# Asymmetries baked into the data. None can be fixed by re-running the analysis.
#
#  1. Quantization differs, and this is the big one. MI355X serves
#     amd/GLM-5.3-Flash-Quark-MXFP4 (Quark MXFP4 weights with the attention,
#     mHC and MTP-layer experts left at block FP8) against B200's
#     RadixArk/GLM-5.3-Flash-NVFP4 (ModelOpt W4A4). So this is not one model on
#     two GPUs -- it is two checkpoints. Read the bucket tables as "where does
#     each platform's time go", not as a per-kernel efficiency ratio.
#
#  2. KV dtype and DSA backend differ, because on ROCm they are one decision and
#     not two: MI355X runs bfloat16 KV + TileLang DSA (FP8 KV crashes the
#     TileLang path on a chunked-prefill continuation -- see the [rocm-kv] note
#     in GLM.sh), B200 runs fp8_e4m3 KV + trtllm DSA. The sparse-MLA bucket
#     therefore compares a bf16 kernel against an fp8 one.
#
#  3. Prefill chunk size. Both set chunked_prefill_size=16384, but the MI355X
#     tree carries the exact-chunk-fill work and fills 16384 tokens while the
#     B200 side stops at 16317, so B200 does 0.41% less work per prefill
#     forward. Same asymmetry as the GLM-5.2 comparison.
#
#  4. B200 kept only TP-0 traces, so the all-reduce skew split (which needs
#     every rank to separate transport from load imbalance) runs for MI355X
#     only.
#
# Deliberately NOT using prof-Fixed-MTP-NVFP4-TP4, the B200 directory with
# speculative decoding on. Its decode forwards are step[VERIFY bs=4] plus
# draft/draft_extend, i.e. 4x(1+6)=28 tokens per forward against the MI355X
# run's 4 tokens in step[DECODE bs=4]; a bucket table over those two would
# compare different amounts of work and read as a 7x B200 win. The non-MTP
# directory has matching step[EXTEND bs=3] / step[DECODE bs=64] annotations in
# both graph-on and graph-off traces, verified 2026-09-22. For a spec-decode
# comparison the MI355X side has to be re-profiled with --mtp (PR #39778 and
# #39779 enable it on ROCm); MTP_B200 below points at the MTP directory so that
# run has somewhere to land.
#
# Buckets come from compare/glm53_buckets.py, not glm52_buckets.py. The GLM-5.2
# rules are name-only and mis-file this model badly: they left 25-29% of both
# sides in a catch-all (the 34 KDA layers and the mHC kernels have no bucket
# there) and read sparse-MLA as 0.137 ms on MI355X against 33.8 ms on B200,
# because TileLang calls its generated kernel `main_kernel` and nothing matched
# it. The GLM-5.3 rules classify on the call site and layer type first, which
# takes unclassified to 0.0% on both sides and flips that sparse-MLA comparison
# to 1.17 vs 0.47 ms. Each run prints its own unclassified share -- check it.
set -euo pipefail

BENCH=${BENCH:-$HOME/SGLang-benchmarks}
RES=$BENCH/results
ANA=$BENCH/analysis_GLM5.3

AMD=$RES/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/prof-Fixed-MXFP4-TP4-PRstack
NV=$RES/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-NVFP4-TP4
MTP_B200=$RES/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-MTP-NVFP4-TP4

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
    4) ON_P=8; OFF_P=4 ;;
    64) ON_P=128; OFF_P=64 ;;
    esac

    OUT=$ANA/Docker0914_10PR_SGLang_i8k_conc${CONC}
    SBS=$OUT/sidebyside
    mkdir -p "$SBS"

    AMD_ON=$AMD/prof_in8192_out16_conc${CONC}_p${ON_P}/in8192_out16_conc${CONC}_p${ON_P}-AMD-TP-0
    AMD_OFF=$AMD/no-cuda-graph/prof_in8192_out16_conc${CONC}_p${OFF_P}/in8192_out16_conc${CONC}_p${OFF_P}-AMD-TP-0
    NV_ON=$NV/prof_in8192_out16_conc${CONC}_p${ON_P}/in8192_out16_conc${CONC}_p${ON_P}-NV-TP-0
    NV_OFF=$NV/no-cuda-graph/prof_in8192_out16_conc${CONC}_p${OFF_P}/in8192_out16_conc${CONC}_p${OFF_P}-NV-TP-0

    echo "======== conc${CONC} PREFILL ========"
    analyze "MI355X prefill" "${AMD_ON}-EXTEND.trace.json.gz" "${AMD_OFF}-EXTEND-NoGraph.trace.json.gz" \
        "EXTEND bs=3" "$SBS" "_MI355X_prefill"
    analyze "B200 prefill" "${NV_ON}-EXTEND.trace.json.gz" "${NV_OFF}-EXTEND-NoGraph.trace.json.gz" \
        "EXTEND bs=3" "$SBS" "_B200_prefill"

    # "DECODE bs=" rather than "DECODE bs=$CONC": the same match is applied to
    # the graph-OFF trace too, and each trace holds exactly one decode batch
    # size. Verified here that all four are bs=$CONC, unlike the GLM-5.2 run
    # where B200's no-graph conc64 decode came out at bs=40.
    echo "======== conc${CONC} DECODE ========"
    analyze "MI355X decode" "${AMD_ON}-DECODE.trace.json.gz" "${AMD_OFF}-DECODE-NoGraph.trace.json.gz" \
        "DECODE bs=" "$SBS" "_MI355X_decode"
    analyze "B200 decode" "${NV_ON}-DECODE.trace.json.gz" "${NV_OFF}-DECODE-NoGraph.trace.json.gz" \
        "DECODE bs=" "$SBS" "_B200_decode"

    TITLE_TAIL="MI355X (amd/GLM-5.3-Flash-Quark-MXFP4, AITER MoE + TileLang DSA, bf16 KV, main@242d8a70c0 + 10 Day-0 PRs) vs B200 (RadixArk/GLM-5.3-Flash-NVFP4 modelopt_fp4, trtllm DSA + flashinfer_trtllm MoE, fp8_e4m3 KV, stock v0.5.20). Both TP=4. Two checkpoints, not one model on two GPUs. MI355X prefill fills 16384 tokens, B200 16317."

    for PHASE in prefill decode; do
        echo "---- conc${CONC} $PHASE buckets (step3: decoder layers) ----"
        python3 trace_analysis/compare/glm53_buckets.py --phase "$PHASE" \
            --out "$OUT/cmp_glm53_${PHASE}_MI355XvsB200.csv" \
            --src MI355X "$SBS/step3_layer_breakdown_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step3_layer_breakdown_B200_${PHASE}.xlsx"

        # step1 scope too: step3 attributes kernels by matching the timed
        # graph-ON forward against the graph-OFF module tree by name, which can
        # drop shape-specialised GEMMs. Where the two scopes agree either can be
        # quoted; where they disagree, step1 is the ground truth.
        echo "---- conc${CONC} $PHASE buckets (step1: whole forward) ----"
        python3 trace_analysis/compare/glm53_buckets.py --phase "$PHASE" \
            --out "$OUT/cmp_glm53_${PHASE}_MI355XvsB200_step1scope.csv" \
            --src MI355X "$SBS/step1_kernel_stats_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step1_kernel_stats_B200_${PHASE}.xlsx"

        # Call order, each side in its own. Σ_ms is per row (Avg x that row's own
        # LaunchCnt), so a call site that runs on 11 of 45 layers is counted 11
        # times -- which matters more here than on GLM-5.2, because this model's
        # layers are two different kinds.
        echo "---- conc${CONC} $PHASE call-order workbook ----"
        python3 trace_analysis/compare/side_by_side.py --align none \
            --out "$OUT/callorder_i8k_conc${CONC}_MI355X_vs_B200_${PHASE}.xlsx" \
            --src MI355X "$SBS/step3_layer_breakdown_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step3_layer_breakdown_B200_${PHASE}.xlsx" \
            --summary-csv "$OUT/cmp_glm53_${PHASE}_MI355XvsB200.csv" \
            --title "GLM-5.3-Flash ${PHASE} i8k conc${CONC} — ${TITLE_TAIL} ONE forward per side, call order, NOT aligned."

        echo "---- conc${CONC} $PHASE lcs workbook ----"
        python3 trace_analysis/compare/side_by_side.py --align lcs \
            --out "$OUT/compare_i8k_conc${CONC}_MI355X_vs_B200_${PHASE}.xlsx" \
            --src MI355X "$SBS/step3_layer_breakdown_MI355X_${PHASE}.xlsx" \
            --src B200 "$SBS/step3_layer_breakdown_B200_${PHASE}.xlsx" \
            --trace MI355X "${AMD_ON}-$([ "$PHASE" = prefill ] && echo EXTEND || echo DECODE).trace.json.gz" \
            --trace B200 "${NV_ON}-$([ "$PHASE" = prefill ] && echo EXTEND || echo DECODE).trace.json.gz" \
            --title "GLM-5.3-Flash ${PHASE} i8k conc${CONC} — ${TITLE_TAIL} ONE forward per side, the median of that side's captured forwards."
    done

    # Bucket tables compare Σ_kernel, which cannot say whether a forward is slow
    # because its kernels are slow or because they do not overlap. Wall vs Σ is
    # what tells those apart.
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

    # MI355X only: a collective's duration on one rank is transport plus however
    # long that rank waited for the slowest one, and splitting those needs all
    # four ranks. The B200 directory kept TP-0 alone.
    {
        echo "######## conc${CONC} PREFILL bs=3 all-reduce skew -- MI355X ########"
        # shellcheck disable=SC2086
        python3 trace_analysis/diagnostics/comm_skew_split.py \
            --traces $AMD/prof_in8192_out16_conc${CONC}_p${ON_P}/*-AMD-TP-*-EXTEND.trace.json.gz \
            --stack sglang --phase prefill --match "EXTEND bs=3"
        echo
        echo "######## B200: skipped -- only TP-0 was kept in $NV ########"
    } > "$OUT/allreduce_skew_prefill.txt" 2>&1

    echo ">>> conc${CONC} done -> $OUT"
done

echo
echo "MTP (B200 only, no MI355X counterpart yet): $MTP_B200"
echo "  decode forwards there are step[VERIFY bs=N] + draft/draft_extend."
