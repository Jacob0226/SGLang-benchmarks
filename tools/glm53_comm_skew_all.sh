#!/usr/bin/env bash
# Split the TP collective into transport vs waiting, on every rank, both
# platforms, both phases. Now possible on B200 too: its traces were re-uploaded
# with all four ranks (the first upload was TP-0 only).
#
# The question this answers: a single all-reduce launch of 2471-3265 us sat in
# each captured B200 decode forward against a 4.6 us median, and that run's own
# median ITL was 6.08 ms -- too small to contain it. So is that launch real rank
# skew, or an artifact of profiling? comm_skew_split takes the MIN duration
# across ranks for each call as the transport estimate; whatever a rank spends
# above that is waiting for a slower peer.
#
# Output: ~/comm_skew_all.txt
exec > /home/jacchang/comm_skew_all.txt 2>&1
cd /home/jacchang/SGLang-benchmarks || exit 1

A=results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/prof-Fixed-MXFP4-TP4-PRstack
B=results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-NVFP4-TP4
S=trace_analysis/diagnostics/comm_skew_split.py

for conc in 4 64; do
    case $conc in 4) on=8;; 64) on=128;; esac
    for side in MI355X B200; do
        if [ "$side" = MI355X ]; then dir=$A; tag=AMD; else dir=$B; tag=NV; fi
        d=$dir/prof_in8192_out16_conc${conc}_p${on}
        echo "################ conc${conc} DECODE -- $side"
        python3 $S --stack sglang --phase decode --match "DECODE bs=${conc}" \
            --traces $d/*-${tag}-TP-*-DECODE.trace.json.gz
        echo
        echo "################ conc${conc} PREFILL bs=3 -- $side"
        python3 $S --stack sglang --phase prefill --match "EXTEND bs=3" \
            --traces $d/*-${tag}-TP-*-EXTEND.trace.json.gz
        echo
    done
done
