#!/usr/bin/env bash
# Is the decode all-reduce in these traces communication, or a rank-skew spin?
# One launch of B200's one-shot push all-reduce took 3.27 ms of a ~11 ms conc4
# forward while the median launch was 4.6 us, and 96.7% of its time overlapped
# nothing -- that is a barrier absorbing rank skew, not transport. Re-check at
# conc64 on both platforms before quoting any all-reduce comparison.
# Output: ~/why_conc64.txt
exec > /home/jacchang/why_conc64.txt 2>&1
cd /home/jacchang/SGLang-benchmarks || exit 1

B=results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-NVFP4-TP4
A=results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/prof-Fixed-MXFP4-TP4-PRstack
D=trace_analysis/diagnostics/why_sigma_exceeds_wall.py

echo "================ B200 conc64 decode (all_reduce)"
python3 $D "$B/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-NV-TP-0-DECODE.trace.json.gz" \
    --match "DECODE bs=64" --kernel all_reduce

echo
echo "================ MI355X conc64 decode (cross_device_reduce)"
python3 $D "$A/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-DECODE.trace.json.gz" \
    --match "DECODE bs=64" --kernel cross_device_reduce

echo
echo "================ MI355X conc4 decode (cross_device_reduce)"
python3 $D "$A/prof_in8192_out16_conc4_p8/in8192_out16_conc4_p8-AMD-TP-0-DECODE.trace.json.gz" \
    --match "DECODE bs=4" --kernel cross_device_reduce
