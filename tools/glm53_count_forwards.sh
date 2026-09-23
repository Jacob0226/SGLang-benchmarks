#!/usr/bin/env bash
# Did GLM.sh capture the same number of decode forwards on both platforms?
# Output: ~/count_forwards.txt
exec > /home/jacchang/count_forwards.txt 2>&1
cd /home/jacchang/SGLang-benchmarks || exit 1

A=results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/prof-Fixed-MXFP4-TP4-PRstack
B=results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-NVFP4-TP4
D=trace_analysis/diagnostics/count_forwards_per_trace.py

echo "################ MI355X conc4 decode"
python3 $D "$A/prof_in8192_out16_conc4_p8/in8192_out16_conc4_p8-AMD-TP-0-DECODE.trace.json.gz" \
    --match "DECODE bs=" --kernel cross_device_reduce
echo
echo "################ MI355X conc64 decode"
python3 $D "$A/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-DECODE.trace.json.gz" \
    --match "DECODE bs=" --kernel cross_device_reduce
echo
echo "################ B200 conc4 decode"
python3 $D "$B/prof_in8192_out16_conc4_p8/in8192_out16_conc4_p8-NV-TP-0-DECODE.trace.json.gz" \
    --match "DECODE bs=" --kernel all_reduce
echo
echo "################ B200 conc64 decode"
python3 $D "$B/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-NV-TP-0-DECODE.trace.json.gz" \
    --match "DECODE bs=" --kernel all_reduce
