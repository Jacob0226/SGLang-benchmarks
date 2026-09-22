#!/usr/bin/env bash
# Per-launch duration histogram for the decode collective on both platforms.
# The question is whether a big Sigma means "communication is expensive" or
# "one launch per forward sat at a barrier waiting for the other ranks".
# Output: ~/allreduce_hist.txt
exec > /home/jacchang/allreduce_hist.txt 2>&1
cd /home/jacchang/SGLang-benchmarks || exit 1

B=results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130/prof-Fixed-NVFP4-TP4
A=results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/prof-Fixed-MXFP4-TP4-PRstack

python3 - "$B/prof_in8192_out16_conc4_p8/in8192_out16_conc4_p8-NV-TP-0-DECODE.trace.json.gz" all_reduce \
         "$B/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-NV-TP-0-DECODE.trace.json.gz" all_reduce \
         "$A/prof_in8192_out16_conc4_p8/in8192_out16_conc4_p8-AMD-TP-0-DECODE.trace.json.gz" cross_device_reduce \
         "$A/prof_in8192_out16_conc64_p128/in8192_out16_conc64_p128-AMD-TP-0-DECODE.trace.json.gz" cross_device_reduce <<'PY'
import gzip, json, sys, statistics

args = sys.argv[1:]
for path, pat in zip(args[0::2], args[1::2]):
    ev = json.load(gzip.open(path, "rt")).get("traceEvents", [])
    durs = sorted(e.get("dur", 0) for e in ev
                  if e.get("ph") == "X" and e.get("cat") == "kernel" and pat in e.get("name", ""))
    if not durs:
        print(f"{path.split('/')[-1]}: no {pat} kernels\n")
        continue
    buckets = [(0, 20), (20, 100), (100, 500), (500, 2000), (2000, 10 ** 9)]
    print(f"=== {path.split('/')[-1]}   filter={pat}")
    print(f"    launches={len(durs)}  Sigma={sum(durs)/1000:.3f} ms  "
          f"median={statistics.median(durs):.1f} us  max={durs[-1]:.1f} us")
    for lo, hi in buckets:
        sel = [d for d in durs if lo <= d < hi]
        if sel:
            hi_s = "inf" if hi > 10 ** 8 else str(hi)
            print(f"      {lo:>5}-{hi_s:<5} us : n={len(sel):<4} Sigma={sum(sel)/1000:8.3f} ms"
                  f"  ({100*sum(sel)/sum(durs):5.1f}% of Sigma)")
    tail = [d for d in durs if d >= 500]
    if tail:
        print(f"    launches >=500us: {[round(d) for d in tail]}")
    print()
PY
