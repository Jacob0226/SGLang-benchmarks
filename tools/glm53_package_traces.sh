#!/usr/bin/env bash
# Drop the provenance file into the results dir and package it for upload.
# The archive keeps the directory structure but carries TP-0 traces only
# (the other three ranks are ~3x the bytes and say the same thing).
# Runs inside the container, which owns those root-created files.
set -uo pipefail
exec > /home/jacchang/SGLang-benchmarks/tmp/logs/glm53_package.log 2>&1

RES=/home/jacchang/SGLang-benchmarks/results/amd_GLM-5.3-Flash-Quark-MXFP4/rocm_sgl-dev-v0.5.19-rocm720-mi35x-20260914/prof-Fixed-MXFP4-TP4-PRstack
OUT=/home/jacchang/SGLang-benchmarks/tmp/Docker0914-10PR.zip

cp /home/jacchang/glm53_run_provenance.json "$RES/run_provenance.json"
echo "provenance -> $RES/run_provenance.json"

python3 - "$RES" "$OUT" <<'PY'
import os, sys, zipfile

res, out = sys.argv[1], sys.argv[2]
root_name = os.path.basename(res)
os.makedirs(os.path.dirname(out), exist_ok=True)

kept, skipped = [], 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for dirpath, dirnames, filenames in os.walk(res):
        dirnames[:] = [d for d in dirnames if not d[0].isdigit()]  # stray raw profiler dirs
        for fn in sorted(filenames):
            if fn.endswith(".trace.json.gz") and "TP-0" not in fn:
                skipped += 1
                continue
            full = os.path.join(dirpath, fn)
            arc = os.path.join(root_name, os.path.relpath(full, res))
            z.write(full, arc)
            kept.append(arc)

print(f"files written : {len(kept)}")
print(f"traces skipped: {skipped} (non-TP-0)")
print(f"archive       : {out}  {os.path.getsize(out)/1e6:.1f} MB")
for a in kept:
    print("  ", a)
PY
