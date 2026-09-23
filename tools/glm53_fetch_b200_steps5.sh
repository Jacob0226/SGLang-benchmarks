#!/usr/bin/env bash
# Download the B200 steps=5 capture and place it under results/.
# Inspects the archive's layout first: whether it is a new profile directory or
# a replacement for an existing one decides where it goes, and guessing that
# wrong either strands the old traces or mixes two captures in one directory.
#
# Output: ~/glm53_fetch_b200_steps5.log
set -uo pipefail
exec > /home/jacchang/glm53_fetch_b200_steps5.log 2>&1

REPO="${REPO:-JacobChang/GLM5.3-Flash-FP4}"
FILE="${FILE:-B200_Docker_v0.5.20-cu130_steps5.zip}"
STAGE="${STAGE:-/home/jacchang/SGLang-benchmarks/tmp/b200_steps5}"
DEST_ROOT=/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130

python3 -c 'import huggingface_hub' 2>/dev/null \
    || python3 -m pip install -q huggingface_hub \
    || python3 -m pip install -q --break-system-packages huggingface_hub

mkdir -p "$STAGE"
echo "=== downloading $FILE at $(date '+%F %T') ==="
python3 - "$REPO" "$FILE" "$STAGE" <<'PY'
import sys
from huggingface_hub import hf_hub_download
print("saved to", hf_hub_download(repo_id=sys.argv[1], repo_type="dataset",
                                  filename=sys.argv[2], local_dir=sys.argv[3]))
PY

echo
echo "=== archive layout ==="
python3 - "$STAGE/$FILE" "$DEST_ROOT" <<'PY'
import sys, zipfile, collections, os
z = zipfile.ZipFile(sys.argv[1]); dest = sys.argv[2]
names = [n for n in z.namelist() if not n.endswith("/")]
roots = collections.Counter(n.split("/")[0] for n in names)
print(f"  {len(names)} files")
for r, c in roots.most_common():
    exists = os.path.isdir(os.path.join(dest, r))
    print(f"  root: {r}  ({c} files)   already in results/: {exists}")
traces = [n for n in names if n.endswith(".trace.json.gz")]
ranks = collections.Counter(r for t in traces for r in ("TP-0","TP-1","TP-2","TP-3") if r in t)
print(f"  traces: {len(traces)}   per rank: {dict(sorted(ranks.items()))}")
shapes = sorted({n.split('/')[-2] for n in traces})
print("  profile dirs:")
for s in shapes:
    print("    ", s)
PY

echo
echo "=== extracting into $DEST_ROOT ==="
python3 - "$STAGE/$FILE" "$DEST_ROOT" <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1]); z.extractall(sys.argv[2])
print(f"  extracted {len(z.namelist())} entries")
PY

echo
echo "=== result ==="
for d in "$DEST_ROOT"/*/; do
    printf "  %-40s %4s traces\n" "$(basename "$d")" "$(find "$d" -name '*.trace.json.gz' | wc -l)"
done
rm -rf "$STAGE"
df -h /home | tail -1
