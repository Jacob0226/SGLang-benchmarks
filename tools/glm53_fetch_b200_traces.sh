#!/usr/bin/env bash
# Re-download the B200 GLM-5.3-Flash traces from the Hub and replace the local
# copies. The first upload carried TP-0 only, which is why the all-reduce skew
# split could never run on the B200 side; these have all four ranks.
#
# Downloads and inspects first, replaces second: the zips' internal layout has
# to be checked against results/nvidia_GLM-5.3-Flash-NVFP4/... before anything
# local is deleted, or a layout change silently strands the old traces.
#
# Output: ~/glm53_fetch_b200.log
set -uo pipefail
exec > /home/jacchang/glm53_fetch_b200.log 2>&1

REPO="${REPO:-JacobChang/GLM5.3-Flash-FP4}"
STAGE="${STAGE:-/home/jacchang/SGLang-benchmarks/tmp/b200_traces}"
DEST_ROOT=/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130

python3 -c 'import huggingface_hub' 2>/dev/null \
    || python3 -m pip install -q huggingface_hub \
    || python3 -m pip install -q --break-system-packages huggingface_hub

mkdir -p "$STAGE"
echo "=== downloading at $(date '+%F %T') ==="
python3 - "$REPO" "$STAGE" <<'PY'
import sys
from huggingface_hub import hf_hub_download

repo, stage = sys.argv[1], sys.argv[2]
for name in ("B200_Docker_v0.5.20-cu130.zip", "B200_Docker_v0.5.20-cu130_MTP.zip"):
    p = hf_hub_download(repo_id=repo, repo_type="dataset", filename=name,
                        local_dir=stage)
    print("downloaded", p)
PY

echo
echo "=== zip layout (top two levels, before touching anything local) ==="
for z in "$STAGE"/B200_Docker_v0.5.20-cu130.zip "$STAGE"/B200_Docker_v0.5.20-cu130_MTP.zip; do
    echo "--- $(basename "$z")"
    python3 - "$z" <<'PY'
import sys, zipfile, collections
z = zipfile.ZipFile(sys.argv[1])
names = z.namelist()
print(f"    {len(names)} entries")
roots = collections.Counter(n.split("/")[0] for n in names)
for r, c in roots.most_common():
    print(f"    root: {r}  ({c} entries)")
traces = [n for n in names if n.endswith(".trace.json.gz")]
print(f"    traces: {len(traces)}")
ranks = collections.Counter()
for t in traces:
    for r in ("TP-0", "TP-1", "TP-2", "TP-3"):
        if r in t:
            ranks[r] += 1
print("    per rank:", dict(sorted(ranks.items())))
for t in sorted(traces)[:4]:
    print("      e.g.", t)
PY
done
echo
echo "=== NOT replacing anything yet; inspect the layout above first ==="
du -sh "$STAGE"
