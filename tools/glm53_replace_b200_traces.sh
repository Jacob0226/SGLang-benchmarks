#!/usr/bin/env bash
# Replace the local B200 traces with the 4-rank re-upload staged by
# glm53_fetch_b200_traces.sh. Layout was verified first: each zip's root is the
# prof directory name that already exists locally, so this extracts over the top
# rather than deleting anything -- the new set is a superset (same TP-0 files,
# plus TP-1..3).
#
# Afterwards it lists any file that existed locally but is NOT in the zip, so a
# stale leftover is reported rather than silently kept, and drops the zips
# because /home is at 100%.
#
# Output: ~/glm53_replace_b200.log
set -uo pipefail
exec > /home/jacchang/glm53_replace_b200.log 2>&1

STAGE=/home/jacchang/SGLang-benchmarks/tmp/b200_traces
DEST_ROOT=/home/jacchang/SGLang-benchmarks/results/nvidia_GLM-5.3-Flash-NVFP4/lmsysorg_sglang-v0.5.20-cu130

cd "$DEST_ROOT" || exit 1

for pair in "B200_Docker_v0.5.20-cu130.zip prof-Fixed-NVFP4-TP4" \
            "B200_Docker_v0.5.20-cu130_MTP.zip prof-Fixed-MTP-NVFP4-TP4"; do
    set -- $pair
    zip="$STAGE/$1"; dir="$2"
    echo "================ $dir  <-  $1"
    before=$(find "$dir" -type f 2>/dev/null | sed "s|^$dir/||" | sort)
    echo "  local files before : $(echo "$before" | grep -c . )"
    echo "  local traces before: $(find "$dir" -name '*.trace.json.gz' 2>/dev/null | wc -l)"

    python3 - "$zip" . <<'PY'
import sys, zipfile
z = zipfile.ZipFile(sys.argv[1])
z.extractall(sys.argv[2])
print(f"  extracted {len(z.namelist())} entries")
PY

    after_traces=$(find "$dir" -name '*.trace.json.gz' | wc -l)
    echo "  local traces after : $after_traces"
    for r in 0 1 2 3; do
        printf "    TP-%s: %s traces\n" "$r" "$(find "$dir" -name "*-TP-$r-*.trace.json.gz" | wc -l)"
    done

    inzip=$(python3 -c "
import sys, zipfile
d = sys.argv[2] + '/'
print('\n'.join(sorted(n[len(d):] for n in zipfile.ZipFile(sys.argv[1]).namelist()
                       if n.startswith(d) and not n.endswith('/'))))" "$zip" "$dir")
    stale=$(comm -23 <(echo "$before") <(echo "$inzip"))
    if [ -n "$stale" ]; then
        echo "  !! present locally but NOT in the zip (kept, check if stale):"
        echo "$stale" | sed 's/^/       /'
    else
        echo "  no local-only leftovers"
    fi
    echo
done

echo "=== reclaiming staging space (/home is at 100%) ==="
rm -rf "$STAGE"
du -sh "$DEST_ROOT"
df -h /home | tail -1
