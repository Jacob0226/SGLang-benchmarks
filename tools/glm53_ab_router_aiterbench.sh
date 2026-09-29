#!/bin/bash
# A/B the router GEMM (N=288, K=4096) with aiter's own benchmark, by moving
# GEMM-A16W16-N=288-K=4096.json in and out of the gfx950 config directory.
#
# This is the table the aiter PR needs: a reviewer can rerun the exact command.
# The numbers in analysis_GLM5.3/router_gemm_0928/bucket_sweep.csv came from our
# own HIP-graph harness, which is not something a reviewer can reproduce.
#
# M values cover the gated range (1..64) plus M=128/256, which resolve to
# buckets copied verbatim from DEFAULT.json and must not move.
set -uo pipefail
exec 200>/tmp/glm53_bench.lock
flock -n 200 || { echo "another GPU run holds the lock; refusing"; exit 0; }

OUT=/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928
mkdir -p "$OUT" && chmod 777 "$OUT" 2>/dev/null
LOG=/home/jacchang/SGLang-benchmarks/tmp/logs/ab_router_aiterbench.log
exec > "$LOG" 2>&1

BENCH=/sgl-workspace/aiter/op_tests/op_benchmarks/triton/bench_gemm_a16w16.py
CFG_DIR=/sgl-workspace/aiter/aiter/ops/triton/configs/gfx950/triton/gemm/gemm_a16w16
DST=$CFG_DIR/GEMM-A16W16-N=288-K=4096.json
SRC=$OUT/GEMM-A16W16-N=288-K=4096.best_single.json

export HIP_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/home/jacchang/SGLang-benchmarks/tmp/cache-router-gemm-0928
mkdir -p "$TRITON_CACHE_DIR"
cd /sgl-workspace/aiter

MS="1 2 4 8 16 32 64 128 256"

run_arm() {
    local tag=$1
    : > "$OUT/aiterbench_$tag.txt"
    for M in $MS; do
        echo "### M=$M" >> "$OUT/aiterbench_$tag.txt"
        python3 "$BENCH" --shape "$M" 288 4096 --metric time \
            >> "$OUT/aiterbench_$tag.txt" 2>&1
    done
}

echo "=== arm A: without the config (DEFAULT.json fallback) ==="
rm -f "$DST"
run_arm without

echo "=== arm B: with the config ==="
cp "$SRC" "$DST"
chmod 644 "$DST"
run_arm with

echo
echo "=== markdown table ==="
python3 - <<'PY'
import re
import os

OUT = "/home/jacchang/SGLang-benchmarks/analysis_GLM5.3/router_gemm_0928"
MS = [1, 2, 4, 8, 16, 32, 64, 128, 256]


def parse(tag):
    """triton.testing.perf_report prints a small table; take the last float."""
    out = {}
    cur = None
    for line in open(f"{OUT}/aiterbench_{tag}.txt"):
        m = re.match(r"### M=(\d+)", line)
        if m:
            cur = int(m.group(1))
            continue
        vals = re.findall(r"(\d+\.\d+)", line)
        if cur is not None and vals and not line.lstrip().startswith("M "):
            out[cur] = float(vals[-1])
    return out


a, b = parse("without"), parse("with")
print("| M | without | with |")
print("| --- | --- | --- |")
for M in MS:
    x, y = a.get(M), b.get(M)
    if x is None or y is None:
        print(f"| {M} | {x} | {y} |")
        continue
    cell = f"**{y:.1f}**" if y < x * 0.97 else f"{y:.1f}"
    print(f"| {M} | {x:.1f} | {cell} |")
PY
chmod -R a+rw "$OUT" 2>/dev/null
chmod a+rw "$LOG" 2>/dev/null
echo "AB_DONE"
