#!/usr/bin/env bash
# Idempotent patch: add per-round median_ttft + p99_ttft to the hicache
# bench_multiturn.py per-round summary dict.
#
# Why: bench_multiturn already collects the raw per-request TTFT list per
# round (round_metrics["ttft"]) and even computes median/p99 for the
# WHOLE-RUN summary, but the per-round dict only emits "average_ttft"
# (mean). Mean is heavily skewed by cold-start / long-tail outliers
# (notably on the ROCm aiter path), so per-round median is needed to
# compare steady-state TTFT. The whole-run "percentile" helper is in
# scope at the round loop, so we reuse it.
#
# Usage (inside the container):
#   bash /home/jacchang/SGLang-benchmarks/HiCachePatch/bench-multiturn-round-percentiles.sh
#   bash .../bench-multiturn-round-percentiles.sh /path/to/sglang/repo
#
# Default target is /sgl-workspace/sglang (the container's sglang checkout).
# The host edit repo at ~/PR/sglang must NOT be patched.
#
# Exit codes:
#   0 = applied OR already applied (safe to re-run)
#   2 = anchor found but format differs (upstream changed, inspect manually)
#   3 = target file not found
set -euo pipefail

REPO="${1:-/sgl-workspace/sglang}"
TARGET="$REPO/benchmark/hicache/bench_multiturn.py"

if [[ ! -f "$TARGET" ]]; then
  echo "ERROR: $TARGET not found" >&2
  exit 3
fi

python3 - "$TARGET" <<'PY'
import sys, pathlib

p = pathlib.Path(sys.argv[1])
src = p.read_text()

anchor = (
    '                performance_data["round"][round_key] = {\n'
    '                    "average_ttft": (\n'
    '                        sum(round_metrics["ttft"]) / len(round_metrics["ttft"])\n'
    '                        if round_metrics["ttft"]\n'
    '                        else 0\n'
    '                    ),\n'
)

insert = (
    '                    "median_ttft": (\n'
    '                        percentile(sorted(round_metrics["ttft"]), 0.5)\n'
    '                        if round_metrics["ttft"]\n'
    '                        else 0\n'
    '                    ),\n'
    '                    "p99_ttft": (\n'
    '                        percentile(sorted(round_metrics["ttft"]), 0.99)\n'
    '                        if round_metrics["ttft"]\n'
    '                        else 0\n'
    '                    ),\n'
)

# Idempotency key: the per-round insert block is unique (it uses
# `sorted(round_metrics["ttft"])`, distinct from the whole-run summary's
# own median_ttft which sorts a different list). Checking the bare
# string "median_ttft" would false-positive on that summary field.
if insert in src:
    print(f"[bench-multiturn-round-percentiles] already applied in {p} (no-op)")
elif anchor in src:
    p.write_text(src.replace(anchor, anchor + insert))
    print(f"[bench-multiturn-round-percentiles] added per-round median_ttft/p99_ttft to {p}")
else:
    print(
        "[bench-multiturn-round-percentiles] WARNING: per-round average_ttft "
        "anchor not found; upstream bench_multiturn.py format likely changed. "
        "Inspect manually.",
        file=sys.stderr,
    )
    sys.exit(2)
PY
