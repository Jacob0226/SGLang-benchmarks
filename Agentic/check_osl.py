#!/usr/bin/env python3
"""Per-point output-length sanity check for an AgentX sweep directory.

The AgentX scenario injects ignore_eos=true, so every request is meant to run
to the max_tokens its trace recorded. OSL landing on that cap is therefore the
expected shape, not a symptom, and a raw OSL histogram cannot tell a healthy
server from one that never terminates. What can:

  at cap   share of requests whose OSL equals the requested max_tokens
  short    share that stopped early anyway (a stop token ignore_eos does not cover)
  over     requests that exceeded max_tokens; must be 0
  AL       mean MTP acceptance length from the server's own metrics. Degenerate,
           repetitive output drafts unusually well, so an AL far above the other
           platform's at the same point is the throughput-inflation tell.

    ./check_osl.py <sweep-dir>
"""

import csv
import glob
import json
import os
import statistics
import sys


def metric(record, key):
    return (record["metrics"].get(key) or {}).get("value")


def accept_length(point_dir):
    path = os.path.join(point_dir, "aiperf_artifacts", "server_metrics_export.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        for row in csv.reader(fh):
            if len(row) > 4 and row[2] == "sglang:spec_accept_length":
                return float(row[4])
    return None


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    root = sys.argv[1]

    rows = []
    for point_dir in glob.glob(os.path.join(root, "*conc*")):
        export = os.path.join(point_dir, "aiperf_artifacts", "profile_export.jsonl")
        # A point still in flight has a partial export and no result JSON yet.
        if not os.path.exists(export) or not glob.glob(os.path.join(point_dir, "*_local-*.json")):
            continue
        with open(export) as fh:
            records = [json.loads(line) for line in fh]
        ok = [r for r in records if r["metadata"]["benchmark_phase"] == "profiling" and not r.get("error")]
        if not ok:
            continue
        osl = [metric(r, "output_sequence_length") for r in ok]
        diff = [metric(r, "osl_mismatch_diff_pct") or 0.0 for r in ok]
        requested = [o / (1 + d / 100) for o, d in zip(osl, diff) if d > -100]
        rows.append(
            {
                "conc": int(point_dir.rsplit("conc", 1)[1].split("_")[0]),
                "n": len(ok),
                "osl_p50": statistics.median(osl),
                "osl_p90": statistics.quantiles(osl, n=10)[8],
                "req_p50": statistics.median(requested),
                "at_cap": 100.0 * sum(abs(d) < 1 for d in diff) / len(ok),
                "short": 100.0 * sum(d < -1 for d in diff) / len(ok),
                "over": sum(d > 1 for d in diff),
                "al": accept_length(point_dir),
            }
        )
    rows.sort(key=lambda r: r["conc"])

    head = f"{'conc':>5}{'n':>7}{'OSL p50':>9}{'OSL p90':>9}{'req p50':>9}{'at cap':>8}{'short':>7}{'over':>6}{'AL':>6}"
    print(head)
    print("-" * len(head))
    for r in rows:
        al = f"{r['al']:.2f}" if r["al"] is not None else "?"
        print(
            f"{r['conc']:>5}{r['n']:>7}{r['osl_p50']:>9.0f}{r['osl_p90']:>9.0f}{r['req_p50']:>9.0f}"
            f"{r['at_cap']:>7.0f}%{r['short']:>6.0f}%{r['over']:>6}{al:>6}"
        )
    bad = [r["conc"] for r in rows if r["over"]]
    print()
    print("沒有超出 max_tokens 的請求" if not bad else f"有請求超出 max_tokens: conc {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
