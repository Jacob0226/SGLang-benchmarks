#!/usr/bin/env python3
"""Per-point health check for an AgentX sweep directory.

The summariser reports performance; this reports whether the point is one you
are allowed to quote. Three things decide that and none of them are in the
performance table: whether the scheduler died and restarted mid-run, whether
AIPerf marked the window long enough to submit, and what fraction of requests
errored.

OOM warnings are printed for context, not as a failure signal: the caching
allocator logs one whenever it has to hand memory back to the driver and
retry, which succeeds. Only a scheduler exception is a real failure.

    ./check_health.py <sweep-dir>
"""

import glob
import json
import os
import sys


def count_crashes_during_measurement(server_text, point_dir):
    """Scheduler exceptions that happened while the benchmark was running.

    Teardown is noisy and looks identical to a failure at a glance: killing the
    launcher leaves the TP workers to notice their peer vanish, and each logs
    `Scheduler hit an exception` wrapping a gloo "Connection reset by peer".
    Those land seconds AFTER the profiling window closed and say nothing about
    the measurement. Counting them flagged a clean six-point curve as five
    bad points.

    So compare against the moment AIPerf stopped sending. No timestamp to
    compare with means fall back to the raw count, which errs toward flagging.
    """
    marker = "Phase profiling (profiling) sending complete"
    recipe_log = os.path.join(point_dir, "recipe.log")
    cutoff = None
    try:
        with open(recipe_log, errors="ignore") as fh:
            for line in fh:
                if marker in line:
                    cutoff = line.split(None, 1)[0]  # HH:MM:SS.mmm
    except OSError:
        pass

    hits = [ln for ln in server_text.splitlines() if "Scheduler hit an exception" in ln]
    if cutoff is None:
        return len(hits)

    cutoff_hms = cutoff.split(".")[0]
    during = 0
    for line in hits:
        # "[2026-10-02 07:04:41 TP1] Scheduler hit an exception: ..."
        #   parts[0]="[2026-10-02"  parts[1]="07:04:41"  parts[2]="TP1]"
        parts = line.split()
        stamp = parts[1] if len(parts) > 1 and parts[1].count(":") == 2 else None
        if stamp is None or stamp <= cutoff_hms:
            during += 1
    return during


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    root = sys.argv[1]

    rows = []
    for result_path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        if "aiperf_artifacts" in result_path:
            continue
        try:
            with open(result_path) as fh:
                blob = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if "conc" not in blob:
            continue
        point_dir = os.path.dirname(result_path)
        server_log = os.path.join(point_dir, "server.log")
        text = ""
        if os.path.exists(server_log):
            with open(server_log, errors="ignore") as fh:
                text = fh.read()
        crash = count_crashes_during_measurement(text, point_dir)
        acct = blob.get("request_accounting", {})
        # records_error_dropped counts errors across every record, and warmup
        # runs with max_tokens=1, which AIPerf files as
        # InvalidInferenceResultError -- hundreds per run, harmless, and all of
        # them warmup. The number that matters is errors among *profiled*
        # records, which is what AIPerf's own 10% gate gates on. Recover it
        # from the accounting identity rather than from records_error_dropped:
        #   total = warmup_dropped + profiled + errored_profiled
        profiled = max(acct.get("records_profiled", 0), 1)
        errored_profiled = max(
            acct.get("records_total", 0)
            - acct.get("records_warmup_dropped", 0)
            - acct.get("records_profiled", 0),
            0,
        )
        rows.append(
            {
                "conc": blob["conc"],
                "crash": crash,
                "oom": text.count("memory allocation failed with OOM"),
                "valid": blob.get("submission_valid"),
                "err_pct": 100.0 * errored_profiled / profiled,
                "profiled": acct.get("records_profiled"),
                "total": acct.get("records_total"),
            }
        )
    rows.sort(key=lambda r: r["conc"])

    head = f"{'conc':>5}{'crash':>7}{'oom_warn':>10}{'submittable':>13}{'err%':>8}{'profiled/all':>14}"
    print(head)
    print("-" * len(head))
    for r in rows:
        requests = f"{r['profiled']}/{r['total']}"
        print(
            f"{r['conc']:>5}{r['crash']:>7}{r['oom']:>10}{str(r['valid']):>13}"
            f"{r['err_pct']:>7.1f}%{requests:>14}"
        )
    # 10% is AIPerf's own post-run gate (AIPERF_FAILED_REQUEST_THRESHOLD).
    bad = [
        r["conc"] for r in rows
        if r["crash"] or r["valid"] is False or r["err_pct"] > 10.0
    ]
    print()
    print("全部可用" if not bad else f"有問題的點: conc {bad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
