#!/usr/bin/env python3
"""
cache_monitor.py — emit a per-round L1 / L2 / L3 cache-hit-rate report for
an SGLang multi-turn cascade run by polling the server's Prometheus
/metrics endpoint and triggering on round-barrier crossings.

SGLang tags every prefix-cache hit token with which tier serviced it:
  cache_source="device"        → L1 (GPU radix cache)
  cache_source="host"          → L2 (HiCache host DRAM)
  cache_source="storage_<bk>"  → L3 (HiCache storage backend, e.g. file)

We also read sglang:num_requests_total (+ aborted) to know how many
requests have finished. With bench_multiturn.py --enable-round-barrier,
round k finishes when (k+1) * num_clients requests have completed after
the monitor's baseline sample. Each time we cross a round boundary we
snapshot the per-tier counters, compute deltas vs the previous round's
snapshot, and emit one line + one CSV row for that round.

Server must have been launched with --enable-metrics --enable-cache-report.

Example (matches what cascade_dsr1.sh injects):
  python3 cache_monitor.py \\
      --url http://localhost:30000/metrics \\
      --interval 1 \\
      --num-clients 300 \\
      --num-rounds 15 \\
      --csv $LOG_DIR/cache_tiers.csv
"""

import argparse
import csv
import os
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime


METRIC_PROMPT_TOTAL = "sglang:prompt_tokens_total"
METRIC_CACHED_TOTAL = "sglang:cached_tokens_total"
METRIC_NUM_REQUESTS = "sglang:num_requests_total"
METRIC_NUM_ABORTED = "sglang:num_aborted_requests_total"


def parse_prometheus_metrics(text):
    """Aggregate the few metrics we care about from a Prometheus text payload.

    Sums across all label combinations of a given (metric, cache_source).
    Returns a dict with float counters; downstream casts to int as needed.
    """
    out = {
        "prompt": 0.0,
        "device": 0.0,
        "host": 0.0,
        "storage": 0.0,
        "fallback": 0.0,
        "num_finished": 0.0,
        "num_aborted": 0.0,
    }
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        try:
            metric_part, value_str = line.rsplit(" ", 1)
            value = float(value_str)
        except ValueError:
            continue
        if metric_part.startswith(METRIC_PROMPT_TOTAL):
            out["prompt"] += value
        elif metric_part.startswith(METRIC_CACHED_TOTAL):
            if 'cache_source="device"' in metric_part:
                out["device"] += value
            elif 'cache_source="host"' in metric_part:
                out["host"] += value
            elif 'cache_source="storage_' in metric_part:
                out["storage"] += value
            elif 'cache_source="total"' in metric_part:
                out["fallback"] += value
        elif metric_part.startswith(METRIC_NUM_REQUESTS):
            out["num_finished"] += value
        elif metric_part.startswith(METRIC_NUM_ABORTED):
            out["num_aborted"] += value
    return out


def pct(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--url", default="http://localhost:30000/metrics",
                   help="SGLang Prometheus /metrics URL")
    p.add_argument("--interval", type=float, default=1.0,
                   help="Polling interval in seconds (default 1, fast enough "
                        "to catch sub-10s rounds cleanly)")
    p.add_argument("--csv", default="cache_tiers.csv",
                   help="Output CSV file (one row per finished round)")
    p.add_argument("--num-clients", type=int, required=True,
                   help="Cascade bench --num-clients (=requests per round, "
                        "assuming all clients participate in every round)")
    p.add_argument("--num-rounds", type=int, required=True,
                   help="Cascade bench --num-rounds; monitor exits after the "
                        "final round's report is emitted")
    p.add_argument("--duration", type=float, default=0,
                   help="Hard upper bound in seconds (0 = no cap; rely on "
                        "round count instead)")
    p.add_argument("--quiet", action="store_true",
                   help="Don't print human-readable lines (CSV only)")
    args = p.parse_args()

    if args.num_clients <= 0 or args.num_rounds <= 0:
        sys.stderr.write("--num-clients and --num-rounds must be positive\n")
        sys.exit(2)

    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    csv_file = open(args.csv, "w", buffering=1, newline="")
    writer = csv.writer(csv_file)
    # One row per finished round. Counters are deltas attributable to that
    # round only (current snapshot minus previous round's snapshot, or
    # baseline for round 0). Rates are computed from those deltas, so each
    # row is independent and reflects per-round behavior — exactly what
    # the cascade plot needs.
    writer.writerow([
        "round_index",
        "ts_end",
        "duration_s",
        "requests",            # finished + aborted attributed to this round
        "prompt_delta",        # input tokens fed during this round
        "device_delta",        # L1 tokens hit during this round
        "host_delta",          # L2 tokens hit during this round
        "storage_delta",       # L3 tokens hit during this round
        "L1_pct", "L2_pct", "L3_pct", "any_pct",
        # Snapshot totals at round end (useful for sanity / joining):
        "prompt_total", "device_total", "host_total", "storage_total",
        "num_finished_total", "num_aborted_total",
    ])

    monitor_start_iso = datetime.now().isoformat(timespec="seconds")
    print(
        f"[{monitor_start_iso}] cache_monitor: url={args.url} "
        f"interval={args.interval}s csv={args.csv} "
        f"num_clients={args.num_clients} num_rounds={args.num_rounds}",
        flush=True,
    )

    def cleanup(*_):
        try:
            csv_file.flush()
            csv_file.close()
        except Exception:
            pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)

    start_mono = time.monotonic()
    deadline = (start_mono + args.duration) if args.duration > 0 else float("inf")

    baseline = None
    last_round_snapshot = None
    last_round_end_mono = None
    next_round_idx = 0

    def fetch_and_parse():
        with urllib.request.urlopen(args.url, timeout=5) as resp:
            payload = resp.read().decode("utf-8", errors="replace")
        return parse_prometheus_metrics(payload)

    def total_done(m):
        # Both finished and aborted counts contribute to "round complete"
        # so a sporadic abort doesn't stall round detection forever.
        return m["num_finished"] + m["num_aborted"]

    while next_round_idx < args.num_rounds:
        loop_start = time.monotonic()
        if loop_start >= deadline:
            print(f"[{datetime.now().isoformat(timespec='seconds')}] "
                  f"reached --duration={args.duration}s before round "
                  f"{next_round_idx}/{args.num_rounds}; exiting",
                  file=sys.stderr, flush=True)
            break

        try:
            m = fetch_and_parse()
        except (urllib.error.URLError, urllib.error.HTTPError,
                ConnectionError, TimeoutError, OSError) as e:
            if not args.quiet:
                print(f"[{datetime.now().isoformat(timespec='seconds')}] "
                      f"metrics fetch failed ({type(e).__name__}: {e}); "
                      f"retrying in {args.interval}s",
                      file=sys.stderr, flush=True)
            time.sleep(args.interval)
            continue

        ts_iso = datetime.now().isoformat(timespec="seconds")

        if baseline is None:
            baseline = dict(m)
            last_round_snapshot = dict(m)
            last_round_end_mono = loop_start
            if not args.quiet:
                print(
                    f"[{ts_iso}] baseline captured: "
                    f"finished={int(m['num_finished'])} "
                    f"aborted={int(m['num_aborted'])} "
                    f"prompt={int(m['prompt'])} "
                    f"L1={int(m['device'])} L2={int(m['host'])} L3={int(m['storage'])}",
                    flush=True,
                )
            time.sleep(args.interval)
            continue

        done_since_baseline = total_done(m) - total_done(baseline)
        # Emit one row per round boundary crossed since the previous poll.
        # If multiple rounds completed in one poll interval (e.g. very
        # short rounds with --output-length 1 + high cache hit), the
        # delta gets attributed to the latest crossed round only — pick a
        # finer --interval (e.g. 0.5s) if that matters for your run.
        while next_round_idx < args.num_rounds and \
                done_since_baseline >= (next_round_idx + 1) * args.num_clients:
            dp = m["prompt"] - last_round_snapshot["prompt"]
            dl1 = m["device"] - last_round_snapshot["device"]
            dl2 = m["host"] - last_round_snapshot["host"]
            dl3 = m["storage"] - last_round_snapshot["storage"]
            dreq = int(total_done(m) - total_done(last_round_snapshot))
            duration_s = loop_start - last_round_end_mono if last_round_end_mono else 0.0

            l1p, l2p, l3p, anyp = (
                pct(dl1, dp), pct(dl2, dp), pct(dl3, dp), pct(dl1 + dl2 + dl3, dp)
            )

            writer.writerow([
                next_round_idx,
                ts_iso,
                f"{duration_s:.2f}",
                dreq,
                int(dp), int(dl1), int(dl2), int(dl3),
                f"{l1p:.3f}", f"{l2p:.3f}", f"{l3p:.3f}", f"{anyp:.3f}",
                int(m["prompt"]), int(m["device"]), int(m["host"]), int(m["storage"]),
                int(m["num_finished"]), int(m["num_aborted"]),
            ])
            csv_file.flush()

            if not args.quiet:
                print(
                    f"[{ts_iso}] Round {next_round_idx:>2}/{args.num_rounds-1}: "
                    f"L1={l1p:6.2f}%  L2={l2p:6.2f}%  L3={l3p:6.2f}%  any={anyp:6.2f}%  "
                    f"| {dreq:>4} req  prompt={int(dp):>10,} tok  duration={duration_s:6.1f}s  "
                    f"({(dp/duration_s/1000.0 if duration_s > 0 else 0):6.1f} kT/s)",
                    flush=True,
                )

            last_round_snapshot = dict(m)
            last_round_end_mono = loop_start
            next_round_idx += 1

        sleep_for = args.interval - (time.monotonic() - loop_start)
        if sleep_for > 0:
            time.sleep(sleep_for)

    cleanup()


if __name__ == "__main__":
    main()
