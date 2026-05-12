#!/usr/bin/env python3
"""
cache_monitor.py — emit a periodic snapshot CSV that fuses (a) per-tier cache
hit rate from SGLang's Prometheus /metrics endpoint with (b) host RAM state
from /proc/meminfo, sampled together every --interval seconds (default 10s).

Replaces the older split between cache_monitor.py (per-round, event-driven
on round-barrier crossings) and host_mem_monitor.py (1 Hz time-driven).
Both signals share the same `ts` column on every row, so post-hoc
correlation between "TTFT spiked at round 3" and "host RAM crossed 85%"
becomes a CSV groupby on `round_index` instead of a wall-clock join.

Round detection. SGLang tags every prefix-cache hit token with which tier
serviced it:
    cache_source="device"        → L1 (GPU radix cache)
    cache_source="host"          → L2 (HiCache host DRAM)
    cache_source="storage_<bk>"  → L3 (HiCache storage backend, e.g. file)
The cumulative sglang:num_requests_total counter (+ aborted, so a single
fail doesn't stall round detection forever) tells us how many cascade
requests have completed since baseline. With bench_multiturn.py
--enable-round-barrier, round k is "in flight" when the k*num_clients-th
request through (k+1)*num_clients-th request is being served. We compute:

    reqs_done = max(0, num_done_total - baseline_done_total)
    round_index = -1                            # warmup / pre-cascade idle
    round_index = (reqs_done - 1) // num_clients  # round currently in flight

So during round 0, round_index=0; the moment the 300th request of round 0
completes the next sample sees reqs_done=301 → round_index=1, etc. When a
new round starts we snapshot the cache counters; per-row L1_pct/L2_pct/
L3_pct/any_pct are running within-round percentages, so the LAST sample
inside a round gives the round's final hit rate (equivalent to what the
old per-round cache_tiers.csv emitted).

Server must have been launched with --enable-metrics --enable-cache-report.

Example invocation (matches what cascade_dsr1.sh injects):
    python3 cache_monitor.py \\
        --url http://localhost:30000/metrics \\
        --interval 10 \\
        --num-clients 300 \\
        --num-rounds 15 \\
        --csv $LOG_DIR/cache_tiers.csv

CSV columns
-----------
ts                 ISO-8601 wall-clock (seconds)
elapsed_s          monotonic seconds since monitor start (for plotting)
round_index        -1 if pre-cascade, k = round currently in flight,
                   num_rounds when all rounds done (last row before exit)
round_elapsed_s    seconds since this round_index started
round_reqs         requests completed in current round (1..num_clients)
mem_total_gb       /proc/meminfo MemTotal in GiB
mem_avail_gb       MemAvailable in GiB
mem_used_pct       100 * (1 - MemAvailable / MemTotal); when this crosses
                   ~85% the write_through cliff is imminent
page_cache_gb      Cached + Buffers (mmap'd safetensors weights live here;
                   if this drops fast while mlocked rises, kernel is
                   evicting weights to make room for HiCache)
mlocked_gb         Mlocked (HiCache pinned host pool sits here; should
                   grow until --hicache-size target then plateau)
anon_gb            AnonPages (SGLang worker processes' working sets)
swap_used_gb       SwapTotal - SwapFree (any non-zero is a smoking gun)
prompt_total       sglang:prompt_tokens_total (cumulative since server start)
device_total       sglang:cached_tokens_total{cache_source="device"}
host_total         sglang:cached_tokens_total{cache_source="host"}
storage_total      sglang:cached_tokens_total{cache_source="storage_*"}
round_prompt_delta input tokens served so far in this round
round_device_delta L1 hit tokens so far in this round
round_host_delta   L2 hit tokens so far in this round
round_storage_delta L3 hit tokens so far in this round
L1_pct             100 * round_device_delta / round_prompt_delta (so far)
L2_pct             100 * round_host_delta   / round_prompt_delta (so far)
L3_pct             100 * round_storage_delta/ round_prompt_delta (so far)
any_pct            100 * (L1+L2+L3 hits)    / round_prompt_delta (so far)
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

MEMINFO_PATH = "/proc/meminfo"
# Subset of /proc/meminfo keys we project into CSV. All others are skipped
# at parse time so the per-sample cost stays cheap even at 0.1s intervals.
MEMINFO_KEYS = (
    "MemTotal",
    "MemFree",
    "MemAvailable",
    "Buffers",
    "Cached",
    "AnonPages",
    "Mlocked",
    "SwapTotal",
    "SwapFree",
)

CSV_FIELDS = (
    "ts",
    "elapsed_s",
    "round_index",
    "round_elapsed_s",
    "round_reqs",
    "mem_total_gb",
    "mem_avail_gb",
    "mem_used_pct",
    "page_cache_gb",
    "mlocked_gb",
    "anon_gb",
    "swap_used_gb",
    "prompt_total",
    "device_total",
    "host_total",
    "storage_total",
    "round_prompt_delta",
    "round_device_delta",
    "round_host_delta",
    "round_storage_delta",
    "L1_pct",
    "L2_pct",
    "L3_pct",
    "any_pct",
)


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


def read_meminfo():
    """Return /proc/meminfo as a dict of int kB values for MEMINFO_KEYS."""
    out = {}
    with open(MEMINFO_PATH) as f:
        for line in f:
            key, _, rest = line.partition(":")
            if key in MEMINFO_KEYS:
                out[key] = int(rest.strip().split()[0])
    return out


def kb_to_gb(value_kb):
    """Match `free -g` GiB convention so numbers line up with what the user
    sees in shell."""
    return value_kb / (1024 * 1024)


def pct(numerator, denominator):
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def total_done(m):
    """Round detection counts both finished AND aborted requests so a
    single sporadic failure can't stall round-boundary detection forever."""
    return m["num_finished"] + m["num_aborted"]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--url",
        default="http://localhost:30000/metrics",
        help="SGLang Prometheus /metrics URL",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=10.0,
        help="Sampling interval in seconds (default 10; 1s is fine but "
        "produces ~1800 rows over a 30-min cascade run)",
    )
    p.add_argument(
        "--csv",
        default="cache_tiers.csv",
        help="Output CSV file (one row per sample, see header for schema)",
    )
    p.add_argument(
        "--num-clients",
        type=int,
        required=True,
        help="bench_multiturn --num-clients (= requests per round)",
    )
    p.add_argument(
        "--num-rounds",
        type=int,
        required=True,
        help="bench_multiturn --num-rounds; monitor exits after the final "
        "round completes",
    )
    p.add_argument(
        "--duration",
        type=float,
        default=0,
        help="Hard upper bound in seconds (0 = no cap; rely on round count)",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Don't print human-readable status lines (CSV only)",
    )
    args = p.parse_args()

    if args.num_clients <= 0 or args.num_rounds <= 0:
        sys.stderr.write("--num-clients and --num-rounds must be positive\n")
        sys.exit(2)

    csv_dir = os.path.dirname(os.path.abspath(args.csv))
    if csv_dir:
        os.makedirs(csv_dir, exist_ok=True)

    # buffering=1 + flush after every row so killed-mid-run still leaves a
    # parseable CSV (matters because cascade_dsr1.sh's EXIT trap SIGTERM's
    # this process; we want the last few samples before the crash).
    csv_file = open(args.csv, "w", buffering=1, newline="")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    writer.writeheader()

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

    # Block until /metrics responds (server may still be starting). cap at
    # ~60s of retries so we don't deadlock if cascade_dsr1.sh is buggy.
    start_mono = time.monotonic()
    metrics = None
    for _ in range(30):
        try:
            with urllib.request.urlopen(args.url, timeout=5) as resp:
                metrics = parse_prometheus_metrics(
                    resp.read().decode("utf-8", errors="replace")
                )
            break
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            ConnectionError,
            TimeoutError,
            OSError,
        ) as e:
            if not args.quiet:
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"waiting for {args.url}: {type(e).__name__}",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(2)
    if metrics is None:
        print(f"[cache_monitor] gave up waiting for {args.url}", file=sys.stderr)
        cleanup()

    # Baseline: zero out anything from server startup, warmup, and the
    # GSM8K precheck so per-round percentages reflect cascade traffic only.
    baseline_done = total_done(metrics)
    cur_round = -1
    round_t0_mono = start_mono
    round_start = {
        "prompt": metrics["prompt"],
        "device": metrics["device"],
        "host": metrics["host"],
        "storage": metrics["storage"],
    }

    deadline = (start_mono + args.duration) if args.duration > 0 else float("inf")

    while True:
        loop_start = time.monotonic()
        if loop_start >= deadline:
            print(
                f"[{datetime.now().isoformat(timespec='seconds')}] "
                f"reached --duration={args.duration}s; exiting",
                file=sys.stderr,
                flush=True,
            )
            break

        try:
            with urllib.request.urlopen(args.url, timeout=5) as resp:
                m = parse_prometheus_metrics(
                    resp.read().decode("utf-8", errors="replace")
                )
            mi = read_meminfo()
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            ConnectionError,
            TimeoutError,
            OSError,
            KeyError,
        ) as e:
            if not args.quiet:
                print(
                    f"[{datetime.now().isoformat(timespec='seconds')}] "
                    f"sample error ({type(e).__name__}: {e}); retrying in "
                    f"{args.interval}s",
                    file=sys.stderr,
                    flush=True,
                )
            time.sleep(args.interval)
            continue

        ts_iso = datetime.now().isoformat(timespec="seconds")
        elapsed = loop_start - start_mono

        reqs_done = int(total_done(m) - baseline_done)
        if reqs_done <= 0:
            new_round = -1
        else:
            new_round = (reqs_done - 1) // args.num_clients

        # Round transition: snapshot the per-round counter baseline so the
        # new round's percentages start at 0.
        if new_round != cur_round:
            # If we skipped multiple round boundaries in one interval (rare:
            # rounds shorter than --interval), print a heads-up so the human
            # knows the resolution loss happened.
            if not args.quiet and cur_round >= 0 and new_round - cur_round > 1:
                print(
                    f"[{ts_iso}] WARNING: skipped {new_round - cur_round - 1} "
                    f"round boundary samples (round was shorter than "
                    f"--interval={args.interval}s); use a smaller interval "
                    f"if per-round resolution matters",
                    file=sys.stderr,
                    flush=True,
                )
            cur_round = new_round
            round_t0_mono = loop_start
            round_start = {
                "prompt": m["prompt"],
                "device": m["device"],
                "host": m["host"],
                "storage": m["storage"],
            }
            if not args.quiet and cur_round >= 0:
                print(
                    f"[{ts_iso}] >>> Round {cur_round}/{args.num_rounds - 1} "
                    f"started (reqs_done={reqs_done})",
                    flush=True,
                )

        d_prompt = max(int(m["prompt"] - round_start["prompt"]), 0)
        d_device = max(int(m["device"] - round_start["device"]), 0)
        d_host = max(int(m["host"] - round_start["host"]), 0)
        d_storage = max(int(m["storage"] - round_start["storage"]), 0)

        mem_total = kb_to_gb(mi["MemTotal"])
        mem_avail = kb_to_gb(mi["MemAvailable"])
        page_cache = kb_to_gb(mi.get("Cached", 0) + mi.get("Buffers", 0))
        mlocked = kb_to_gb(mi.get("Mlocked", 0))
        anon = kb_to_gb(mi.get("AnonPages", 0))
        swap_used = kb_to_gb(mi.get("SwapTotal", 0) - mi.get("SwapFree", 0))
        mem_used_pct = 100.0 * (1.0 - mem_avail / mem_total) if mem_total else 0.0

        round_reqs = (
            reqs_done - cur_round * args.num_clients if cur_round >= 0 else 0
        )

        l1p = pct(d_device, d_prompt)
        l2p = pct(d_host, d_prompt)
        l3p = pct(d_storage, d_prompt)
        anyp = pct(d_device + d_host + d_storage, d_prompt)

        writer.writerow(
            {
                "ts": ts_iso,
                "elapsed_s": round(elapsed, 2),
                "round_index": cur_round,
                "round_elapsed_s": (
                    round(loop_start - round_t0_mono, 2) if cur_round >= 0 else 0.0
                ),
                "round_reqs": round_reqs,
                "mem_total_gb": round(mem_total, 2),
                "mem_avail_gb": round(mem_avail, 2),
                "mem_used_pct": round(mem_used_pct, 2),
                "page_cache_gb": round(page_cache, 2),
                "mlocked_gb": round(mlocked, 2),
                "anon_gb": round(anon, 2),
                "swap_used_gb": round(swap_used, 2),
                "prompt_total": int(m["prompt"]),
                "device_total": int(m["device"]),
                "host_total": int(m["host"]),
                "storage_total": int(m["storage"]),
                "round_prompt_delta": d_prompt,
                "round_device_delta": d_device,
                "round_host_delta": d_host,
                "round_storage_delta": d_storage,
                "L1_pct": round(l1p, 3),
                "L2_pct": round(l2p, 3),
                "L3_pct": round(l3p, 3),
                "any_pct": round(anyp, 3),
            }
        )
        csv_file.flush()

        if not args.quiet:
            round_label = f"R{cur_round:>2}" if cur_round >= 0 else "warm"
            print(
                f"[{ts_iso}] t={elapsed:>6.0f}s {round_label} "
                f"reqs={round_reqs:>3d}/{args.num_clients} "
                f"mem_used={mem_used_pct:>5.1f}% "
                f"mlocked={mlocked:>5.0f}GB "
                f"page_cache={page_cache:>4.0f}GB "
                f"swap={swap_used:>4.1f}GB "
                f"L1/L2/L3={l1p:>5.1f}/{l2p:>5.1f}/{l3p:>5.1f}%",
                flush=True,
            )

        # Exit when we've seen the final round complete. With round-barrier,
        # the (num_rounds * num_clients)-th request completing means every
        # round is done — but cur_round only advances past num_rounds-1 if
        # bench_multiturn fires extra requests, which it usually doesn't.
        # So watch reqs_done directly, AND require that we've already
        # written at least one row for the final round (round_reqs > 0)
        # so the per-round final hit-rate is captured.
        if (
            cur_round >= args.num_rounds - 1
            and reqs_done >= args.num_rounds * args.num_clients
        ):
            if not args.quiet:
                print(
                    f"[{ts_iso}] all {args.num_rounds} rounds finished "
                    f"(reqs_done={reqs_done}); exiting",
                    flush=True,
                )
            break

        sleep_for = args.interval - (time.monotonic() - loop_start)
        if sleep_for > 0:
            time.sleep(sleep_for)

    cleanup()


if __name__ == "__main__":
    main()
