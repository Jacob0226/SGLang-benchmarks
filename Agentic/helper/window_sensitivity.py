#!/usr/bin/env python3
"""Ask what a shorter measurement window would have reported.

A 3600s AgentX point costs ~70 minutes wall clock, most of it measurement. If a
1200s window lands on the same numbers, iteration gets 3x cheaper. Rather than
spend GPU hours running both, truncate a finished run's records to the first N
seconds of its profiling window and re-run InferenceX's own aggregation over
the subset -- compute_throughput_stats derives its duration from the records it
is given, so the denominator follows the truncation.

This is a faithful simulation with one caveat: a real short run cancels
whatever is still in flight when the window closes, while truncation simply
never sees those records. Both drop the same tail, so the bias is second-order.

    ./window_sensitivity.py --windows 600,1200,1800,3600 <result-dir>...
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("IX", "/home/jacchang/InferenceX"))

from infx.results.agentic import build_result  # noqa: E402
from infx.results.agentic.artifacts import (  # noqa: E402
    load_aggregate,
    load_records_with_accounting,
    load_server_metrics,
    resolve_artifact_dir,
)


def load_env_from_result(result_json):
    """Rebuild the env build_result needs from the recorded result."""
    return {
        "RUNNER_TYPE": result_json.get("hw", ""),
        "CONC": str(result_json.get("conc", 0)),
        "IMAGE": result_json.get("image", ""),
        "MODEL": result_json.get("model", ""),
        "MODEL_PREFIX": result_json.get("infmax_model_prefix", ""),
        "FRAMEWORK": result_json.get("framework", ""),
        "PRECISION": result_json.get("precision", ""),
        "SPEC_DECODING": result_json.get("spec_decoding", "none"),
        "KV_OFFLOADING": result_json.get("kv_offloading", "none"),
        "KV_OFFLOAD_BACKEND": "",
        "IS_MULTINODE": "false",
        "TP": str(result_json.get("tp", 1)),
        "EP_SIZE": str(result_json.get("ep", 1)),
        "PP_SIZE": str(result_json.get("pp", 1)),
        "DCP_SIZE": str(result_json.get("dcp_size", 1)),
        "PCP_SIZE": str(result_json.get("pcp_size", 1)),
        "DP_ATTENTION": str(result_json.get("dp_attention", "false")),
    }


def truncate(records, seconds):
    """Keep records that both started and finished within the first N seconds."""
    starts = [int(r["metadata"]["request_start_ns"]) for r in records]
    if not starts:
        return []
    t0 = min(starts)
    cutoff = t0 + int(seconds * 1e9)
    return [r for r in records if int(r["metadata"]["request_end_ns"]) <= cutoff]


def row_for(records, aggregate, server_metrics, env):
    agg = build_result(records, aggregate, server_metrics, env)
    metrics = agg["request_metrics"]
    latency, throughput = metrics["latency"], metrics["throughput"]
    return {
        "n": len(records),
        "secs": throughput["duration_seconds"],
        "itl_p50": latency["itl"]["p50"] * 1000,
        "itl_p90": latency["itl"]["p90"] * 1000,
        "intvty_p90": latency["intvty"]["p90"],
        "ttft_p90": latency["ttft"]["p90"],
        "tput": throughput["total"]["tokens_per_second"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--windows", default="600,1200,1800,3600")
    args = parser.parse_args()
    windows = [int(w) for w in args.windows.split(",")]

    head = (
        f"{'run':<34}{'window':>8}{'reqs':>7}{'actual_s':>10}"
        f"{'ITLp50':>9}{'ITLp90':>9}{'P90intvty':>11}{'TTFTp90':>9}{'tot tok/s':>11}"
    )
    print(head)
    print("-" * len(head))

    for root in args.dirs:
        for result_path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
            if "aiperf_artifacts" in result_path:
                continue
            with open(result_path) as fh:
                result_json = json.load(fh)
            if "request_metrics" not in result_json:
                continue
            result_dir = Path(os.path.dirname(result_path))
            artifact_dir = resolve_artifact_dir(result_dir)
            jsonl = artifact_dir / "profile_export.jsonl"
            if not jsonl.exists():
                continue

            records, _ = load_records_with_accounting(jsonl)
            if len(records) < 2:
                continue
            # A point that died in warmup leaves the jsonl but no aggregate.
            aggregate_path = artifact_dir / "profile_export_aiperf.json"
            if not aggregate_path.exists():
                continue
            aggregate = load_aggregate(aggregate_path)
            server_metrics = load_server_metrics(artifact_dir / "server_metrics_export.json")
            env = load_env_from_result(result_json)

            label = f"{os.path.basename(root)[-18:]} c{result_json['conc']}"
            # The full window is the reference every shorter one is scored
            # against, so compute it before the loop rather than letting the
            # first (shortest) row stand in for it.
            baseline = row_for(records, aggregate, server_metrics, env)
            for window in windows:
                subset = truncate(records, window)
                if len(subset) < 2:
                    continue
                row = row_for(subset, aggregate, server_metrics, env)
                dt = 100 * (row["tput"] / baseline["tput"] - 1)
                di = 100 * (row["intvty_p90"] / baseline["intvty_p90"] - 1)
                delta = f"  tput{dt:+6.1f}%  intvty{di:+6.1f}%"
                print(
                    f"{label:<34}{window:>8}{row['n']:>7}{row['secs']:>10.0f}"
                    f"{row['itl_p50']:>9.2f}{row['itl_p90']:>9.2f}{row['intvty_p90']:>11.1f}"
                    f"{row['ttft_p90']:>9.2f}{row['tput']:>11.0f}{delta}"
                )
            print()


if __name__ == "__main__":
    main()
