#!/usr/bin/env python3
"""Summarize InferenceX AgentX result JSONs onto the board's two axes.

The board plots P90 interactivity (tok/s/user, = 1/ITL_p90) against total tokens
per $1 USD, where tokens/$ = per-GPU total throughput x 3600 / TCO($/chip/hr).
MI355X is $1.50/chip/hr in SemiAnalysis's July 2026 owning-hyperscaler model.

    ./ix_agentx_summarize.py <result-dir> [more dirs...]
"""
import argparse
import glob
import json
import os

TCO_PER_CHIP_HR = {"mi355x": 1.50, "b200": 1.73, "b300": 2.26}


def load(root, label=None):
    rows = []
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        if "aiperf_artifacts" in path:
            continue
        try:
            with open(path) as fh:
                blob = json.load(fh)
            metrics = blob["request_metrics"]
        except (json.JSONDecodeError, KeyError, OSError):
            continue
        latency, throughput = metrics["latency"], metrics["throughput"]
        per_gpu = throughput["per_gpu"]["total_tput_tps"]
        rows.append(
            {
                "label": label or os.path.basename(root.rstrip("/")),
                "tp": blob["tp"],
                "ep": blob.get("ep", 1),
                "conc": blob["conc"],
                "p90_intvty": latency["intvty"]["p90"],
                "itl_p50_ms": latency["itl"]["p50"] * 1000,
                "itl_p90_ms": latency["itl"]["p90"] * 1000,
                "ttft_p90": latency["ttft"]["p90"],
                "per_gpu_tps": per_gpu,
                "mtok_per_usd": per_gpu * 3600 / TCO_PER_CHIP_HR["mi355x"] / 1e6,
                "ok": blob["num_requests_successful"],
                "total": blob["num_requests_total"],
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dirs", nargs="+")
    parser.add_argument("--label", action="append", default=None)
    args = parser.parse_args()

    rows = []
    for i, d in enumerate(args.dirs):
        label = args.label[i] if args.label and i < len(args.label) else None
        rows.extend(load(d, label))
    rows.sort(key=lambda r: (r["label"], r["conc"]))

    head = (
        f"{'run':<26}{'cfg':<10}{'conc':>5}{'ITLp50':>9}{'ITLp90':>9}"
        f"{'P90 intvty':>12}{'TTFTp90':>9}{'per-GPU':>9}{'Mtok/$':>9}{'req ok':>11}"
    )
    print(head)
    print("-" * len(head))
    for r in rows:
        print(
            f"{r['label']:<26}TP{r['tp']}/EP{r['ep']:<5}{r['conc']:>5}"
            f"{r['itl_p50_ms']:>9.2f}{r['itl_p90_ms']:>9.2f}{r['p90_intvty']:>12.1f}"
            f"{r['ttft_p90']:>9.2f}{r['per_gpu_tps']:>9.0f}{r['mtok_per_usd']:>9.2f}"
            f"{r['ok']:>7}/{r['total']:<4}"
        )


if __name__ == "__main__":
    main()
