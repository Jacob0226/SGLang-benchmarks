#!/usr/bin/env python3
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jacchang/SGLang-benchmarks/PrefillKernel")
DATA_ROOT = Path("/home/jacchang/plots/pagesize_pr25556_matrix_20260521_025634")
INPUT_LEN = 4096
FIGSIZE = (9.5, 6.2)
SUBPLOTS_ADJUST = {"left": 0.10, "right": 0.90, "top": 0.88, "bottom": 0.20}
CONCS = [4, 16, 64, 256]
CASES = {
    "page1_fp8off": {"page_size": 1, "method": "SGLANG_AITER_FP8_PREFILL_ATTN=0"},
    "page1_pr25556": {"page_size": 1, "method": "PR25556"},
    "page64_fp8off": {"page_size": 64, "method": "SGLANG_AITER_FP8_PREFILL_ATTN=0"},
    "page64_pr25556": {"page_size": 64, "method": "PR25556"},
}


def load_ttft_data():
    rows = []
    for case, meta in CASES.items():
        for c in CONCS:
            p = DATA_ROOT / case / f"bench_in{INPUT_LEN}_out1_conc{c}.jsonl"
            with p.open() as f:
                obj = json.loads(f.readline())
            rows.append(
                {
                    "case": case,
                    "method": meta["method"],
                    "page_size": meta["page_size"],
                    "concurrency": c,
                    "mean_ttft_ms": float(obj["mean_ttft_ms"]),
                }
            )
    return rows


def write_csv(rows):
    out = ROOT / "ttft_comparison.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["case", "method", "page_size", "concurrency", "mean_ttft_ms"]
        )
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["mean_ttft_ms"] = f"{r2['mean_ttft_ms']:.3f}"
            w.writerow(r2)
    return out


def build_series(rows, page_size):
    fp8 = {r["concurrency"]: r["mean_ttft_ms"] for r in rows if r["page_size"] == page_size and r["method"] == "SGLANG_AITER_FP8_PREFILL_ATTN=0"}
    pr = {r["concurrency"]: r["mean_ttft_ms"] for r in rows if r["page_size"] == page_size and r["method"] == "PR25556"}
    base = np.array([fp8[c] for c in CONCS], dtype=float)
    pr_vals = np.array([pr[c] for c in CONCS], dtype=float)
    ratio_pct = (pr_vals / base) * 100.0
    return base, pr_vals, ratio_pct


def plot_one(page_size, base, pr_vals, ratio_pct):
    fig, ax1 = plt.subplots(figsize=FIGSIZE)
    x = np.arange(len(CONCS))

    bars = ax1.bar(
        x,
        ratio_pct,
        width=0.55,
        color="#7DA6FF",
        alpha=0.7,
        label="mla_fp8_prefill_attn / flash_attn_varlen_func (%)",
        zorder=2,
    )
    ax1.axhline(100.0, color="gray", linestyle="--", linewidth=1.2, zorder=1)
    ax1.set_ylabel("mla_fp8_prefill_attn / flash_attn_varlen_func (%)")
    ax1.set_ylim(0, max(120, float(np.max(ratio_pct) * 1.15)))
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in CONCS])
    ax1.set_xlabel("Concurrency")
    ax1.grid(axis="y", linestyle=":", alpha=0.3, zorder=0)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        base,
        marker="o",
        linewidth=2.0,
        color="#D55E00",
        label="flash_attn_varlen_func",
    )
    ax2.plot(
        x,
        pr_vals,
        marker="o",
        linewidth=2.0,
        color="#009E73",
        label="mla_fp8_prefill_attn",
    )
    ymin = float(min(np.min(base), np.min(pr_vals)) * 0.9)
    ymax = float(max(np.max(base), np.max(pr_vals)) * 1.1)
    ax2.set_ylim(ymin, ymax)
    ax2.set_ylabel("TTFT (ms)")

    ax1.set_title(
        f"Model: DS-R1-0528 FP8-TP8 | "
        f"Page Size = {page_size}, InputLength={INPUT_LEN}: Ratio + TTFT"
    )

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(
        h1 + h2,
        l1 + l2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=3,
    )

    for b, r in zip(bars, ratio_pct):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.2,
            f"{r:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.subplots_adjust(**SUBPLOTS_ADJUST)
    out = ROOT / f"page{page_size}_ratio_ttft.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    rows = load_ttft_data()
    csv_path = write_csv(rows)
    print(csv_path)
    for ps in (1, 64):
        base, pr_vals, ratio = build_series(rows, ps)
        print(plot_one(ps, base, pr_vals, ratio))


if __name__ == "__main__":
    main()
