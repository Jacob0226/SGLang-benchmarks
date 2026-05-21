#!/usr/bin/env python3
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/home/jacchang/SGLang-benchmarks/PrefillKernel")
PIVOT_CSV = ROOT / "input_throughput_comparison_pivot.csv"


def load_rows():
    with PIVOT_CSV.open(newline="") as f:
        return list(csv.DictReader(f))


def get_series(rows, page_size):
    fp8off = None
    pr = None
    for r in rows:
        if int(r["page_size"]) != page_size:
            continue
        if r["method"] == "SGLANG_AITER_FP8_PREFILL_ATTN=0":
            fp8off = r
        elif r["method"] == "PR25556":
            pr = r

    if fp8off is None or pr is None:
        raise RuntimeError(f"Missing rows for page_size={page_size}")

    concs = [4, 16, 64, 256]
    base = np.array([float(fp8off[f"conc_{c}"]) for c in concs], dtype=float)
    pr_vals = np.array([float(pr[f"conc_{c}"]) for c in concs], dtype=float)
    ratio_pct = (pr_vals / base) * 100.0
    return concs, base, pr_vals, ratio_pct


def plot_one(page_size, concs, base, pr_vals, ratio_pct):
    fig, ax1 = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(concs))

    # Left axis: ratio bars
    bars = ax1.bar(
        x,
        ratio_pct,
        width=0.55,
        color="#7DA6FF",
        alpha=0.7,
        label="PR25556 / fp8off (%)",
        zorder=2,
    )
    ax1.axhline(100.0, color="gray", linestyle="--", linewidth=1.2, zorder=1)
    ax1.set_ylabel("PR25556 / fp8off (%)")
    ax1.set_ylim(0, max(120, float(np.max(ratio_pct) * 1.15)))
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(c) for c in concs])
    ax1.set_xlabel("Concurrency")
    ax1.grid(axis="y", linestyle=":", alpha=0.3, zorder=0)

    # Right axis: throughput lines
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
    ymax = float(max(np.max(base), np.max(pr_vals)) * 1.15)
    ax2.set_ylim(0, ymax)
    ax2.set_ylabel("Input token throughput (tok/s)")

    title = (
        f"Model: DS-R1-0528 FP8-TP8 | "
        f"Page Size = {page_size}, InputLength=4096: Ratio + Throughput"
    )
    ax1.set_title(title)

    # Joint legend
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    # Annotate ratio on bars
    for b, r in zip(bars, ratio_pct):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 1.2,
            f"{r:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    out = ROOT / f"page{page_size}_ratio_throughput.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    rows = load_rows()
    outputs = []
    for ps in (1, 64):
        concs, base, pr_vals, ratio_pct = get_series(rows, ps)
        outputs.append(plot_one(ps, concs, base, pr_vals, ratio_pct))
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
