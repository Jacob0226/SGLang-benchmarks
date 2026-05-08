#!/usr/bin/env python3
"""Plot per-round TTFT / cache-hit-rate from HiCache.sh multiturn runs.

Reproduces the "Per-turn Performance" figure style of the Mooncake
HiCache benchmark page (https://kvcache-ai.github.io/Mooncake/
performance/sglang-hicache-benchmark-results-v1.html), but with multiple
platform tags overlaid so MI355X-vs-B200 cascade through L1 → L2 → L3
is visible at a glance.

Inputs:
  Walks ~/SGLang-benchmarks/results/ for bench_multiturn.jsonl files
  produced by HiCache.sh, grouped by (tag, cache_mode, hicache_size).
  Each file's "round" dict provides per-round average_ttft and
  cache_hit_rate that get plotted as one curve.

Usage:
  # Plot all tags that contain "MI355X" or "B200":
  python3 plot_cascade.py --tags MI355X_cascade B200_cascade --out cascade.png

  # Plot only specific cache_modes (drop noise):
  python3 plot_cascade.py --tags MI355X_cascade B200_cascade \
                          --cache-modes radix hicache hicache_file \
                          --hicache-size 192 \
                          --out cascade.png

  # Just one platform, all cache_modes:
  python3 plot_cascade.py --tags MI355X_cascade --out mi355x.png
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib not installed; install via 'pip install matplotlib'", file=sys.stderr)
    sys.exit(1)


PATH_RE = re.compile(
    r"results/(?P<docker>[^/]+)/"
    r"(?P<model_tag>[^/]+)-HiCache(?:-(?P<tag>[^/]+))?/"
    r"(?P<cache_mode>[^/]+)"
    r"(?:/size_(?P<size>\d+))?/bench_multiturn\.jsonl$"
)

CACHE_MODE_ORDER = [
    "no_cache", "L1", "L2", "L3_file", "L3_hf3fs", "L3_mooncake",
    # Legacy names from older runs (kept so the parser doesn't drop them):
    "no_radix", "radix", "hicache", "hicache_file",
    "hicache_hf3fs", "hicache_mooncake",
]
CACHE_MODE_LABEL = {
    "no_cache":         "no cache",
    "L1":               "L1 (GPU radix)",
    "L2":               "L1+L2",
    "L3_file":          "L1+L2+L3 (file)",
    "L3_hf3fs":         "L1+L2+L3 (hf3fs)",
    "L3_mooncake":      "L1+L2+L3 (Mooncake)",
    # Legacy aliases:
    "no_radix":         "no cache",
    "radix":            "L1 (GPU radix)",
    "hicache":          "L1+L2",
    "hicache_file":     "L1+L2+L3 (file)",
    "hicache_hf3fs":    "L1+L2+L3 (hf3fs)",
    "hicache_mooncake": "L1+L2+L3 (Mooncake)",
}
# Distinct color per cache_mode; line style per platform tag.
CACHE_MODE_COLOR = {
    "no_cache":         "#888888",
    "L1":               "#1f77b4",
    "L2":               "#ff7f0e",
    "L3_file":          "#2ca02c",
    "L3_hf3fs":         "#d62728",
    "L3_mooncake":      "#9467bd",
    "no_radix":         "#888888",
    "radix":            "#1f77b4",
    "hicache":          "#ff7f0e",
    "hicache_file":     "#2ca02c",
    "hicache_hf3fs":    "#d62728",
    "hicache_mooncake": "#9467bd",
}
TAG_LINESTYLE = {
    0: ("solid",  "o"),  # first tag
    1: ("dashed", "s"),  # second tag
    2: ("dotted", "^"),
    3: ("dashdot", "D"),
}


def parse_path(path: Path, root: Path) -> dict | None:
    abs_str = str(path.resolve())
    idx = abs_str.find("/results/")
    if idx == -1:
        return None
    sub = abs_str[idx + 1:]
    m = PATH_RE.search(sub)
    if not m:
        return None
    return {
        "docker":     m.group("docker"),
        "model":      m.group("model_tag"),
        "tag":        m.group("tag") or "",
        "cache_mode": m.group("cache_mode"),
        "size":       int(m.group("size")) if m.group("size") else None,
        "path":       path,
    }


def load_round_data(jsonl_path: Path) -> tuple[list[float], list[float]]:
    """Return (ttft_per_round_seconds, hit_rate_per_round_fraction)."""
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            r = rec.get("round") or {}
            keys = sorted([k for k in r if k.startswith("round_")],
                          key=lambda k: int(k.split("_")[1]))
            if not keys:
                continue
            ttft = [r[k]["average_ttft"]      for k in keys]
            hit  = [r[k]["cache_hit_rate"]    for k in keys]
            return ttft, hit
    return [], []


def discover_runs(root: Path, tags: list[str], cache_modes: list[str] | None,
                  hicache_size: int | None) -> list[dict]:
    runs = []
    for path in root.rglob("bench_multiturn.jsonl"):
        if path.stat().st_size == 0:
            continue
        meta = parse_path(path, root)
        if not meta:
            continue
        if tags and meta["tag"] not in tags:
            continue
        if cache_modes and meta["cache_mode"] not in cache_modes:
            continue
        if hicache_size is not None and meta["cache_mode"].startswith("hicache") \
                and meta["size"] != hicache_size:
            continue
        ttft, hit = load_round_data(path)
        if not ttft:
            continue
        meta["ttft"] = ttft
        meta["hit"] = hit
        runs.append(meta)
    return runs


def plot_cascade(runs: list[dict], out_path: Path, title: str | None = None):
    if not runs:
        raise SystemExit("No runs found matching filters")

    # Order: tag-major, cache_mode_rank-minor
    tag_list = []
    for r in runs:
        if r["tag"] not in tag_list:
            tag_list.append(r["tag"])
    runs.sort(key=lambda r: (
        tag_list.index(r["tag"]),
        CACHE_MODE_ORDER.index(r["cache_mode"]) if r["cache_mode"] in CACHE_MODE_ORDER else 99,
        r["size"] or 0,
    ))

    fig, (ax_ttft, ax_hit) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax_ttft.set_title("Prefill Performance (per round)")
    ax_ttft.set_xlabel("# Round")
    ax_ttft.set_ylabel("Avg TTFT (sec)")
    ax_ttft.grid(True, alpha=0.3)

    ax_hit.set_title("Cache Hit Rate (per round)")
    ax_hit.set_xlabel("# Round")
    ax_hit.set_ylabel("Cache Hit Rate (%)")
    ax_hit.set_ylim(-2, 102)
    ax_hit.grid(True, alpha=0.3)

    for r in runs:
        rounds = list(range(1, len(r["ttft"]) + 1))
        color = CACHE_MODE_COLOR.get(r["cache_mode"], "black")
        tag_idx = tag_list.index(r["tag"])
        linestyle, marker = TAG_LINESTYLE.get(tag_idx, ("solid", "o"))

        size_suffix = f"_{r['size']}" if r["size"] is not None else ""
        label = f"[{r['tag']}] {CACHE_MODE_LABEL.get(r['cache_mode'], r['cache_mode'])}{size_suffix}"

        ax_ttft.plot(rounds, r["ttft"], label=label,
                     color=color, linestyle=linestyle, marker=marker, markersize=6)
        ax_hit.plot(rounds, [h * 100 for h in r["hit"]], label=label,
                    color=color, linestyle=linestyle, marker=marker, markersize=6)

    # One shared legend below the plots.
    handles, labels = ax_ttft.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, -0.05),
               ncol=min(3, len(runs)), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96 if title else 1])
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(runs)} curves)")


def default_out_path(runs: list[dict], root: Path) -> Path:
    """Pick a sensible default output location:
      - 1 tag, 1 cache_mode → drop the PNG inside that bench folder
      - multiple tags or modes → drop a comparison PNG in <root>/cascade/
    """
    tags  = sorted({r["tag"] for r in runs})
    modes = sorted({r["cache_mode"] for r in runs})
    if len(tags) == 1 and len(modes) == 1:
        # use the parent of the bench dir (i.e. the size_N or cache_mode dir)
        return runs[0]["path"].parent / f"cascade_{tags[0]}_{modes[0]}.png"
    fname = "cascade_" + "_vs_".join(tags) + ".png"
    out_dir = root / "cascade"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


def main():
    p = argparse.ArgumentParser()
    default_root = Path(os.environ.get("HOME", "~")).expanduser() / "SGLang-benchmarks" / "results"
    p.add_argument("--root-dir", default=str(default_root),
                   help="Top-level results dir (default: %(default)s)")
    p.add_argument("--tags", nargs="+", required=True,
                   help="One or more --tag values to overlay (e.g. MI355X_cascade B200_cascade)")
    p.add_argument("--cache-modes", nargs="+", default=None,
                   help="Filter to specific cache modes (default: all found)")
    p.add_argument("--hicache-size", type=int, default=None,
                   help="If set, only plot hicache_* runs at this --hicache-size GB")
    p.add_argument("--title", default=None, help="Optional plot title")
    p.add_argument("--out", default=None,
                   help="Output PNG path (default: inside the run's bench folder "
                        "for single-tag/single-mode, else <root>/cascade/)")
    args = p.parse_args()

    root = Path(args.root_dir).expanduser().resolve()
    runs = discover_runs(root, args.tags, args.cache_modes, args.hicache_size)
    if not runs:
        raise SystemExit("No runs found matching filters")
    out_path = Path(args.out) if args.out else default_out_path(runs, root)
    plot_cascade(runs, out_path, title=args.title)


if __name__ == "__main__":
    main()
