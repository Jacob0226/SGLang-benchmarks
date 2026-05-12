#!/usr/bin/env python3
"""Plot per-round TTFT / cache-hit-rate from HiCache.sh / cascade_dsr1.sh
multiturn runs.

Reproduces the "Per-turn Performance" figure style of the Mooncake
HiCache benchmark page (https://kvcache-ai.github.io/Mooncake/
performance/sglang-hicache-benchmark-results-v1.html). Supports two
plotting layouts in one script:

  (A) Cross-platform compare: --MI355X X.jsonl --B200 Y.jsonl
      One curve per platform tag (typically same cache_mode).

  (B) Same-platform cascade compare: --B200 L1.jsonl L2.jsonl L3.jsonl
      One curve per cache_mode under a single platform tag. Curves are
      colored by cache_mode (L1 blue / L2 orange / L3_file green / ...)
      with distinct markers so the legend stays readable.

Inputs:
  --MI355X / --B200 each accept ONE OR MORE bench_multiturn.jsonl
  paths. The first JSON line's "round" dict provides per-round
  average_ttft and cache_hit_rate that get plotted as one curve per
  file. Cache mode + hicache size are auto-extracted from the path if
  it follows the conventional layout
    results/<docker>/<MODEL>-(HiCache|cascade)[-<tag>]/<cache_mode>/[size_N]/
  otherwise the parser falls back to scanning path segments for any
  known cache_mode token (L1 / L2 / L3_file / ...).

Paths:
  All paths (--MI355X / --B200 / --out) are resolved to absolute at
  parse time, so relative inputs still work but the rendered PNG always
  lands in a predictable spot regardless of cwd.

Usage:
  # Same platform, L1 vs L1+L2 vs L1+L2+L3 cascade (this user's typical case):
  python3 plot_cascade.py \
      --Title "DSR1-0528 B200 cascade" \
      --B200 $HOME/SGLang-benchmarks/results/.../L1/bench_multiturn.jsonl \
             $HOME/SGLang-benchmarks/results/.../L2/size_192/bench_multiturn.jsonl \
             $HOME/SGLang-benchmarks/results/.../L3_file/size_192/bench_multiturn.jsonl \
      --out  $HOME/SGLang-benchmarks/results/cascade_b200_3modes.png

  # Cross-platform compare (same cache_mode):
  python3 plot_cascade.py \
      --Title "DSR1-0528 HiCache L3_file (192 GB)" \
      --MI355X $HOME/SGLang-benchmarks/results/.../L3_file/size_192/bench_multiturn.jsonl \
      --B200   $HOME/SGLang-benchmarks/results/.../L3_file/size_192/bench_multiturn.jsonl \
      --out    $HOME/SGLang-benchmarks/results/cascade_L3_file_192.png
"""

from __future__ import annotations

import argparse
import json
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
    r"(?P<model_tag>[^/]+?)-(?:HiCache|cascade)(?:-(?P<tag>[^/]+))?/"
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


def parse_path(path: Path) -> dict | None:
    """Extract (tag, cache_mode, size) from a HiCache.sh / cascade_dsr1.sh
    results path. Tries the strict layout first, then falls back to scanning
    segments for any known cache_mode token so non-standard parents (e.g.
    custom result dirs) still light up cache_mode in the legend.

    Returns None only when the path is anywhere outside /results/.
    """
    abs_str = str(path.resolve())
    idx = abs_str.find("/results/")
    if idx == -1:
        return None
    sub = abs_str[idx + 1:]
    m = PATH_RE.search(sub)
    if m:
        return {
            "docker":     m.group("docker"),
            "model":      m.group("model_tag"),
            "tag":        m.group("tag") or "",
            "cache_mode": m.group("cache_mode"),
            "size":       int(m.group("size")) if m.group("size") else None,
            "path":       path,
        }
    # Fallback: walk segments looking for a recognized cache_mode and an
    # optional size_N sibling. Keeps the legend informative even when the
    # parent dir doesn't follow the -HiCache / -cascade naming.
    parts = sub.split("/")
    cache_mode = ""
    size = None
    for i, seg in enumerate(parts):
        if seg in CACHE_MODE_ORDER:
            cache_mode = seg
            if i + 1 < len(parts) and parts[i + 1].startswith("size_"):
                try:
                    size = int(parts[i + 1].split("_", 1)[1])
                except (ValueError, IndexError):
                    pass
            break
    if not cache_mode:
        return None
    return {
        "docker":     parts[1] if len(parts) > 1 else "",
        "model":      parts[2] if len(parts) > 2 else "",
        "tag":        "",
        "cache_mode": cache_mode,
        "size":       size,
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


# Per-token KV cost (bytes per rank) for known model families. MLA is
# replicated across TP, GQA is sharded — see tools/calc_kv_per_token.py.
KV_BYTES_PER_TOKEN_PER_RANK = {
    "deepseek": 35136,   # MLA: (kv_lora 512 + qk_rope 64) × 61 layers × 1 byte FP8
    "gpt-oss":  2304,    # GQA TP-sharded: 8 KV heads × 64 head_dim × 2 (K+V) × 18 full layers / 8 TP × 1 byte
}


def load_fill_thresholds(jsonl_path: Path) -> dict | None:
    """Read sibling bench_meta.json + server.log to compute when L1 / L1+L2
    fill across rounds. Returns None when the metadata is incomplete (e.g.
    older runs without bench_meta.json). When server.log has the real
    "KV Cache is allocated. ... KV size: X GB" line we use that as L1
    instead of bench_meta's estimate, since SGLang's hicache+aiter combo
    auto-reduces mem_fraction below the user-specified value."""
    bench_dir = jsonl_path.parent
    meta_path = bench_dir / "bench_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception:
        return None

    # Per-rank GiB added per round = clients × req_len × kv_bytes / 2^30
    family = meta.get("model_family", "deepseek")
    kv_bytes = KV_BYTES_PER_TOKEN_PER_RANK.get(family, 35136)
    nc = meta.get("num_clients", 0)
    rl = meta.get("request_length", 0)
    if nc <= 0 or rl <= 0:
        return None
    round_inc_gib = nc * rl * kv_bytes / (1024 ** 3)

    # L1: prefer the actual "KV size" from server.log if present.
    l1_gib = None
    server_log = bench_dir / "server.log"
    if server_log.exists():
        try:
            with open(server_log, "r", errors="ignore") as f:
                for ln in f:
                    if "KV Cache is allocated" in ln and "KV size:" in ln:
                        # "[TPx] KV Cache is allocated. #tokens: N, KV size: X GB"
                        try:
                            l1_gib = float(ln.split("KV size:")[1].split("GB")[0].strip())
                            break
                        except Exception:
                            pass
        except Exception:
            pass
    if l1_gib is None:
        # Fall back to bench_meta's pre-launch estimate (often optimistic).
        l1_gib = float(meta.get("device_pool_gb") or 0)

    l2_gib = float(meta.get("hicache_size_gb") or 0)

    return {
        "round_inc_gib": round_inc_gib,
        "l1_gib": l1_gib,
        "l2_gib": l2_gib,
        "l1_fill_round":      l1_gib / round_inc_gib if round_inc_gib else None,
        "l1_l2_fill_round":  (l1_gib + l2_gib) / round_inc_gib if round_inc_gib else None,
    }


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

    # Two panels stacked vertically (TTFT on top, Cache Hit Rate below) so
    # both share the same X-axis width and any vertical fill-threshold lines
    # in the top panel align visually with the corresponding round in the
    # bottom panel. sharex=True hides the redundant top-panel x-tick labels.
    # constrained_layout handles the legend / suptitle spacing automatically
    # and is sharex/gridspec-friendly (unlike tight_layout, which warns).
    fig, (ax_ttft, ax_hit) = plt.subplots(
        2, 1, figsize=(11, 8.5), sharex=True,
        gridspec_kw={"hspace": 0.18},
        constrained_layout=True,
    )
    ax_ttft.set_title("Prefill Performance (per round)")
    ax_ttft.set_ylabel("Avg TTFT (sec)")
    ax_ttft.grid(True, alpha=0.3)

    ax_hit.set_title("Cache Hit Rate (per round)")
    ax_hit.set_xlabel("# Round")
    ax_hit.set_ylabel("Cache Hit Rate (%)")
    ax_hit.set_ylim(-2, 102)
    ax_hit.grid(True, alpha=0.3)

    # Show every round on the x-axis (no stride). Use the longest run's
    # round count so single-platform and cross-platform plots both label
    # rounds 1..N individually rather than matplotlib's default 2/4/6/...
    max_rounds = max(len(r["ttft"]) for r in runs)
    xticks = list(range(1, max_rounds + 1))
    ax_ttft.set_xticks(xticks)
    ax_hit.set_xticks(xticks)

    # Style strategy depends on how many tags vs cache_modes are present:
    #   - single tag + multi mode  → color by cache_mode (typical L1/L2/L3
    #                                  cascade on one platform)
    #   - multi tag + single mode  → color by platform (NVIDIA green vs AMD
    #                                  orange cross-platform compare)
    #   - multi tag + multi mode   → color by platform, linestyle by mode
    #                                  index within that platform
    tag_palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]

    def color_for_tag(tag: str) -> str | None:
        t = tag.lower()
        if any(p in t for p in ("b200", "h200", "h800", "h100", "a100", "nvidia")):
            return "#2ca02c"  # green for NVIDIA
        if any(p in t for p in ("mi355", "mi325", "mi300", "mi250", "mi210", "amd", "rocm")):
            return "#ff7f0e"  # orange for AMD
        return None

    modes_per_tag: dict[str, list[str]] = {}
    for r in runs:
        modes_per_tag.setdefault(r["tag"], [])
        if r["cache_mode"] not in modes_per_tag[r["tag"]]:
            modes_per_tag[r["tag"]].append(r["cache_mode"])
    multi_tag = len(tag_list) > 1
    multi_mode_some_tag = any(len(v) > 1 for v in modes_per_tag.values())
    markers = ("o", "s", "^", "D", "v", "P")

    # Same-platform multi-mode runs share the same model/clients/req_len, so
    # their L1 / L1+L2 fill rounds collapse onto a single x-position per
    # threshold. Track which rounded rounds we've already drawn so labels
    # don't stack on top of each other. Cross-platform runs naturally won't
    # collide here (different GPUs → different l1_gib → different rounds).
    seen_l1_fill_rounds: set[float] = set()
    seen_l1l2_fill_rounds: set[float] = set()

    for r in runs:
        rounds = list(range(1, len(r["ttft"]) + 1))
        tag_idx = tag_list.index(r["tag"])
        mode_idx_in_tag = modes_per_tag[r["tag"]].index(r["cache_mode"])

        if not multi_tag and multi_mode_some_tag:
            color = (CACHE_MODE_COLOR.get(r["cache_mode"])
                     or tag_palette[mode_idx_in_tag % len(tag_palette)])
            linestyle = "solid"
            marker = markers[mode_idx_in_tag % len(markers)]
        elif multi_tag and multi_mode_some_tag:
            color = (color_for_tag(r["tag"])
                     or tag_palette[tag_idx % len(tag_palette)])
            linestyle, marker = TAG_LINESTYLE.get(mode_idx_in_tag, ("solid", "o"))
        else:
            color = (color_for_tag(r["tag"])
                     or CACHE_MODE_COLOR.get(r["cache_mode"])
                     or tag_palette[tag_idx % len(tag_palette)])
            linestyle, marker = "solid", markers[tag_idx % len(markers)]

        # Vertical lines marking when L1 and L1+L2 are predicted to fill,
        # so the reader can verify TTFT spikes line up with cache-tier
        # boundary crossings instead of guessing. Labels are staggered
        # per tag to avoid overlapping when two platforms fill at nearby
        # rounds (typical for MI355X-vs-B200 L1 events). Dedup on rounded
        # round so the SAME threshold (collapsed L1 fill across L1/L2/L3
        # cascade modes on one platform) isn't redrawn 3 times.
        thr = r.get("fill_thresholds")
        if thr:
            l1_r = thr.get("l1_fill_round")
            ll_r = thr.get("l1_l2_fill_round")
            tag_label = r["tag"]
            top_y = 0.92 - 0.08 * (tag_idx % 2)
            bot_y = 0.04 + 0.08 * (tag_idx % 2)
            if l1_r and 1 <= l1_r <= max_rounds:
                key = round(l1_r, 1)
                if key not in seen_l1_fill_rounds:
                    seen_l1_fill_rounds.add(key)
                    line_color = color if multi_tag else "#555555"
                    label_pre = f" {tag_label} " if multi_tag else " "
                    ax_ttft.axvline(l1_r, color=line_color, linestyle=":", alpha=0.55, linewidth=1.2)
                    ax_ttft.text(l1_r, bot_y,
                                 f"{label_pre}L1 fill (r={l1_r:.1f})",
                                 transform=ax_ttft.get_xaxis_transform(),
                                 color=line_color, fontsize=8, va="bottom", ha="left", alpha=0.95)
            if ll_r and 1 <= ll_r <= max_rounds:
                key = round(ll_r, 1)
                if key not in seen_l1l2_fill_rounds:
                    seen_l1l2_fill_rounds.add(key)
                    line_color = color if multi_tag else "#555555"
                    label_pre = f" {tag_label} " if multi_tag else " "
                    ax_ttft.axvline(ll_r, color=line_color, linestyle="--", alpha=0.55, linewidth=1.2)
                    ax_ttft.text(ll_r, top_y,
                                 f"{label_pre}L1+L2 fill (r={ll_r:.1f})",
                                 transform=ax_ttft.get_xaxis_transform(),
                                 color=line_color, fontsize=8, va="top", ha="left", alpha=0.95)

        mode_label = CACHE_MODE_LABEL.get(r["cache_mode"], r["cache_mode"])
        size_suffix = f" ({r['size']} GB)" if r["size"] is not None else ""
        label = f"[{r['tag']}] {mode_label}{size_suffix}".strip()

        ax_ttft.plot(rounds, r["ttft"], label=label,
                     color=color, linestyle=linestyle, marker=marker, markersize=6)
        ax_hit.plot(rounds, [h * 100 for h in r["hit"]], label=label,
                    color=color, linestyle=linestyle, marker=marker, markersize=6)

    # One shared legend below the plots. With constrained_layout the figure
    # automatically reserves space for the legend, so a positive y offset
    # keeps it inside the saved canvas instead of relying on bbox_inches.
    handles, labels = ax_ttft.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center",
               ncol=min(3, len(runs)), fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(runs)} curves)")


def main():
    p = argparse.ArgumentParser(
        description="Plot per-round TTFT / cache-hit-rate from MI355X vs B200 "
                    "bench_multiturn.jsonl files (HiCache.sh output).",
    )
    p.add_argument("--Title", default=None, help="Plot title")
    p.add_argument("--MI355X", nargs="+", default=None,
                   help="One or more MI355X bench_multiturn.jsonl paths")
    p.add_argument("--B200", nargs="+", default=None,
                   help="One or more B200 bench_multiturn.jsonl paths")
    p.add_argument("--out", required=True, help="Output PNG path")
    args = p.parse_args()

    sources = [
        ("MI355X", args.MI355X or []),
        ("B200",   args.B200   or []),
    ]
    runs: list[dict] = []
    for tag, jsonl_paths in sources:
        for jsonl_path in jsonl_paths:
            p_jsonl = Path(jsonl_path).expanduser()
            if not p_jsonl.is_file():
                sys.exit(f"ERROR: file not found: {jsonl_path}")
            if p_jsonl.stat().st_size == 0:
                sys.exit(f"ERROR: empty file: {jsonl_path}")

            # Recover cache_mode / size from the path. parse_path() handles
            # both the strict <MODEL>-(HiCache|cascade)/<mode>[/size_N]
            # layout and a permissive fallback that just scans segments for
            # a known cache_mode token.
            meta = parse_path(p_jsonl) or {
                "tag": "", "cache_mode": "", "size": None, "path": p_jsonl,
            }
            meta["tag"] = tag  # user-specified --MI355X / --B200 always wins

            ttft, hit = load_round_data(p_jsonl)
            if not ttft:
                sys.exit(f"ERROR: no per-round data in {jsonl_path}")
            meta["ttft"] = ttft
            meta["hit"] = hit
            meta["fill_thresholds"] = load_fill_thresholds(p_jsonl)
            runs.append(meta)

    if not runs:
        sys.exit("ERROR: pass at least one of --MI355X / --B200")

    # Always resolve --out to an absolute path so the PNG lands in a
    # predictable spot regardless of cwd. Relative inputs still work,
    # they just get attached to the cwd at parse time.
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plot_cascade(runs, out_path, title=args.Title)


if __name__ == "__main__":
    main()
