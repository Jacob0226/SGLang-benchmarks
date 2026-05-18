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

Style (fixed):
  Color = platform family. B200 = green family, MI355X = orange family.
  Lightness goes dark (L1) -> light (L3+) so the curve color encodes both
  vendor and cache-tier depth at a glance:
      B200:   L1 dark green   / L1+L2 green        / L1+L2+L3 light green
      MI355X: L1 dark orange  / L1+L2 orange       / L1+L2+L3 light orange
  Marker = cache_mode. L1 = circle, L2 = square, L3 = triangle (same marker
  across both platforms so tier lines up visually even at low contrast).
  Other tags (H200, MI325, ...) fall back to a generic palette.

Outputs:
  --out <name>.png provides only the BASENAME; each PNG is auto-placed
  in the appropriate cascade root dir so each platform's results folder
  stays self-contained:
      <MI355X-root>/<name>.png         combined plot (up to 6 lines)
      <B200-root>/<name>.B200.png      just the B200 curves
      <MI355X-root>/<name>.MI355X.png  just the MI355X curves
  Combined plot defaults to MI355X root (typical "I'm on the MI355X box"
  workflow); falls back to B200 root if no MI355X data, then to --out's
  parent for non-MI/B platforms. Per-platform plots are skipped if that
  platform has no runs.

Usage:
  # Same platform, L1 vs L1+L2 vs L1+L2+L3 cascade (this user's typical case):
  python3 plot_cascade.py \
      --Title "DSR1-0528 B200 cascade" \
      --B200 $HOME/SGLang-benchmarks/results/.../L1/bench_multiturn.jsonl \
             $HOME/SGLang-benchmarks/results/.../L2/size_192/bench_multiturn.jsonl \
             $HOME/SGLang-benchmarks/results/.../L3_file/size_192/bench_multiturn.jsonl \
      --out  $HOME/SGLang-benchmarks/results/cascade_b200_3modes.png

  # Cross-platform 3x3 cascade compare (produces 3 PNGs):
  python3 plot_cascade.py \
      --Title "DSR1-0528 cascade" \
      --B200 .../B200_cascade/{L1,L2/size_192,L3_file/size_192}/bench_multiturn.jsonl \
      --MI355X .../MI355X_cascade/{L1,L2/size_192,L3_file/size_192}/bench_multiturn.jsonl \
      --out $HOME/SGLang-benchmarks/results/cascade_compare.png
  # → cascade_compare.png + cascade_compare.B200.png + cascade_compare.MI355X.png
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

# Color = platform; lightness within a platform = cache tier depth.
# B200 = green family, MI355X = orange family, both go dark (L1) -> light
# (L3+) so a glance at the line color tells you both vendor and tier depth.
# Same family per platform across all 3 tiers makes legend reading easy
# even at thumbnail scale (e.g. on slides).
PLATFORM_PALETTE = {
    "B200": {
        # Hue shift: dark green (L1) -> warm green (L2) -> lime (L3+) so the
        # three tiers stay distinguishable on small thumbnails / projector.
        "no_cache":         "#0d3311",  # very dark green (rare)
        "L1":               "#1b5e20",  # dark green
        "L2":               "#2e7d32",  # forest green (slightly darker than before)
        "L3_file":          "#9ccc65",  # lime green (clear contrast vs L2)
        "L3_hf3fs":         "#c5e1a5",
        "L3_mooncake":      "#dcedc8",
        # Legacy aliases:
        "no_radix":         "#0d3311",
        "radix":             "#1b5e20",
        "hicache":           "#2e7d32",
        "hicache_file":      "#9ccc65",
        "hicache_hf3fs":     "#c5e1a5",
        "hicache_mooncake":  "#dcedc8",
    },
    "MI355X": {
        # Hue shift: dark red (L1) -> orange (L2) -> amber/yellow (L3+) so
        # the lightest tier doesn't look like a faded version of L2.
        "no_cache":         "#7f1d00",
        "L1":               "#bf360c",  # dark orange / red-brown
        "L2":               "#ef6c00",  # warm orange (deeper)
        "L3_file":          "#ffd54f",  # amber (yellow-toned, clear gap from L2)
        "L3_hf3fs":         "#ffe082",
        "L3_mooncake":      "#fff59d",
        "no_radix":         "#7f1d00",
        "radix":             "#bf360c",
        "hicache":           "#ef6c00",
        "hicache_file":      "#ffd54f",
        "hicache_hf3fs":     "#ffe082",
        "hicache_mooncake":  "#fff59d",
    },
}

# Marker = cache_mode (consistent across platforms so a reader can match
# tier across the green and orange lines at a glance, instead of having
# to read the legend every time).
MODE_MARKER = {
    "no_cache":         "x",
    "L1":               "o",  # circle
    "L2":               "s",  # square
    "L3_file":          "^",  # triangle up
    "L3_hf3fs":         "D",  # diamond
    "L3_mooncake":      "P",  # plus
    # Legacy aliases:
    "no_radix":         "x",
    "radix":             "o",
    "hicache":           "s",
    "hicache_file":      "^",
    "hicache_hf3fs":     "D",
    "hicache_mooncake":  "P",
}

# Fallback palette for non-B200 / non-MI355X tags (e.g. user adds a
# third platform). Keeps the script generic.
FALLBACK_PALETTE = ["#1f77b4", "#9467bd", "#17becf", "#bcbd22", "#8c564b"]


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
    write_policy = meta.get("hicache_write_policy", "write_through")

    # When there's no L2 pool (L1-only run, hicache_size_gb is null), the
    # L2-side fill round collapses onto "L1 fill" and adds no information —
    # suppress it so the plot doesn't draw two overlapping labels.
    #
    # HiCache write policy controls whether L1 and L2 hold the SAME blocks
    # (write_through: L2 mirrors L1, every L1 write is replicated to L2 →
    # effective cache cap = max(L1, L2) = L2 in well-sized setups) or
    # DISJOINT blocks (write_back: L1 evicts to L2 lazily → effective cap =
    # L1 + L2). cascade_dsr1.sh defaults to write_through and that's what
    # both reference runs use, so the hit-rate cliffs in the empirical plot
    # line up with the L2 boundary, not the L1+L2 sum.
    l1_l2_fill = None
    l1_l2_cap_label = "L1+L2 cap"
    if round_inc_gib and l2_gib > 0:
        if write_policy == "write_through":
            cap_gib = max(l1_gib, l2_gib)
            l1_l2_cap_label = "L2 cap"   # write_through: L2 alone is the cap
        else:
            cap_gib = l1_gib + l2_gib
            l1_l2_cap_label = "L1+L2 cap"  # write_back: tiers are disjoint
        l1_l2_fill = cap_gib / round_inc_gib

    return {
        "round_inc_gib": round_inc_gib,
        "l1_gib": l1_gib,
        "l2_gib": l2_gib,
        "write_policy": write_policy,
        "l1_l2_cap_label": l1_l2_cap_label,
        "l1_fill_round":     l1_gib / round_inc_gib if round_inc_gib else None,
        "l1_l2_fill_round":  l1_l2_fill,
    }


def plot_cascade(runs: list[dict], out_path: Path,
                 title: str | None = None,
                 ymax_ttft: float | None = None):
    """Render one TTFT + cache-hit-rate figure. ymax_ttft, when given,
    forces the TTFT panel's upper Y limit so multiple figures sharing
    the same workload land on the same scale (handy for side-by-side
    comparison of MI355X.png, B200.png, MI355X_VS_B200.png)."""
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
    if ymax_ttft is not None:
        ax_ttft.set_ylim(0, ymax_ttft)

    ax_hit.set_title("Cache Hit Rate (per round)")
    ax_hit.set_xlabel("# Round")
    ax_hit.set_ylabel("Cache Hit Rate (%)")
    ax_hit.set_ylim(-2, 102)
    ax_hit.grid(True, alpha=0.3)

    # Show every round on the x-axis (no stride). Use the longest run's
    # round count so single-platform and cross-platform plots both label
    # rounds 1..N individually rather than matplotlib's default 2/4/6/...
    # Force tick labels visible on the TOP panel too (sharex=True hides
    # them by default) so the reader can read round numbers next to TTFT
    # spikes without sliding their eye down to the hit-rate panel.
    max_rounds = max(len(r["ttft"]) for r in runs)
    xticks = list(range(1, max_rounds + 1))
    ax_ttft.set_xticks(xticks)
    ax_hit.set_xticks(xticks)
    ax_ttft.tick_params(labelbottom=True)
    ax_ttft.set_xlabel("# Round")

    # Fixed style scheme:
    #   color  = platform family (B200 green / MI355X orange), darker for
    #            shallower cache tier so L1 stands out vs L3+ at a glance.
    #   marker = cache_mode (consistent across platforms so the reader can
    #            line up tiers between green and orange curves).
    # Tags outside B200 / MI355X fall back to a generic palette so an
    # H200 / MI325 user can still get distinct colors.
    fallback_idx = 0
    fallback_for_tag: dict[str, str] = {}

    def style_for(tag: str, mode: str) -> tuple[str, str]:
        nonlocal fallback_idx
        pal = PLATFORM_PALETTE.get(tag)
        if pal and mode in pal:
            color = pal[mode]
        elif pal:
            color = next(iter(pal.values()))  # any from the platform palette
        else:
            if tag not in fallback_for_tag:
                fallback_for_tag[tag] = FALLBACK_PALETTE[fallback_idx % len(FALLBACK_PALETTE)]
                fallback_idx += 1
            color = fallback_for_tag[tag]
        marker = MODE_MARKER.get(mode, "o")
        return color, marker

    modes_per_tag: dict[str, list[str]] = {}
    for r in runs:
        modes_per_tag.setdefault(r["tag"], [])
        if r["cache_mode"] not in modes_per_tag[r["tag"]]:
            modes_per_tag[r["tag"]].append(r["cache_mode"])
    multi_tag = len(tag_list) > 1

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

        color, marker = style_for(r["tag"], r["cache_mode"])
        linestyle = "solid"

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
                    # Mirror the fill marker onto BOTH panels so the cause
                    # (cache tier saturates) lines up visually with the
                    # effect (hit-rate plateaus / TTFT spikes).
                    for ax in (ax_ttft, ax_hit):
                        ax.axvline(l1_r, color=line_color, linestyle=":", alpha=0.55, linewidth=1.2)
                    ax_ttft.text(l1_r, bot_y,
                                 f"{label_pre}L1 hard cap (r={l1_r:.1f})",
                                 transform=ax_ttft.get_xaxis_transform(),
                                 color=line_color, fontsize=8, va="bottom", ha="left", alpha=0.95)
            if ll_r and 1 <= ll_r <= max_rounds:
                key = round(ll_r, 1)
                if key not in seen_l1l2_fill_rounds:
                    seen_l1l2_fill_rounds.add(key)
                    line_color = color if multi_tag else "#555555"
                    label_pre = f" {tag_label} " if multi_tag else " "
                    cap_lbl = thr.get("l1_l2_cap_label", "L1+L2 cap")
                    for ax in (ax_ttft, ax_hit):
                        ax.axvline(ll_r, color=line_color, linestyle="--", alpha=0.55, linewidth=1.2)
                    ax_ttft.text(ll_r, top_y,
                                 f"{label_pre}{cap_lbl} (r={ll_r:.1f})",
                                 transform=ax_ttft.get_xaxis_transform(),
                                 color=line_color, fontsize=8, va="top", ha="left", alpha=0.95)

        mode_label = CACHE_MODE_LABEL.get(r["cache_mode"], r["cache_mode"])
        # Size suffix only on L2: that's the GB number that actually describes
        # the *L2 host pool*. For L3_file the same number is still the L2
        # host pool (the path is .../L3_file/size_<HICACHE_SIZE>/), but
        # writing "L1+L2+L3 (file) (320 GB)" misleadingly reads as "L3 has
        # 320 GB capacity" so we drop it for L3.
        if r["cache_mode"] == "L2" and r["size"] is not None:
            size_suffix = f" ({r['size']} GB)"
        else:
            size_suffix = ""
        label = f"[{r['tag']}] {mode_label}{size_suffix}".strip()

        ax_ttft.plot(rounds, r["ttft"], label=label,
                     color=color, linestyle=linestyle, marker=marker, markersize=6)
        ax_hit.plot(rounds, [h * 100 for h in r["hit"]], label=label,
                    color=color, linestyle=linestyle, marker=marker, markersize=6)

    # One shared legend below the plots. matplotlib fills columns top-to-bottom
    # (column-major), so legend layout depends on ncol vs item count:
    #   - multi-platform: one column per tag → each column reads top-down as
    #     L1 / L1+L2 / L1+L2+L3 for that platform; columns left→right by tag
    #     order (MI355X first, then B200, since main() loads MI355X first).
    #   - single-platform: one row, all modes left-to-right.
    handles, labels = ax_ttft.get_legend_handles_labels()
    n_tags = len(tag_list)
    if n_tags >= 2:
        legend_ncol = n_tags
    else:
        legend_ncol = min(len(runs), 6)
    fig.legend(handles, labels, loc="outside lower center",
               ncol=legend_ncol, fontsize=9)

    if title:
        fig.suptitle(title, fontsize=12)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Wrote {out_path} ({len(runs)} curves)")


MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun",
               "Jul","Aug","Sep","Oct","Nov","Dec"]

# Display-name shortenings for common model dirs. Add more as needed.
MODEL_DISPLAY_NAME = {
    "DeepSeek-R1-0528":         "DSR1-0528",
    "DeepSeek-R1":              "DSR1",
    "DeepSeek-V3":              "DSV3",
    "DeepSeek-V3.2-Exp":        "DSV3.2",
    "DeepSeek-R1-0528-MXFP4":   "DSR1-0528-MXFP4",
}


def discover_jsonls(cascade_dir: Path) -> list[Path]:
    """Walk a cascade root dir looking for bench_multiturn.jsonl under
    L1/, L2/size_*, L3_file/size_*, none/. Returns one jsonl per mode in
    CACHE_MODE_ORDER. When multiple size_* dirs exist under L2 or L3,
    picks the largest size."""
    found: list[tuple[str, int, Path]] = []
    for mode in CACHE_MODE_ORDER:
        mode_dir = cascade_dir / mode
        if not mode_dir.is_dir():
            continue
        # L1 / no_cache: no size subdir.
        direct = mode_dir / "bench_multiturn.jsonl"
        if direct.is_file() and direct.stat().st_size > 0:
            found.append((mode, 0, direct))
            continue
        # L2 / L3_file: pick the size_<N>/ with the largest N.
        size_dirs = sorted(
            (d for d in mode_dir.iterdir() if d.is_dir() and d.name.startswith("size_")),
            key=lambda d: int(d.name.split("_", 1)[1]) if d.name.split("_", 1)[1].isdigit() else 0,
            reverse=True,
        )
        for sd in size_dirs:
            j = sd / "bench_multiturn.jsonl"
            if j.is_file() and j.stat().st_size > 0:
                try:
                    sz = int(sd.name.split("_", 1)[1])
                except (ValueError, IndexError):
                    sz = 0
                found.append((mode, sz, j))
                break
    return [p for _, _, p in found]


def extract_date(cascade_dir: Path) -> str:
    """Return a short 'Mon DD' date for the cascade run. Tries (in order):
        1) 8-digit YYYYMMDD in the parent docker dir name (build date).
        2) 4-digit MMDD suffix in the tag (e.g. B200_15rounds_0512).
        3) Newest bench_multiturn.jsonl mtime."""
    docker_name = cascade_dir.parent.name if cascade_dir.parent else ""
    m = re.search(r"(20\d{2})(\d{2})(\d{2})", docker_name)
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= mo <= 12 and 1 <= da <= 31:
            return f"{MONTH_NAMES[mo-1]} {da:d}"
    m = re.search(r"_(\d{2})(\d{2})(?:_|$)", cascade_dir.name)
    if m:
        mo, da = int(m.group(1)), int(m.group(2))
        if 1 <= mo <= 12 and 1 <= da <= 31:
            return f"{MONTH_NAMES[mo-1]} {da:d}"
    import datetime
    newest = max(
        (j.stat().st_mtime for j in cascade_dir.rglob("bench_multiturn.jsonl") if j.is_file()),
        default=None,
    )
    if newest is not None:
        d = datetime.datetime.fromtimestamp(newest)
        return f"{MONTH_NAMES[d.month-1]} {d.day:d}"
    return ""


def extract_model_display(cascade_dir: Path, override: str | None = None) -> str:
    """Extract display-friendly model name from '<model>-cascade-<tag>'
    dir name. Honors --model-name override if given."""
    if override:
        return override
    name = cascade_dir.name
    raw = name.split("-cascade-", 1)[0] if "-cascade-" in name else name
    return MODEL_DISPLAY_NAME.get(raw, raw)


def load_runs(cascade_dir: Path, tag: str, max_rounds: int | None) -> list[dict]:
    """Discover + load all bench_multiturn.jsonl files under a cascade root."""
    jsonls = discover_jsonls(cascade_dir)
    if not jsonls:
        sys.exit(f"ERROR: no bench_multiturn.jsonl under {cascade_dir}")
    runs: list[dict] = []
    for p_jsonl in jsonls:
        meta = parse_path(p_jsonl) or {
            "tag": "", "cache_mode": "", "size": None, "path": p_jsonl,
        }
        meta["tag"] = tag
        ttft, hit = load_round_data(p_jsonl)
        if not ttft:
            print(f"WARNING: no per-round data in {p_jsonl}, skipping", file=sys.stderr)
            continue
        if max_rounds is not None:
            ttft = ttft[:max_rounds]
            hit  = hit[:max_rounds]
        meta["ttft"] = ttft
        meta["hit"]  = hit
        meta["fill_thresholds"] = load_fill_thresholds(p_jsonl)
        runs.append(meta)
    return runs


def main():
    p = argparse.ArgumentParser(
        description="Plot per-round TTFT / cache-hit-rate for MI355X and B200 "
                    "cascade runs. Takes two cascade root dirs (one per "
                    "platform), auto-discovers L1/L2/L3_file jsonls inside, "
                    "and writes 3 PNGs: MI355X.png, B200.png, MI355X_VS_B200.png.",
    )
    p.add_argument("--MI355X-dir", required=True,
                   help="Path to MI355X cascade root, e.g. "
                        "results/<docker>/<MODEL>-cascade-<tag>/")
    p.add_argument("--B200-dir", required=True,
                   help="Path to B200 cascade root, same layout.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write the 3 PNGs. Default: MI355X-dir.")
    p.add_argument("--model-name", default=None,
                   help="Override the model display name (default: derive "
                        "from cascade dir, e.g. DeepSeek-R1-0528 → DSR1-0528).")
    p.add_argument("--max-rounds", type=int, default=None,
                   help="Truncate every curve to at most N rounds. Useful "
                        "when MI355X (10 rounds) vs B200 (15 rounds) and you "
                        "want a fair window (--max-rounds 10).")
    args = p.parse_args()

    mi_dir   = Path(args.MI355X_dir).expanduser().resolve()
    b2_dir   = Path(args.B200_dir).expanduser().resolve()
    out_dir  = Path(args.out_dir).expanduser().resolve() if args.out_dir else mi_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for d in (mi_dir, b2_dir):
        if not d.is_dir():
            sys.exit(f"ERROR: not a directory: {d}")

    runs_mi = load_runs(mi_dir, "MI355X", args.max_rounds)
    runs_b2 = load_runs(b2_dir, "B200",   args.max_rounds)
    all_runs = runs_mi + runs_b2

    # Shared TTFT Y-axis: round(max + 0.5) + 1 → e.g. 12.6 → 14, 11.7 → 13.
    # Same rule for all three PNGs so they're visually directly comparable.
    max_ttft = max((max(r["ttft"]) for r in all_runs), default=1.0)
    ymax_ttft = round(max_ttft + 0.5) + 1

    model = extract_model_display(mi_dir, args.model_name)
    date_mi = extract_date(mi_dir)
    date_b2 = extract_date(b2_dir)

    # 3 outputs.
    plot_cascade(
        runs_mi,
        out_dir / "MI355X.png",
        title=f"{model} MI355X {date_mi}".strip(),
        ymax_ttft=ymax_ttft,
    )
    plot_cascade(
        runs_b2,
        out_dir / "B200.png",
        title=f"{model} B200 {date_b2}".strip(),
        ymax_ttft=ymax_ttft,
    )
    plot_cascade(
        all_runs,
        out_dir / "MI355X_VS_B200.png",
        title=f"{model} MI355X {date_mi} vs B200 {date_b2}".strip(),
        ymax_ttft=ymax_ttft,
    )


if __name__ == "__main__":
    main()
