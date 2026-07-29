#!/usr/bin/env python3
"""
compare_glm52_sglang_atom.py

GLM-5.2-specific, apples-to-apples kernel comparison between an SGLang trace and
an ATOM (rocm/atom-dev) trace for ONE forward pass (prefill chunk or decode step).

Why a bespoke tool: analyze_trace.py (SGLang) and analyze_atom_trace.py (ATOM)
emit different Section/Kernel taxonomies, so compare_breakdown.py's row alignment
falls apart. This script instead maps BOTH stacks' kernels into the SAME GLM-5.2
functional buckets (sparse-MLA attn, DSA indexer+topk, MLA/dense GEMM, MoE
up/gate GEMM, MoE down GEMM, all-reduce, rmsnorm/quant, rope/kv-cache, other),
so the two columns are directly comparable.

It is intentionally GLM-5.2-only: bucket regexes and the 16384-token prefill /
bs=64 decode heuristics assume the GLM-5.2 DSA+MLA+MoE architecture.

Segmentation is stream-aware (kernels + wrapper annotations restricted to the
dominant GPU stream), isolating ONE forward between consecutive wrapper markers:
  - SGLang:  gpu_user_annotation  "step[EXTEND ...]" / "step[DECODE ...]"
  - ATOM:    gpu_user_annotation  "prefill[...]"     / "decode[...]"

For SGLang prefill, triton kernels can be JIT/autotune-inflated on their first
occurrence; when several qualifying forwards exist we take the FASTEST (min
Sigma) as the steady-state estimate and warn if only one is available.

Usage:
  python compare_glm52_sglang_atom.py --phase prefill \
      --sglang TRACE_SGLANG_EXTEND.json.gz --atom TRACE_ATOM.json.gz \
      --out cmp_prefill.csv
  python compare_glm52_sglang_atom.py --phase decode \
      --sglang TRACE_SGLANG_DECODE.json.gz --atom TRACE_ATOM.json.gz \
      --out cmp_decode.csv
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


# --------------------------------------------------------------------------- #
# GLM-5.2 functional buckets. Order = display order. First matching regex wins.
# Patterns are lowercased substring/regex tests applied to the kernel name.
# --------------------------------------------------------------------------- #
import re

BUCKET_RULES = [
    # all-reduce / TP communication (quickreduce INT4, rccl, aiter allgather, etc.)
    ("all-reduce/comm", [
        r"reduce_scatter", r"all_?reduce", r"allgather", r"all_gather",
        r"nccl", r"rccl", r"cross_device", r"quick.*reduce", r"custom_all",
    ]),
    # DSA indexer + top-k selection (paged MQA logits, hadamard, fp8 index quant).
    # NOTE: use SPECIFIC topk names (topk_transform / radix_topk) so we don't
    # swallow MoE's expert top-k kernels (grouped_topk, moe_reduction_..._topk9).
    ("DSA indexer+topk", [
        r"paged_mqa_logits", r"mqa_logits", r"deepgemm_fp8_paged", r"hadamard",
        r"topk_transform", r"radix_topk", r"indexer", r"transform_index",
        r"convert_req_index", r"fused_qk_rmsnorm_group_quant",
    ]),
    # sparse MLA attention core (triton sparse-mla, aiter mla, tilelang mla)
    ("sparse-MLA attn", [
        r"sparse_mla", r"_mla_", r"\bmla\b", r"mla_decode", r"mla_fwd",
        r"flash.*mla", r"aiter\d*::mla", r"mla_a8w8",
    ]),
    # MoE experts up/gate projection GEMM (moe1 / silu-mul fused)
    ("MoE up/gate GEMM (moe1)", [
        r"moe1", r"moe_?stage1", r"gate_?up", r"silu_mul", r"silu_and_mul",
    ]),
    # MoE experts down projection GEMM (moe2)
    ("MoE down GEMM (moe2)", [
        r"moe2", r"moe_?stage2", r"down_proj_moe",
    ]),
    # other MoE plumbing: routing/sorting, expert top-k, output reduction, gather
    ("MoE routing/other", [
        r"moe_sort", r"moe_align", r"moe_sorting", r"fused_moe", r"fused_mx_quant_moe",
        r"grouped_?gemm", r"group_?gemm", r"grouped_topk", r"moe_reduction",
        r"append_shared_expert", r"\bmoe\b", r"expert",
    ]),
    # dense / linear GEMM: attention q/kv/o projections + dense MLP (non-MoE)
    ("dense/linear GEMM", [
        r"cijk", r"hgemm", r"\bgemm\b", r"cshuffle", r"gemm_xdl", r"tensile",
        r"matmul", r"wgrad", r"a8w8", r"f8_.*gemm", r"gemm_a8", r"gemm_afp4",
    ]),
    # rmsnorm / quant / activation (non-indexer)
    ("rmsnorm/quant/act", [
        r"rmsnorm", r"\bnorm\b", r"layernorm", r"quant", r"dequant", r"silu",
        r"gelu", r"activation", r"scaled_",
    ]),
    # rope + kv-cache writes
    ("rope/kv-cache", [
        r"rope", r"kv_?cache", r"cache_flat", r"set_kv", r"store_kv",
        r"reshape_and_cache", r"append_kv",
    ]),
    # embedding
    ("embedding", [r"embed"]),
    # elementwise plumbing
    ("elementwise/other", [
        r"elementwise", r"\bcopy\b", r"copybuffer", r"\bcast\b", r"\bfill\b",
        r"memset", r"\badd\b", r"vectorized_elementwise", r"index_", r"permute",
        r"transpose", r"\bcat\b", r"concat", r"masked_embedding",
    ]),
]
COMPILED = [(name, [re.compile(p) for p in pats]) for name, pats in BUCKET_RULES]

EXCLUDE_TAIL = False  # set from --exclude-tail


def classify(kernel_name: str) -> str:
    n = kernel_name.lower()
    for name, pats in COMPILED:
        for p in pats:
            if p.search(n):
                return name
    return "other"


# --------------------------------------------------------------------------- #
def load_trace(path: str):
    p = Path(path)
    if not p.exists():
        sys.exit(f"[ERROR] file not found: {path}")
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        t = json.load(f)
    return t if isinstance(t, list) else t.get("traceEvents", [])


def _x(e):
    return isinstance(e, dict) and e.get("ph") == "X"


def dominant_stream(kernels):
    return Counter((k["pid"], k["tid"]) for k in kernels).most_common(1)[0][0]


def collect(evs):
    kernels = [e for e in evs if _x(e) and str(e.get("cat", "")).lower() == "kernel"]
    if not kernels:
        sys.exit("[ERROR] no kernel events in trace")
    dom = dominant_stream(kernels)
    kernels = [{"name": k.get("name", "?"), "ts": float(k["ts"]),
                "dur": float(k.get("dur", 0.0))}
               for k in kernels if (k["pid"], k["tid"]) == dom]
    kernels.sort(key=lambda k: k["ts"])
    ann = [{"name": a.get("name", "?"), "ts": float(a["ts"]), "dur": float(a.get("dur", 0.0))}
           for a in evs if _x(a) and a.get("cat") == "gpu_user_annotation"
           and (a["pid"], a["tid"]) == dom]
    ann.sort(key=lambda a: a["ts"])
    return kernels, ann, dom


def wrapper_markers(ann, phase, stack):
    """Return the forward-boundary wrapper annotations (all forwards), sorted."""
    if stack == "sglang":
        prefixes = ("step[",)
    else:  # atom
        prefixes = ("prefill[", "decode[")
    return [a for a in ann if any(a["name"].startswith(p) for p in prefixes)]


def _is_target(name, phase, stack):
    n = name
    if phase == "prefill":
        if stack == "sglang":
            return n.startswith("step[EXTEND") and "toks=16" in n
        return n.startswith("prefill[") and "tok=16384" in n
    else:  # decode
        if stack == "sglang":
            return n.startswith("step[DECODE") or (n.startswith("step[") and "DECODE" in n)
        return n.startswith("decode[") and "bs=64" in n


def segment_and_sum(kernels, ann, phase, stack):
    """Isolate one representative forward and return (label, span_us, {bucket: us},
    total_us, nkernels, per_kernel[(bucket,name,us,count)])."""
    marks = wrapper_markers(ann, phase, stack)
    if not marks:
        sys.exit(f"[ERROR] no {stack} {phase} wrapper markers found on dominant stream")
    mts = [m["ts"] for m in marks]
    kts = [k["ts"] for k in kernels]

    def window_sum(s0, s1):
        lo = bisect.bisect_left(kts, s0)
        hi = bisect.bisect_left(kts, s1)
        return kernels[lo:hi]

    # candidate target forwards
    cands = []
    for m in marks:
        if not _is_target(m["name"], phase, stack):
            continue
        s0 = m["ts"]
        j = bisect.bisect_right(mts, s0)
        s1 = mts[j] if j < len(mts) else (kts[-1] + kernels[-1]["dur"] + 1)
        win = window_sum(s0, s1)
        tot = sum(k["dur"] for k in win)
        cands.append((tot, m["name"], s1 - s0, win))
    if not cands:
        sys.exit(f"[ERROR] no target {phase} forward (full-chunk/bs=64) found for {stack}")

    # SGLang triton prefill may be JIT-inflated on first occurrence -> take the
    # fastest (min Sigma) as steady-state; decode/ATOM -> median is fine.
    if stack == "sglang" and phase == "prefill" and len(cands) > 1:
        cands.sort(key=lambda c: c[0])
        chosen = cands[0]
        note = f"min-of-{len(cands)} (JIT-guard)"
    else:
        cands.sort(key=lambda c: c[0])
        chosen = cands[len(cands) // 2]
        note = f"median-of-{len(cands)}" if len(cands) > 1 else "only-1"

    tot, label, span, win = chosen
    if EXCLUDE_TAIL and win:
        # Drop end-of-step tail (lm_head / sampling / logits allgather / broadcast
        # / next-token embedding / metadata) that runs after the last decoder layer,
        # so scope matches SGLang's per-layer breakdown. Boundary = end of the last
        # per-layer TP all-reduce (reduce_scatter / cross_device / quickreduce twoshot).
        _ar = re.compile(r"reduce_scatter|cross_device|allreduce_prototype|quickreduce")
        ends = [k["ts"] + k["dur"] for k in win if _ar.search(k["name"].lower())]
        if ends:
            cut = max(ends)
            win = [k for k in win if k["ts"] < cut + 1]
    buckets = defaultdict(float)
    perk = defaultdict(lambda: [0.0, 0, float("inf")])  # sum, count, first_ts
    for k in win:
        b = classify(k["name"])
        buckets[b] += k["dur"]
        e = perk[(b, k["name"])]
        e[0] += k["dur"]; e[1] += 1; e[2] = min(e[2], k["ts"])
    perk_rows = [(b, nm, us, c) for (b, nm), (us, c, _t) in perk.items()]
    # call-order: aggregated kernels sorted by first-seen timestamp
    callorder = sorted([(t, b, nm, us, c) for (b, nm), (us, c, t) in perk.items()],
                       key=lambda x: x[0])
    return label, span, dict(buckets), tot, len(win), perk_rows, note, callorder


BUCKET_ORDER = [name for name, _ in BUCKET_RULES] + ["other"]


def write_callorder_xlsx(path, phase, labels, meta, cats, sb, ab, sco, aco):
    """Single-forward, tail-excluded call-order side-by-side (SGLang | ATOM),
    each kernel tagged by functional category, + a category summary at the bottom.
    Both sides come from the SAME segment_and_sum single-forward window, so counts
    and durations are on the identical scale (no multi-forward aggregation)."""
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    from openpyxl.utils import get_column_letter
    LA, LB = labels
    wb = Workbook(); ws = wb.active; ws.title = f"{phase}_callorder"
    bold = Font(name="Arial", size=10, bold=True)
    reg = Font(name="Arial", size=9)
    mono = Font(name="Consolas", size=9)
    white = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    src_fill = PatternFill("solid", fgColor="305496")
    hdr_fill = PatternFill("solid", fgColor="8EAADB")
    tot_fill = PatternFill("solid", fgColor="FCE4D6")
    center = Alignment(horizontal="center")
    COLS = ["Category", "KernelName", "Σ_ms", "Cnt"]
    ncol = len(COLS); gap = 1
    ws.cell(1, 1, f"GLM-5.2 {phase} — call order (single forward, tail-excluded), "
                  f"tagged by category. {LA} | {LB}, NOT aligned.").font = bold
    ws.cell(2, 1, meta[0]).font = reg
    ws.cell(3, 1, meta[1]).font = reg
    r0 = 5
    for si, lab in enumerate((LA, LB)):
        c0 = 1 + si * (ncol + gap)
        for j in range(ncol):   # fill the band cells (no merge -> avoids Excel repair prompt)
            cell = ws.cell(r0, c0 + j, lab if j == 0 else "")
            cell.font = white; cell.fill = src_fill; cell.alignment = center
    hr = r0 + 1
    for si in range(2):
        c0 = 1 + si * (ncol + gap)
        for j, h in enumerate(COLS):
            cell = ws.cell(hr, c0 + j, h); cell.font = bold; cell.fill = hdr_fill
    ws.freeze_panes = f"A{hr+1}"
    maxlen = max(len(sco), len(aco))
    for i in range(maxlen):
        rr = hr + 1 + i
        for si, co in enumerate((sco, aco)):
            if i >= len(co):
                continue
            _t, b, nm, us, c = co[i]
            c0 = 1 + si * (ncol + gap)
            ws.cell(rr, c0 + 0, b).font = reg
            ws.cell(rr, c0 + 1, nm[:70]).font = mono
            ws.cell(rr, c0 + 2, round(us / 1000.0, 3)).font = reg
            ws.cell(rr, c0 + 3, c).font = reg
    # category summary
    sr = hr + 1 + maxlen + 2
    ws.cell(sr, 1, "=== Category summary (Σ ms per category, same single forward) ===").font = bold
    sr += 1
    for j, h in enumerate(["Category", f"{LA}_ms", f"{LB}_ms",
                           f"Delta_{LA}_minus_{LB}_ms", f"{LA}_over_{LB}"]):
        cell = ws.cell(sr, 1 + j, h); cell.font = white; cell.fill = src_fill
    sr += 1
    for c in cats:
        a = sb.get(c, 0.0) / 1000.0; b = ab.get(c, 0.0) / 1000.0
        ws.cell(sr, 1, c).font = bold
        ws.cell(sr, 2, round(a, 3)).font = reg
        ws.cell(sr, 3, round(b, 3)).font = reg
        ws.cell(sr, 4, round(a - b, 3)).font = reg
        ws.cell(sr, 5, round(a / b, 3) if b > 1e-9 else "").font = reg
        sr += 1
    stot = sum(sb.values()) / 1000.0; atot = sum(ab.values()) / 1000.0
    ws.cell(sr, 1, "TOTAL").font = bold
    ws.cell(sr, 2, round(stot, 3)).font = bold
    ws.cell(sr, 3, round(atot, 3)).font = bold
    ws.cell(sr, 4, round(stot - atot, 3)).font = bold
    ws.cell(sr, 5, round(stot / atot, 3) if atot else "").font = bold
    for cc in range(1, 6):
        ws.cell(sr, cc).fill = tot_fill
    widths = [22, 60, 9, 6, 3, 22, 60, 9, 6]
    for c, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(c)].width = w
    wb.save(path)
    print(f"[INFO] call-order + summary xlsx written: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["prefill", "decode"], required=True)
    ap.add_argument("--sglang", required=True, metavar="TRACE",
                    help="SGLang graph-ON trace: EXTEND for prefill, DECODE for decode")
    ap.add_argument("--atom", required=True, metavar="TRACE",
                    help="ATOM graph-ON trace (combined)")
    ap.add_argument("--out", required=True, metavar="CSV")
    ap.add_argument("--xlsx", metavar="XLSX", default=None,
                    help="also write the call-order + bucket-summary workbook "
                         "(kernels tagged by bucket; no module labels — for those "
                         "use callorder_sidebyside.py on step3 files)")
    ap.add_argument("--labels", nargs=2, default=["SGLANG", "ATOM"])
    ap.add_argument("--top", type=int, default=6,
                    help="top-N kernels per bucket to include in the detail CSV")
    ap.add_argument("--exclude-tail", dest="exclude_tail", action="store_true",
                    help="drop end-of-step tail (lm_head/sampling/embedding/broadcast/"
                         "metadata) after the last layer's TP all-reduce, for a fair "
                         "per-layer scope on both sides")
    args = ap.parse_args()
    global EXCLUDE_TAIL
    EXCLUDE_TAIL = args.exclude_tail

    sk, sa, sd = collect(load_trace(args.sglang))
    ak, aa, ad = collect(load_trace(args.atom))
    slabel, sspan, sb, stot, snk, sperk, snote, sco = segment_and_sum(sk, sa, args.phase, "sglang")
    alabel, aspan, ab, atot, ank, aperk, anote, aco = segment_and_sum(ak, aa, args.phase, "atom")

    LA, LB = args.labels
    print(f"\n=== GLM-5.2 {args.phase} comparison ({LA} vs {LB}) ===")
    print(f"  {LA}: {slabel}")
    print(f"        forward={snote}  span={sspan/1000:.2f}ms  Sigma_kernel={stot/1000:.2f}ms  nkernels={snk}")
    print(f"  {LB}: {alabel}")
    print(f"        forward={anote}  span={aspan/1000:.2f}ms  Sigma_kernel={atot/1000:.2f}ms  nkernels={ank}")
    cats = sorted(set(sb) | set(ab), key=lambda c: BUCKET_ORDER.index(c) if c in BUCKET_ORDER else 99)
    print(f"\n  {'Bucket':<24}{LA+'_ms':>12}{LB+'_ms':>12}{'Δ(A-B)ms':>12}{'A/B':>8}")
    print(f"  {'-'*24}{'-'*12}{'-'*12}{'-'*12}{'-'*8}")
    for c in cats:
        a = sb.get(c, 0.0) / 1000
        b = ab.get(c, 0.0) / 1000
        ratio = (a / b) if b > 1e-9 else float("inf")
        print(f"  {c:<24}{a:>12.2f}{b:>12.2f}{a-b:>12.2f}{ratio:>8.2f}")
    print(f"  {'-'*24}{'-'*12}{'-'*12}{'-'*12}{'-'*8}")
    print(f"  {'TOTAL Sigma_kernel':<24}{stot/1000:>12.2f}{atot/1000:>12.2f}{(stot-atot)/1000:>12.2f}"
          f"{(stot/atot if atot else float('inf')):>8.2f}")

    # --- write CSV -------------------------------------------------------- #
    import csv
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# GLM-5.2 {args.phase} comparison", LA, "vs", LB])
        w.writerow([f"# {LA} forward", slabel, snote, f"span_ms={sspan/1000:.2f}",
                    f"sigma_ms={stot/1000:.2f}", f"nkernels={snk}"])
        w.writerow([f"# {LB} forward", alabel, anote, f"span_ms={aspan/1000:.2f}",
                    f"sigma_ms={atot/1000:.2f}", f"nkernels={ank}"])
        w.writerow([])
        w.writerow(["Bucket", f"{LA}_ms", f"{LB}_ms",
                    f"Delta_{LA}_minus_{LB}_ms", f"{LA}_over_{LB}"])
        for c in cats:
            a = sb.get(c, 0.0) / 1000
            b = ab.get(c, 0.0) / 1000
            ratio = round(a / b, 3) if b > 1e-9 else ""
            w.writerow([c, round(a, 3), round(b, 3), round(a - b, 3), ratio])
        w.writerow(["TOTAL", round(stot/1000, 3), round(atot/1000, 3),
                    round((stot-atot)/1000, 3),
                    round(stot/atot, 3) if atot else ""])
        # per-bucket top kernels for both sides
        w.writerow([])
        w.writerow([f"# top-{args.top} kernels per bucket"])
        w.writerow(["Side", "Bucket", "Kernel", "Sum_ms", "Count"])
        for side, perk in ((LA, sperk), (LB, aperk)):
            by_bucket = defaultdict(list)
            for b, nm, us, c in perk:
                by_bucket[b].append((us, nm, c))
            for c in cats:
                for us, nm, cnt in sorted(by_bucket.get(c, []), reverse=True)[:args.top]:
                    w.writerow([side, c, nm[:80], round(us/1000, 3), cnt])
    print(f"\n[INFO] CSV written: {out}")

    if args.xlsx:
        meta = (f"{LA}: {slabel}  [{snote}, span={sspan/1000:.1f}ms, Σ={stot/1000:.1f}ms, {snk} kernels]",
                f"{LB}: {alabel}  [{anote}, span={aspan/1000:.1f}ms, Σ={atot/1000:.1f}ms, {ank} kernels]")
        write_callorder_xlsx(args.xlsx, args.phase, (LA, LB), meta, cats, sb, ab, sco, aco)


if __name__ == "__main__":
    main()
