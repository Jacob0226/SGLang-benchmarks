#!/usr/bin/env python3
"""
callorder_sidebyside.py

Dump several step3_layer_breakdown.xlsx files side by side, each kept in its OWN
call order (NO cross-source alignment). Every kernel row is tagged with the
class/module that called it (Section > LeafModule) and its call site, so the
reader can classify/match the two stacks manually.

Usage:
  python callorder_sidebyside.py --out combined.xlsx \
      --src SGLANG_new step3_..._SGLANG_new.xlsx \
      --src SGLANG_old step3_..._SGLANG_old.xlsx \
      --src ATOM       step3_..._ATOM.xlsx
"""
import argparse
from collections import defaultdict
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter


def read_layer_types(path):
    """The step3 workbook's LayerTypes sheet, which decodes the LayerType letters
    into layer types (GLM-5.2: full-indexer + MLP / full-indexer + MoE /
    shared-indexer + MoE). Absent on sources that do not group by layer (ATOM)."""
    wb = load_workbook(path, data_only=True)
    if "LayerTypes" not in wb.sheetnames:
        return []
    out = []
    for r in wb["LayerTypes"].iter_rows(min_row=2, values_only=True):
        if not r or not r[0] or str(r[0]) == "NOTE":
            continue
        out.append((r[0], r[1], r[2], r[3]))
    return out


def _pick(row, h, *names):
    """First of `names` present in the header (later ones are pre-rename aliases)."""
    for n in names:
        if n in h:
            return row[h[n]]
    return ""


def read_step3(path):
    ws = load_workbook(path, data_only=True).active
    rows = list(ws.iter_rows(values_only=True))
    h = {n: i for i, n in enumerate(rows[0]) if n}
    out = []
    for r in rows[1:]:
        kn = r[h.get("KernelName")] if "KernelName" in h else None
        if kn is None:
            continue
        out.append({
            "LayerType": r[h.get("LayerType", -1)] if "LayerType" in h else "",
            "Section": r[h.get("Section")] if "Section" in h else "",
            "Leaf": r[h.get("LeafModule")] if "LeafModule" in h else "",
            "Kernel": kn,
            "Avg_us": r[h["AvgDuration_us"]] if "AvgDuration_us" in h else "",
            "Sum_ms": (r[h["SumDuration_us"]] or 0) / 1000.0 if "SumDuration_us" in h else 0,
            "Count": _pick(r, h, "LaunchCount", "Count"),
            "TrSum_ms": _pick(r, h, "KernelSum_ms_fwd", "TraceSum_ms_fwd"),
            "TrCount": _pick(r, h, "KernelCount_fwd", "TraceCount_fwd"),
            "CallSite": r[h.get("CallSite")] if "CallSite" in h else "",
        })
    return out


# Avg_us is the cost of ONE launch. LaunchCnt is how many launches THIS row's call
# site made in the forward — on the SGLang side that is one per layer, so it equals
# the number of layers of the types in LayerType — and Σ_ms = Avg_us x LaunchCnt.
# KernelΣ_ms/KernelCnt are the kernel NAME's totals in the same forward regardless of
# call site, repeated on every row sharing the name (so do NOT sum them): Σ LaunchCnt
# of the same-name rows below KernelCnt means the kernel also runs outside the layers.
COLS = ["LayerType", "Section", "LeafModule (caller)", "KernelName", "Avg_us",
        "Σ_ms", "LaunchCnt", "KernelΣ_ms", "KernelCnt", "CallSite"]

import re
# functional category rules (first match wins); GLM-5.2 specific
_CAT_RULES = [
    ("rccl/all-reduce", [r"reduce_scatter", r"all_?reduce", r"allreduce", r"allgather",
                         r"nccl", r"rccl", r"cross_device", r"quickreduce", r"custom_all"]),
    ("MoE GEMM", [r"moe1", r"moe2", r"fused_moe", r"mfma_moe", r"grouped_?gemm",
                  r"moe_reduction", r"moe_sort", r"fused_mx_quant_moe", r"\bmoe\b", r"expert"]),
    ("MoE routing/topk", [r"grouped_topk", r"moe_align", r"opus_moe_sorting"]),
    ("Indexer/DSA-topk", [r"mqa_logits", r"deepgemm_fp8_paged", r"hadamard", r"topk_transform",
                          r"radix_topk", r"indexer", r"convert_req_index", r"fused_qk_rmsnorm_group"]),
    ("MLA attention", [r"sparse_mla", r"_mla_", r"\bmla\b", r"mla_decode", r"mla_fwd",
                       r"aiter\d*::mla", r"mla_reduce", r"flash.*mla"]),
    ("Dense/linear GEMM", [r"cijk", r"hgemm", r"\bgemm\b", r"cshuffle", r"gemm_xdl",
                           r"tensile", r"matmul", r"batched_gemm", r"a8w8", r"kernel_gemm"]),
    ("rope/kv-cache", [r"rope", r"kv_?cache", r"concat_and_cache", r"reshape_and_cache"]),
    ("rmsnorm/quant/act", [r"rmsnorm", r"\bnorm\b", r"layernorm", r"quant", r"dequant",
                           r"silu", r"gelu", r"act_and_mul"]),
    ("embedding", [r"embed"]),
    ("elementwise/copy", [r"elementwise", r"\bcopy\b", r"copybuffer", r"\bcast\b", r"\bfill\b",
                          r"memset", r"as_strided", r"index", r"gather", r"scatter",
                          r"transpose", r"permute", r"\bcat\b"]),
]
_CAT_COMPILED = [(n, [re.compile(p) for p in ps]) for n, ps in _CAT_RULES]
_CAT_ORDER = [n for n, _ in _CAT_RULES] + ["other"]


def categorize(kn):
    s = str(kn).lower()
    for name, pats in _CAT_COMPILED:
        for p in pats:
            if p.search(s):
                return name
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", nargs=2, action="append", metavar=("LABEL", "XLSX"),
                    required=True, help="repeatable: LABEL path/to/step3.xlsx")
    ap.add_argument("--title", default="")
    ap.add_argument("--summary-csv", dest="summary_csv", default=None,
                    help="sglang_vs_atom_glm52 3-way CSV to use for the (clean, per-forward) "
                         "category summary instead of recomputing from step3")
    args = ap.parse_args()
    summary_csv = args.summary_csv

    sources = [(lab, read_step3(p)) for lab, p in args.src]

    wb = Workbook(); ws = wb.active; ws.title = "call_order_side_by_side"
    bold = Font(name="Arial", size=10, bold=True)
    white = Font(name="Arial", size=10, bold=True, color="FFFFFF")
    reg = Font(name="Arial", size=9)
    mono = Font(name="Consolas", size=9)
    src_fill = PatternFill("solid", fgColor="305496")
    hdr_fill = PatternFill("solid", fgColor="8EAADB")
    tot_fill = PatternFill("solid", fgColor="FCE4D6")
    center = Alignment(horizontal="center")

    ncol = len(COLS)
    gap = 1
    r0 = 1
    if args.title:
        ws.cell(1, 1, args.title).font = bold
        r0 = 2

    for lab, path in args.src:
        legend = read_layer_types(path)
        if not legend:
            continue
        kinds = sorted({name for _l, name, _c, _i in legend})
        text = (f"{lab} — {len(kinds)} layer types: {' / '.join(kinds)}.  "
                f"LayerType column: "
                + " · ".join(f"{l}={name} ({cnt} layer{'' if cnt == 1 else 's'},"
                             f" idx {idx})" for l, name, cnt, idx in legend)
                + "   (see the LayerTypes sheet of the step3 workbook)")
        ws.cell(r0, 1, text).font = reg
        r0 += 1

    # source label band
    for si, (lab, _) in enumerate(sources):
        c0 = 1 + si * (ncol + gap)
        cell = ws.cell(r0, c0, lab); cell.font = white; cell.fill = src_fill
        ws.merge_cells(start_row=r0, start_column=c0, end_row=r0, end_column=c0 + ncol - 1)
        cell.alignment = center
    # header
    hr = r0 + 1
    for si, _ in enumerate(sources):
        c0 = 1 + si * (ncol + gap)
        for j, cname in enumerate(COLS):
            cell = ws.cell(hr, c0 + j, cname); cell.font = bold; cell.fill = hdr_fill
    ws.freeze_panes = f"A{hr+1}"

    maxlen = max(len(rows) for _, rows in sources)
    for i in range(maxlen):
        rr = hr + 1 + i
        for si, (_, rows) in enumerate(sources):
            if i >= len(rows):
                continue
            d = rows[i]; c0 = 1 + si * (ncol + gap)
            vals = [d["LayerType"], d["Section"], d["Leaf"], d["Kernel"],
                    d["Avg_us"], round(d["Sum_ms"], 3), d["Count"],
                    d["TrSum_ms"], d["TrCount"], d["CallSite"]]
            for j, v in enumerate(vals):
                cell = ws.cell(rr, c0 + j, v)
                cell.font = mono if j in (3, 9) else reg

    # --- category summary table at the bottom ---
    sr = hr + 1 + maxlen + 2   # leave a gap
    if summary_csv:
        # Use the clean per-forward category numbers from a sglang_vs_atom_glm52 3-way CSV
        # (each side isolates ONE forward -> directly comparable, unlike step3).
        import csv as _csv
        with open(summary_csv) as f:
            crows = [r for r in _csv.reader(f) if r]
        ws.cell(sr, 1, "Category summary — per ONE forward, Σ ms "
                       "(from sglang_vs_atom_glm52; directly comparable)").font = bold
        sr += 1
        for r in crows:
            if r[0].startswith("#"):
                ws.cell(sr, 1, ", ".join(r)).font = reg; sr += 1; continue
            is_hdr = (r[0] == "Bucket")
            is_tot = (r[0] == "TOTAL")
            for j, v in enumerate(r):
                try:
                    v = float(v)
                except (ValueError, TypeError):
                    pass
                cell = ws.cell(sr, 1 + j, v)
                cell.font = white if is_hdr else (bold if is_tot else reg)
                if is_hdr:
                    cell.fill = src_fill
                elif is_tot:
                    cell.fill = tot_fill
            sr += 1
    else:
        cat_sum = [defaultdict(float) for _ in sources]
        for si, (_, rows) in enumerate(sources):
            for d in rows:
                cat_sum[si][categorize(d["Kernel"])] += d["Sum_ms"]
        cats = [c for c in _CAT_ORDER if any(c in cs for cs in cat_sum)]
        ws.cell(sr, 1, "Category summary (Σ ms per category; NOTE: step3 scale)").font = bold
        sr += 1
        hdr2 = ["Category"] + [lab for lab, _ in sources]
        for j, h in enumerate(hdr2):
            cell = ws.cell(sr, 1 + j, h); cell.font = white; cell.fill = src_fill
        sr += 1
        totals = [0.0] * len(sources)
        for c in cats:
            ws.cell(sr, 1, c).font = bold
            for si in range(len(sources)):
                v = cat_sum[si].get(c, 0.0)
                totals[si] += v
                ws.cell(sr, 2 + si, round(v, 3)).font = reg
            sr += 1
        ws.cell(sr, 1, "TOTAL").font = bold
        for si in range(len(sources)):
            cell = ws.cell(sr, 2 + si, round(totals[si], 3)); cell.font = bold; cell.fill = tot_fill

    # widths
    widths = [16, 22, 24, 46, 9, 8, 5, 8, 6, 40]
    for si in range(len(sources)):
        c0 = 1 + si * (ncol + gap)
        for j, w in enumerate(widths):
            ws.column_dimensions[get_column_letter(c0 + j)].width = w
        if si < len(sources) - 1:
            ws.column_dimensions[get_column_letter(c0 + ncol)].width = 2
    wb.save(args.out)
    print(f"[INFO] written {args.out}  ({', '.join(f'{lab}:{len(rows)}' for lab,rows in sources)})")


if __name__ == "__main__":
    main()
