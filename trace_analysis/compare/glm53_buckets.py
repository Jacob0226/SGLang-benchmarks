#!/usr/bin/env python3
"""
glm53_buckets.py

GLM-5.3-Flash functional-bucket comparison of two (or more) already-analyzed
runs. Same role as glm52_buckets.py, different rules, and the rules are the
point: GLM-5.2 is a uniform DSA stack, GLM-5.3-Flash is a hybrid, and the
GLM-5.2 rules mis-file most of what is specific to it.

What the GLM-5.2 rules get wrong on this model, measured on the 2026-09-22
MI355X-vs-B200 workbooks:

  * `other` absorbed 25-29% of both sides, because the 34 KDA linear-attention
    layers (chunk_gated_delta_rule / causal_conv1d / gla) and the mHC pre/post
    kernels have no bucket at all.
  * `sparse-MLA attn` read 0.137 ms on MI355X against 33.8 ms on B200, a 250x
    "win" that is pure misfiling: TileLang names its generated kernel
    `main_kernel`, which matches no sparse-MLA pattern, so 78 ms of DSA
    attention sat in `other` while only `set_mla_kv_buffer_kernel_norope`
    landed in the bucket.

So this module classifies on the call site and the layer type first, and on the
kernel name only as a fallback:

  * LeafModule is the owning nn.Module path, which is stable across platforms
    even when every kernel name differs. `main_kernel` under
    `DeepseekV2AttentionMLA > RadixAttention` is sparse MLA; the same name under
    a mHC call site would not be.
  * LayerType is what the analyzer already derived from the graph-off module
    tree, and it maps onto GLM-5.3-Flash's three layer types (see the
    GLM5.3-Flash skill, "Layer 分佈"):
        shared+MLP  -> KDA + dense MLP      (layers 0-2,            3 layers)
        shared+MoE  -> KDA + MoE            (4-6, 8-10, ..., 44,   31 layers)
        full+MoE    -> MLA/DSA + MoE        (3, 7, ..., 43,        11 layers)
        all         -> every layer (mHC, the residual plumbing)
    It is what separates a `RowParallelLinear` that is KDA's output projection
    from one that is MLA's o_proj -- the two are the same module class and, on
    B200, even the same kernel.

The two sides name call sites at different depths: MI355X workbooks carry the
full path (`DeepseekV2AttentionMLA > RadixAttention`) and B200 carries the leaf
(`RadixAttention`), so every call-site rule is a substring test.

step1 workbooks have neither column. They are still accepted, because step1 is
the ground truth for totals, but the projection buckets collapse into
`dense/linear GEMM` there and the table says so.

TWO THINGS THIS TABLE CANNOT TELL YOU, both of which have already produced a
wrong conclusion here. Read them before quoting a cross-platform ratio.

  1. Sigma is not wall time. It adds each stream's kernels separately, so a
     platform that overlaps work is inflated. Measured 2026-09-22, i8k conc4
     decode: MI355X ran 2 streams at 1.00x overlap (Sigma == wall) while B200
     ran 47 streams at 1.20x, so B200's whole column is ~20% high against a
     wall-clock reading. Run diagnostics/forward_overlap.py per side and divide.

  2. `all-reduce/comm` is not transport when a collective spin-waits. B200's
     one-shot push all-reduce spends one launch per forward sitting at the
     barrier: of 182 launches in the conc4 window, 180 were <=20 us totalling
     0.86 ms and TWO were 2471 and 3265 us, i.e. 87% of the bucket, and 97% of
     the bucket's time overlapped no other kernel. That made B200 look 5.4x
     worse than MI355X at all-reduce; per-launch medians are 10.3 us vs 9.3 us,
     so transport is actually equal. MI355X showed no such launch (100% of its
     91 launches under 20 us). Use tools/glm53_allreduce_histogram.sh to check,
     and note that telling rank skew from launch-bound needs every rank's trace,
     not just TP-0.

Also normalize before comparing TOTAL rows: a captured window does not hold the
same number of forwards on both sides. In the same conc4 pair MI355X's window
held 91 all-reduce launches (one forward's worth, 2 per layer x 45 layers) and
B200's held 182 (two forwards).

Usage:
  python glm53_buckets.py --phase decode --out cmp.csv \
      --src MI355X step3_layer_breakdown_MI355X_decode.xlsx \
      --src B200   step3_layer_breakdown_B200_decode.xlsx
"""
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

from openpyxl import load_workbook

# --------------------------------------------------------------------------- #
# Rules. Order = display order AND match order; first match wins.
# Each entry: (bucket, callsite_patterns, kernel_patterns, layertype_pred)
# A rule fires when (no callsite pattern or one matches) AND
#                   (no kernel pattern or one matches) AND
#                   (no layertype predicate or it holds).
# Patterns are regexes tested case-insensitively.
# --------------------------------------------------------------------------- #
# A row's LayerType can be a comma-joined list when one call site appears in
# more than one kind of layer ("full+MoE, shared+MoE"), so the predicates ask
# about EVERY part. A mixed row is not KDA and not DSA, and falls through to the
# call-site rules below it -- inventing a split would be worse than not splitting.
def _parts(lt):
    return [p.strip() for p in lt.split(",") if p.strip()]


KDA_LAYER = lambda lt: bool(_parts(lt)) and all(p.startswith("shared") for p in _parts(lt))
DSA_LAYER = lambda lt: bool(_parts(lt)) and all(p.startswith("full") for p in _parts(lt))

RULES = [
    # TP collectives. Name-based: the call site is whatever linear triggered it.
    # splitKreduce/moeFinalize are GEMM/MoE epilogues, not collectives -- they
    # must not be caught here, hence the explicit list rather than /reduce/.
    ("all-reduce/comm", [], [
        r"all_?reduce", r"reduce_scatter", r"all_?gather", r"nccl", r"rccl",
        r"cross_device_reduce", r"quick.*reduce", r"custom_all",
    ], None),

    # mHC: the 4-stream residual plumbing, 4 kernels per layer on every layer.
    # deep_gemm's tf32_hc_prenorm_gemm is B200's hc_pre; aiter's
    # mhc_pre_gemm_sqrsum / mhc_pre_big_fuse / mhc_post are MI355X's.
    ("mHC (4-stream residual)", [r"mhc\.py", r"communicator_mhc"], [
        r"\bmhc", r"hc_prenorm", r"hc_contract", r"sqrsum", r"hc_pre", r"hc_post",
    ], None),

    # DSA lightning indexer on the k-pool cache: pooled MQA logits, the top-k
    # transform that expands 512 pools into 2051 tokens, and the k-cache writes.
    ("DSA indexer+kpool", [r"IndexerKPool"], [
        r"mqa_logits", r"kpool", r"topk_transform", r"transform_index",
        r"index_k", r"indexer", r"fused_store_index",
    ], None),

    # Sparse MLA attention core. RadixAttention is the owning module on both
    # sides; the word boundary keeps RadixLinearAttention (KDA) out.
    ("sparse-MLA attn", [r"(?<!Linear)RadixAttention"], [
        r"sparse_mla", r"\bfmha", r"flash.*mla", r"aiter\d*::mla",
        r"set_mla_kv_buffer",
    ], None),

    # KDA recurrence core: short conv, chunked gated-delta-rule scan, the gated
    # RMSNorm on its output. 34 of 45 layers.
    ("KDA linear-attn core", [r"RadixLinearAttention", r"FusedRMSNormGated"], [
        r"causal_conv1d", r"chunk_kda", r"chunk_gla", r"chunk_gated_delta",
        r"sigmoid_gating_delta_rule", r"layer_norm_gated", r"fused_recurrent_kda",
        r"kda_gate", r"delta_rule",
    ], None),

    # Projections. Three rules each, most specific first, because the module
    # class alone is ambiguous: KDA, MLA and the MoE shared expert all own a
    # RowParallelLinear, and on B200 they are even the same nvjet kernel.
    #   1. MI355X workbooks carry the full module path, so the owner is named.
    #   2. The shared expert / dense MLP is claimed before the layer-type
    #      fallback, or its MergedColumnParallelLinear (whose LayerType is the
    #      mixed "full+MoE, shared+MoE") would be booked as an MLA projection.
    #   3. B200 carries only the leaf, so fall back to an unambiguous LayerType.
    ("KDA projections", [r"Glm5NextLinearAttention.*Linear"], [], None),
    ("MLA projections", [r"AttentionMLA.*Linear"], [], None),
    ("shared-expert/dense MLP", [r"DeepseekV2MLP"], [], None),
    ("KDA projections", [r"Linear"], [], KDA_LAYER),
    ("MLA projections", [r"Linear"], [], DSA_LAYER),

    # MoE expert GEMMs, split by stage. mfma_moe1/moe2 is MI355X's aiter path,
    # bmm_E2m1_E2m1/bmm_Bfloat16_E2m1 is B200's trtllm NVFP4 path.
    ("MoE up/gate GEMM (moe1)", [], [
        r"moe1", r"moe_?stage1", r"silu_mul", r"silu_and_mul", r"gate_?up",
        r"gemm1_a4w4", r"mxmoe_g1", r"bmm_e2m1_e2m1",
    ], None),
    ("MoE down GEMM (moe2)", [], [
        r"moe2", r"moe_?stage2", r"down_proj_moe", r"gemm2_a4w4", r"mxmoe_g2",
        r"bmm_bfloat16_e2m1",
    ], None),

    # Routing: 288-expert top-8 gate, the sort/align that builds the expert
    # batches, and the finalize/reduction that scatters results back.
    ("MoE routing/sort", [r"FusedMoE", r"TopK", r"MoEGate"], [
        r"moe_sort", r"moe_align", r"moe_sorting", r"grouped_topk", r"fused_moe",
        r"moefinalize", r"moe_reduction", r"fused_mx_quant_moe", r"topk_softmax",
    ], None),

    ("embedding/lm_head/sampling", [r"VocabParallel", r"LogitsProcessor", r"Sampler"], [
        r"embedding", r"lm_head", r"argmax", r"sampling", r"top_k_top_p",
    ], None),

    ("rmsnorm/quant/act", [], [
        r"rms_?norm", r"layernorm", r"layer_norm", r"quant", r"silu", r"gelu",
        r"activation", r"add_rmsnorm",
    ], None),

    ("dense/linear GEMM", [r"Linear"], [
        r"\bgemm", r"cijk_", r"hgemm", r"nvjet", r"cublas", r"cutlass",
        r"\bbmm", r"ck::kernel_gemm",
    ], None),

    ("elementwise/other", [], [
        r"elementwise", r"vectorized", r"copy", r"cat_", r"\bfill", r"index_",
        r"reduce_kernel", r"splitkreduce", r"memset",
    ], None),
]

# dict.fromkeys to dedupe while keeping first-seen order: several buckets have
# more than one rule (a specific call-site match, then a layer-type fallback).
BUCKET_ORDER = list(dict.fromkeys(r[0] for r in RULES)) + ["unclassified"]

# LayerType -> the skill's layer taxonomy, for the second table.
LAYER_TYPE_LABEL = {
    "shared+MLP": "KDA + dense MLP (layers 0-2)",
    "shared+MoE": "KDA + MoE (31 layers)",
    "full+MoE": "MLA/DSA + MoE (11 layers)",
    "all": "every layer (mHC etc.)",
}


def classify(kernel: str, callsite: str, layertype: str) -> str:
    k = (kernel or "").lower()
    c = (callsite or "").lower()
    lt = layertype or ""
    for bucket, cs_pats, k_pats, lt_pred in RULES:
        if lt_pred is not None and not lt_pred(lt):
            continue
        cs_ok = (not cs_pats) or any(re.search(p, c, re.I) for p in cs_pats)
        k_ok = (not k_pats) or any(re.search(p, k, re.I) for p in k_pats)
        if cs_pats and k_pats:
            # Either axis is enough when both are given: a call site we know
            # beats an unknown kernel name, and vice versa.
            if cs_ok or k_ok:
                return bucket
        elif cs_ok and k_ok:
            return bucket
    return "unclassified"


def read_workbook(path: Path):
    """-> (rows, scope) where rows = [(kernel, callsite, layertype, ms, count)]."""
    wb = load_workbook(path, read_only=True)
    ws = wb[wb.sheetnames[0]]
    it = ws.iter_rows(values_only=True)
    header = [str(h) for h in next(it)]
    idx = {h: i for i, h in enumerate(header)}
    step3 = "LeafModule" in idx and "KernelName" in idx
    scope = ("decoder layers only (step3)" if step3
             else "whole forward incl. lm_head/sampling tail (step1)")
    rows = []
    for r in it:
        if r is None or r[0] is None:
            continue
        try:
            if step3:
                kernel = str(r[idx["KernelName"]])
                callsite = str(r[idx["LeafModule"]])
                layertype = str(r[idx["LayerType"]])
                ms = float(r[idx["SumDuration_us"]] or 0) / 1000.0
                cnt = int(float(r[idx["LaunchCount"]] or 0))
            else:
                kernel = str(r[idx["Name"]])
                callsite = ""
                layertype = ""
                ms = float(r[idx["SumDuration_us"]] or 0) / 1000.0
                cnt = int(float(r[idx["Count"]] or 0))
        except (KeyError, TypeError, ValueError):
            continue
        rows.append((kernel, callsite, layertype, ms, cnt))
    return rows, scope, step3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", required=True, choices=["prefill", "decode"])
    ap.add_argument("--src", nargs=2, action="append", metavar=("LABEL", "XLSX"),
                    required=True, help="repeatable: --src MI355X step3_....xlsx")
    ap.add_argument("--out", help="write the bucket table to this CSV")
    ap.add_argument("--top", type=int, default=6,
                    help="kernels to list per bucket per side (default 6)")
    args = ap.parse_args()

    sides, scopes, step3_flags = {}, {}, {}
    for label, path in args.src:
        rows, scope, step3 = read_workbook(Path(path))
        sides[label] = rows
        scopes[label] = (scope, Path(path).name)
        step3_flags[label] = step3
    if len(set(s[0] for s in scopes.values())) > 1:
        raise SystemExit("[ERROR] mixing step1 and step3 scopes across sides is refused")
    if not all(step3_flags.values()):
        print("[WARN] a step1 workbook was passed: it has no call site or layer "
              "type, so KDA/MLA projections fall into dense/linear GEMM and the "
              "per-layer-type table is omitted for that side.")

    labels = list(sides)
    bucket_ms = {l: defaultdict(float) for l in labels}
    bucket_kernels = {l: defaultdict(lambda: defaultdict(lambda: [0.0, 0])) for l in labels}
    layer_ms = {l: defaultdict(float) for l in labels}

    for l, rows in sides.items():
        for kernel, callsite, layertype, ms, cnt in rows:
            b = classify(kernel, callsite, layertype)
            bucket_ms[l][b] += ms
            e = bucket_kernels[l][b][kernel]
            e[0] += ms
            e[1] += cnt
            if layertype:
                # A call site shared by several layer types cannot be split
                # between them from this workbook, so it is reported as mixed
                # rather than divided by a guess.
                parts = _parts(layertype)
                key = parts[0] if len(parts) == 1 else "mixed (call site spans layer types)"
                layer_ms[l][key] += ms

    print(f"\n# GLM-5.3-Flash {args.phase} buckets")
    for l in labels:
        tot = sum(bucket_ms[l].values())
        unc = bucket_ms[l].get("unclassified", 0.0)
        print(f"#   {l:<8} {scopes[l][1]}  Σ={tot:.2f} ms  "
              f"unclassified={unc:.3f} ms ({100*unc/tot if tot else 0:.1f}%)")

    w = max(26, *(len(b) for b in BUCKET_ORDER))
    head = f"{'Bucket':<{w}}" + "".join(f"{l+'_ms':>14}" for l in labels)
    if len(labels) == 2:
        head += f"{'Delta_ms':>12}{'ratio':>9}"
    print(head)
    table = []
    for b in BUCKET_ORDER:
        vals = [bucket_ms[l].get(b, 0.0) for l in labels]
        if not any(vals):
            continue
        line = f"{b:<{w}}" + "".join(f"{v:>14.3f}" for v in vals)
        row = [b] + [f"{v:.3f}" for v in vals]
        if len(labels) == 2:
            d = vals[0] - vals[1]
            r = vals[0] / vals[1] if vals[1] else float("inf")
            line += f"{d:>12.3f}{r:>9.3f}"
            row += [f"{d:.3f}", f"{r:.3f}"]
        print(line)
        table.append(row)
    tots = [sum(bucket_ms[l].values()) for l in labels]
    line = f"{'TOTAL':<{w}}" + "".join(f"{v:>14.3f}" for v in tots)
    trow = ["TOTAL"] + [f"{v:.3f}" for v in tots]
    if len(labels) == 2:
        d = tots[0] - tots[1]
        r = tots[0] / tots[1] if tots[1] else float("inf")
        line += f"{d:>12.3f}{r:>9.3f}"
        trow += [f"{d:.3f}", f"{r:.3f}"]
    print(line)
    table.append(trow)

    if any(bucket_ms[l].get("all-reduce/comm", 0.0) for l in labels):
        print("# NOTE all-reduce/comm mixes transport with barrier wait: a "
              "spin-waiting collective books its wait as GPU time. Check the "
              "per-launch spread (tools/glm53_allreduce_histogram.sh) before "
              "reading this row as communication cost.")
    print("# NOTE Sigma counts streams separately. Divide by the side's overlap "
          "factor (diagnostics/forward_overlap.py) for a wall-clock reading.")

    if all(step3_flags.values()) and any(layer_ms[l] for l in labels):
        # Per side only. The two platforms' workbooks name call sites at
        # different depths, so a row that is "full+MoE, shared+MoE" (mixed) on
        # MI355X can be a plain "all" on B200 -- the split is an artifact of the
        # path depth, not of where the time went. Compare buckets across sides,
        # layer types within one side.
        print(f"\n# time by GLM-5.3-Flash layer type (per side only, NOT comparable across sides)")
        lw = 30
        print(f"{'LayerType':<{lw}}" + "".join(f"{l+'_ms':>14}" for l in labels))
        for lt in ["shared+MLP", "shared+MoE", "full+MoE", "all"]:
            vals = [layer_ms[l].get(lt, 0.0) for l in labels]
            if not any(vals):
                continue
            label = LAYER_TYPE_LABEL.get(lt, lt)
            print(f"{label:<{lw}}" + "".join(f"{v:>14.3f}" for v in vals))
        others = sorted(set().union(*[set(layer_ms[l]) for l in labels])
                        - set(LAYER_TYPE_LABEL))
        for lt in others:
            vals = [layer_ms[l].get(lt, 0.0) for l in labels]
            print(f"{lt:<{lw}}" + "".join(f"{v:>14.3f}" for v in vals))

    print(f"\n# top-{args.top} kernels per bucket")
    for b in BUCKET_ORDER:
        for l in labels:
            ks = bucket_kernels[l].get(b)
            if not ks:
                continue
            for kernel, (ms, cnt) in sorted(ks.items(), key=lambda kv: -kv[1][0])[:args.top]:
                print(f"  {l:<8} {b:<26} {ms:9.3f} ms n={cnt:<5d} {kernel[:70]}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            cw = csv.writer(f)
            cw.writerow([f"# GLM-5.3-Flash {args.phase} buckets"])
            for l in labels:
                tot = sum(bucket_ms[l].values())
                unc = bucket_ms[l].get("unclassified", 0.0)
                cw.writerow([f"# {l}", scopes[l][1], f"sigma_ms={tot:.2f}",
                             f"unclassified_ms={unc:.3f}"])
            hdr = ["Bucket"] + [f"{l}_ms" for l in labels]
            if len(labels) == 2:
                hdr += ["Delta_ms", "ratio"]
            cw.writerow(hdr)
            cw.writerows(table)
            cw.writerow([])
            cw.writerow(["# by layer type"] + labels)
            for lt in ["shared+MLP", "shared+MoE", "full+MoE", "all"]:
                vals = [layer_ms[l].get(lt, 0.0) for l in labels]
                if any(vals):
                    cw.writerow([LAYER_TYPE_LABEL.get(lt, lt)] + [f"{v:.3f}" for v in vals])
            cw.writerow([])
            cw.writerow(["# top kernels per bucket"])
            cw.writerow(["Side", "Bucket", "Kernel", "Sum_ms", "Count"])
            for b in BUCKET_ORDER:
                for l in labels:
                    ks = bucket_kernels[l].get(b)
                    if not ks:
                        continue
                    for kernel, (ms, cnt) in sorted(ks.items(), key=lambda kv: -kv[1][0])[:args.top]:
                        cw.writerow([l, b, kernel, f"{ms:.3f}", cnt])
        print(f"\n[INFO] CSV written: {args.out}")


if __name__ == "__main__":
    main()
