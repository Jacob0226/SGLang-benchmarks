#!/usr/bin/env python3
"""
GLM-5.2-SPECIALIZED full/shared decode-layer side-by-side (ATOM | SGLang).

Why specialized: neither side has a usable per-layer module tree in its decode
no-graph trace. SGLang's has only `step[DECODE bs=N]` + torch.compile FX-graph call
markers, and ATOM torch.compiles the whole model, so nothing marks where one decoder
layer ends. (analyze/sglang_trace.py's step3 does separate full+MoE from shared+MoE on the
SGLang side, since it groups layers by the kernels they run — but that gives you a
per-type average, not two concrete neighbouring layers, and it gives you nothing at
all for ATOM.)

This tool instead segments a decode step POSITIONALLY using GLM-5.2 structure:
  - `fused_qk_rmsnorm` fires exactly once per layer  -> layer anchor.
  - the `allreduce_fusion` right before it is the layer's prepare_attn boundary.
  - layer i = [allreduce_before(rmsnorm[i]), allreduce_before(rmsnorm[i+1])).
  - MoE layer  = layer index >= first_k_dense_replace (3).
  - full layer = its kernels include the indexer (kn_entry_2c / paged_mqa_logits).
Section labels are assigned positionally (state flips to MoE at the first MoE GEMM).
Per-kernel TIMES come from the graph-ON decode trace (avg per kernel name), same as
the ATOM side; structure/order comes from the no-graph trace.

Usage:
  python glm52_full_layer_sidebyside.py \
    --atom-time A.graph.json.gz --atom-struct A.nograph.json.gz \
    --sglang-time S.graph.json.gz --sglang-struct S.nograph.json.gz \
    --layer-kind full --out out.xlsx
"""
import argparse, gzip, json, bisect, re, functools, subprocess
from collections import defaultdict, Counter
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment

FIRST_K_DENSE = 3

# TP all-reduce, i.e. the layer boundary on the residual stream. The backend has
# changed name more than once (aiter allreduce_fusion / cross_device_reduce,
# quickreduce, reduce_scatter+store), so match them all; "reduce" alone would also
# catch _sparse_mla_decode_reduce_kernel, which is inside attention.
AR_KERNELS = ("allreduce", "all_reduce", "cross_device_reduce", "quickreduce",
              "reduce_scatter")


# --------------------------------------------------------------- trace loading
def load(path):
    with gzip.open(path, "rt") as f:
        t = json.load(f)
    return t if isinstance(t, list) else t.get("traceEvents", [])

def _k(evs):
    return [{"name": e.get("name","?"), "ts": float(e["ts"]), "dur": float(e.get("dur",0)),
             "pid": e.get("pid"), "tid": e.get("tid")}
            for e in evs if isinstance(e,dict) and e.get("ph")=="X"
            and str(e.get("cat","")).lower()=="kernel"]
def _g(evs):
    return [{"name": e.get("name","?"), "ts": float(e["ts"]), "dur": float(e.get("dur",0)),
             "pid": e.get("pid"), "tid": e.get("tid")}
            for e in evs if isinstance(e,dict) and e.get("ph")=="X"
            and e.get("cat")=="gpu_user_annotation"]
def _dom(ks): return Counter((k["pid"],k["tid"]) for k in ks).most_common(1)[0][0]



# ------------------------------------------------- kernel-name shortening
@functools.lru_cache(maxsize=8192)
def _demangle(name):
    # Traces store raw C++ Itanium-mangled symbols for aiter template kernels
    # (e.g. _ZN5aiter30allreduce_fusion_kernel_1stageI...). Try a real demangler
    # first (llvm-cxxfilt handles the bf16 `DF16b` mangling; binutils c++filt is
    # often too old and returns it unchanged). If none works, fall back to a
    # version-independent heuristic that pulls the clean function name out of the
    # Itanium nested-name encoding (`_ZN <len><ns> <len><fn> ...`).
    if not name.startswith("_Z"):
        return name
    for tool in ("llvm-cxxfilt", "c++filt"):
        try:
            out = subprocess.run([tool, "--", name], capture_output=True, text=True, timeout=5)
            d = out.stdout.strip()
            if out.returncode == 0 and d and not d.startswith("_Z"):
                return d
        except Exception:
            pass
    m = re.match(r"_ZN(.*)", name)
    if m:
        s = m.group(1); idents = []; i = 0
        while i < len(s) and s[i].isdigit():
            j = i
            while j < len(s) and s[j].isdigit():
                j += 1
            ln = int(s[i:j]); idents.append(s[j:j + ln]); i = j + ln
        if idents:
            return idents[-1]  # innermost = function name (drops template/param mangling)
    return name


def short(n):
    n=_demangle(str(n))
    n=n.split("(")[0]
    for pre in ("void ","aiter::","_ZN5aiter","_ZN7sgl_hip","std::","__hip_","c10::"): n=n.replace(pre,"")
    return n[:52]



# ---------------------------------------------------------------- ATOM side
def canon(sec, kn=""):
    # kernel-name override first: allreduce / rmsnorm are comm/norm regardless of
    # which module-section they were attributed to (e.g. cross_device_reduce that
    # SGLang emits inside the MoE section is really a TP allreduce = norm/comm).
    k = str(kn).lower()
    if ("cross_device_reduce" in k or "all_reduce" in k or "allreduce" in k
            or "nccl" in k or "reduce_1stage" in k):
        return "norm/comm"
    # q/k rmsnorm is MLA attention's internal q/k normalization (after q_a/kv_a
    # down-proj), NOT the residual-stream input/post-attn layernorm. Force it to
    # MLA_attention on both sides — ATOM annotates it under a generic "rmsnorm"
    # module (which the "norm" rule below would wrongly send to norm/comm), while
    # SGLang annotates it under DeepseekV2AttentionMLA.
    if "fused_qk_rmsnorm" in k or "qk_rmsnorm" in k:
        return "MLA_attention"
    # DSA sparse indexer kernels are part of attention (produce the topk selection);
    # classify by name so ATOM's (which land under generic annotations / "other")
    # and SGLang's line up in MLA_attention on both sides.
    if any(t in k for t in ("kn_entry_2c", "paged_mqa_logits", "topk_transform",
                            "radix_topk", "convert_req_index", "indexer_k_quant",
                            "hadamard", "wv_splitk")):
        return "MLA_attention"
    s = str(sec).lower()
    if s.startswith("prepare_"):
        return "norm/comm"
    if ("attention" in s or "attn" in s or "mla_decode" in s or "rope_and_kv" in s
            or "q_proj_and_k_up" in s or "v_up_proj_and_o" in s or "kv_cache" in s
            or "indexer" in s):
        return "MLA_attention"
    if any(t in s for t in ("moe", "mlp", "expert", "gate")):
        return "MoE/MLP"
    if "norm" in s or "nccl" in s or "allreduce" in s or "comm" in s:
        return "norm/comm"
    return "other"

def graph_name_avg(evs):
    ks=_k(evs); p,t=_dom(ks); ks=[k for k in ks if (k["pid"],k["tid"])==(p,t)]
    ga=sorted([a for a in _g(evs) if a["name"].startswith("decode[")],key=lambda a:a["ts"])
    s=ga[len(ga)//2]; s0,s1=s["ts"],s["ts"]+s["dur"]
    agg=defaultdict(lambda:[0,0.0])
    for k in ks:
        if s0<=k["ts"]<s1: agg[k["name"]][0]+=1; agg[k["name"]][1]+=k["dur"]
    return {n:v[1]/v[0] for n,v in agg.items()}


def atom_layer(evs, want_full=False):
    ks=_k(evs); p,t=_dom(ks); ks=[k for k in ks if (k["pid"],k["tid"])==(p,t)]
    ga=[a for a in _g(evs) if (a["pid"],a["tid"])==(p,t)]
    l0=sorted([a for a in ga if a["name"].startswith("model.layers.0.")],key=lambda a:a["ts"])
    key=l0[0]["name"]; starts=sorted(a["ts"] for a in ga if a["name"]==key)
    segs=[(starts[i],starts[i+1]) for i in range(len(starts)-1)]
    kts=sorted(k["ts"] for k in ks); kd={k["ts"]:k["dur"] for k in ks}
    def sd(a,b):
        lo=bisect.bisect_left(kts,a);hi=bisect.bisect_left(kts,b);return sum(kd[x] for x in kts[lo:hi])
    durs=[sd(*s) for s in segs]
    f0,f1=segs[sorted(range(len(segs)),key=lambda i:durs[i])[len(segs)//2]]
    lay=defaultdict(list)
    for a in ga:
        if f0<=a["ts"]<f1:
            m=re.match(r"model\.layers\.(\d+)\.",a["name"])
            if m: lay[int(m.group(1))].append(a["ts"])
    layers=sorted(lay); lstart={n:min(lay[n]) for n in layers}
    def _has_indexer(s0,s1):
        # "full" indexer layer = runs the topk indexer this step (vs "shared" which
        # reuses a prior layer's selection). Detected by the indexer kernels.
        return any(("paged_mqa_logits" in k["name"] or "kn_entry_2c" in k["name"]
                    or "topk_transform" in k["name"]) and s0<=k["ts"]<s1 for k in ks)
    cands=[]
    for i,n in enumerate(layers):
        s0=lstart[n]; s1=lstart[layers[i+1]] if i+1<len(layers) else f1
        if n>=3 and any(a["name"]=="mxfp4_moe" and s0<=a["ts"]<s1 for a in ga):
            cands.append((n,s0,s1,_has_indexer(s0,s1)))
    chosen=None
    for (n,s0,s1,full) in cands:
        if full==want_full: chosen=(n,s0,s1); break
    if chosen is None and cands:
        n,s0,s1,_=cands[0]; chosen=(n,s0,s1)
    if chosen is None:
        n=layers[len(layers)//2]; i=layers.index(n)
        chosen=(n,lstart[n],lstart[layers[i+1]] if i+1<len(layers) else f1)
    n,s0,s1=chosen
    labs=sorted([a for a in ga if s0<=a["ts"]<s1 and not a["name"].startswith("decode")
                 and not a["name"].startswith("##")],key=lambda a:a["ts"])
    lts=[a["ts"] for a in labs]
    def inner(k):
        # smallest-duration annotation that *actually encloses* the kernel start
        # (a.ts <= k.ts < a.end). labs[:pos] all start at/before k.ts; keep only
        # those still open at k.ts, then take the innermost (smallest dur).
        pos=bisect.bisect_right(lts,k["ts"]);best=None;bd=1e18
        for j in range(pos-1,-1,-1):
            a=labs[j]
            if a["ts"]+a["dur"]<=k["ts"]: continue  # already closed -> not enclosing
            if a["dur"]<bd: best,bd=a,a["dur"]
        return best
    seq=[]
    for k in sorted([k for k in ks if s0<=k["ts"]<s1],key=lambda k:k["ts"]):
        a=inner(k); raw=a["name"] if a else "(unlabeled)"
        seq.append((canon(raw, k["name"]), k["name"]))
    return n, seq



def _sglang_kernel_time_avg(evs):
    """avg duration per kernel name over one median decode step of the graph-on trace."""
    ks = _k(evs); p, t = _dom(ks); ks = [k for k in ks if (k["pid"], k["tid"]) == (p, t)]
    ann = [a for a in _g(evs) if (a["pid"], a["tid"]) == (p, t)]
    steps = sorted([a for a in ann if a["name"].startswith("step[DECODE") or a["name"].startswith("decode[")],
                   key=lambda a: a["ts"])
    if steps:
        s = steps[len(steps) // 2]; s0, s1 = s["ts"], s["ts"] + s["dur"]
        ks = [k for k in ks if s0 <= k["ts"] < s1]
    agg = defaultdict(lambda: [0, 0.0])
    for k in ks:
        agg[k["name"]][0] += 1; agg[k["name"]][1] += k["dur"]
    return {n: v[1] / v[0] for n, v in agg.items()}


def _canon_positional(kn, state):
    """Section for a SGLang kernel using name + running state ('MLA'|'MoE').
    hgemm is ambiguous (MLA q/o-proj vs MoE/dense gate/up/down) so it follows state."""
    k = kn.lower()
    if any(t in k for t in AR_KERNELS) or "reduce_1stage" in k:
        return "norm/comm", state
    # residual-stream norm, including the load_rmsnorm half of the reduce_scatter +
    # load pair; fused_qk_rmsnorm is MLA's internal q/k norm and stays in attention.
    if "rmsnorm" in k and "qk_rmsnorm" not in k:
        return "norm/comm", state
    if any(t in k for t in ("mfma_moe", "moe_sorting", "grouped_topk", "moe_sort",
                            "append_shared_experts")):
        return "MoE/MLP", "MoE"
    if "act_and_mul" in k or ("silu" in k and "moe" not in k):
        return "MoE/MLP", "MoE"
    if "hgemm" in k:
        return ("MoE/MLP" if state == "MoE" else "MLA_attention"), state
    return "MLA_attention", state


def sglang_layer_glm52(struct_evs, timeavg, want_full):
    ks = _k(struct_evs); p, t = _dom(ks); ks = [k for k in ks if (k["pid"], k["tid"]) == (p, t)]
    ks.sort(key=lambda k: k["ts"])
    ann = [a for a in _g(struct_evs) if (a["pid"], a["tid"]) == (p, t)]
    steps = sorted([a for a in ann if a["name"].startswith("step[DECODE")], key=lambda a: a["ts"])
    if not steps:
        raise SystemExit("no step[DECODE ...] annotation in sglang struct trace")
    s = steps[len(steps) // 2]; s0, s1 = s["ts"], s["ts"] + s["dur"]
    sk = [k for k in ks if s0 <= k["ts"] < s1]
    rms = sorted(k["ts"] for k in sk if "fused_qk_rmsnorm" in k["name"])
    ar = sorted(k["ts"] for k in sk if any(t in k["name"] for t in AR_KERNELS))
    if len(rms) < 5:
        raise SystemExit(f"expected ~78 fused_qk_rmsnorm anchors, got {len(rms)}")

    def boundary(i):
        """Start of layer i: the TP all-reduce that closes layer i-1, or layer i's own
        q/k-rmsnorm when that reduce is fused away (GLM-5.2 traces where the per-layer
        reduce is captured under a name we do not know: taking the nearest EARLIER
        reduce would reach back several layers and collapse the span to nothing)."""
        if i >= len(rms):
            return s1
        prev = rms[i - 1] if i else s0
        j = bisect.bisect_left(ar, rms[i]) - 1
        return ar[j] if j >= 0 and ar[j] > prev else rms[i]

    def layer_span(i):
        return boundary(i), boundary(i + 1)

    def has_indexer(a, b):
        return any(("paged_mqa_logits" in k["name"] or "kn_entry_2c" in k["name"]
                    or "topk_transform" in k["name"]) and a <= k["ts"] < b for k in sk)

    # pick target layer: MoE (i>=FIRST_K_DENSE) with indexer==want_full
    pick = None
    for i in range(len(rms) - 1):
        if i < FIRST_K_DENSE:
            continue
        a, b = layer_span(i)
        if has_indexer(a, b) == want_full:
            pick = i; break
    if pick is None:  # fallback: any layer matching want_full
        for i in range(len(rms) - 1):
            a, b = layer_span(i)
            if has_indexer(a, b) == want_full:
                pick = i; break
    if pick is None:
        pick = FIRST_K_DENSE
    a, b = layer_span(pick)
    seq = []
    state = "MLA"
    for k in sorted([k for k in sk if a <= k["ts"] < b], key=lambda k: k["ts"]):
        sec, state = _canon_positional(k["name"], state)
        seq.append((sec, k["name"], round(timeavg.get(k["name"], 0.0), 2)))
    return pick, seq


BOLD = Font(name="Arial", bold=True); REG = Font(name="Arial")
HFILL = PatternFill("solid", fgColor="D9E1F2")
SF = {"MLA_attention": PatternFill("solid", fgColor="DDEBF7"),
      "MoE/MLP": PatternFill("solid", fgColor="E2EFDA"),
      "norm/comm": PatternFill("solid", fgColor="FCE4D6"),
      "other": PatternFill("solid", fgColor="F2F2F2")}
SECTIONS = ["MLA_attention", "MoE/MLP", "norm/comm", "other"]


def _analyze(atom_time, atom_struct_evs, sg_time, sg_struct_evs, want_full):
    ln, atom_seq = atom_layer(atom_struct_evs, want_full)
    sg_layer, sg_seq = sglang_layer_glm52(sg_struct_evs, sg_time, want_full)
    A = [(s, short(k), round(atom_time.get(k, 0.0), 2)) for s, k in atom_seq]
    S = [(s, short(k), v) for s, k, v in sg_seq]
    asum = defaultdict(float); ssum = defaultdict(float)
    for s, k, u in A: asum[s] += u
    for s, k, u in S: ssum[s] += u
    # A kernel with no time was in the structure trace but not in the timed window.
    # For ATOM that usually means the no-graph trace captured a PREFILL forward (its
    # only kind, in some runs) while timing came from a decode step, so the column is
    # an undercount and must not be read as this stack's per-layer cost.
    for side, rows in (("ATOM", A), ("SGLang", S)):
        miss = sum(1 for _, _, u in rows if not u)
        if miss:
            print(f"[warn] {side}: {miss}/{len(rows)} kernels of the layer have no timing "
                  f"in the timed window — the {side} subtotals are an undercount")
    return ln, sg_layer, A, S, asum, ssum


BLOCK_W = 8   # columns per block: # | ATOM sec | ATOM kernel | ATOM us | spacer | SGLang sec | SGLang kernel | SGLang us
GAP = 3       # blank columns between the shared and full blocks


def _write_block(ws, c0, label, A, S, asum, ssum, body_rows):
    ws.cell(1, c0, label).font = BOLD
    hdr = ["#", "ATOM section", "ATOM kernel", "ATOM us", "", "SGLang section", "SGLang kernel", "SGLang us"]
    for j, h in enumerate(hdr):
        cell = ws.cell(2, c0 + j, h); cell.font = BOLD; cell.fill = HFILL; cell.alignment = Alignment(horizontal="center")
    r = 3
    for i in range(max(len(A), len(S))):
        asec, ak, au = A[i] if i < len(A) else ("", "", "")
        ssec, sk, su = S[i] if i < len(S) else ("", "", "")
        ws.cell(r, c0, i).font = REG
        for j, v in ((1, asec), (2, ak), (3, au)):
            cell = ws.cell(r, c0 + j, v if v != "" else None); cell.font = REG
            if j == 1 and asec in SF: cell.fill = SF[asec]
        for j, v in ((5, ssec), (6, sk), (7, su)):
            cell = ws.cell(r, c0 + j, v if v != "" else None); cell.font = REG
            if j == 5 and ssec in SF: cell.fill = SF[ssec]
        r += 1
    sr = body_rows + 4  # aligned subtotals across blocks
    ws.cell(sr, c0 + 1, "Section subtotals (us/layer)").font = BOLD; sr += 1
    ws.cell(sr, c0 + 1, "Section").font = BOLD; ws.cell(sr, c0 + 3, "ATOM").font = BOLD; ws.cell(sr, c0 + 7, "SGLang").font = BOLD; sr += 1
    for sec in SECTIONS:
        ws.cell(sr, c0 + 1, sec).font = REG
        if sec in SF: ws.cell(sr, c0 + 1).fill = SF[sec]
        ws.cell(sr, c0 + 3, round(asum.get(sec, 0.0), 1)).font = REG
        ws.cell(sr, c0 + 7, round(ssum.get(sec, 0.0), 1)).font = REG; sr += 1
    ws.cell(sr, c0 + 1, "TOTAL").font = BOLD
    ws.cell(sr, c0 + 3, round(sum(asum.values()), 1)).font = BOLD
    ws.cell(sr, c0 + 7, round(sum(ssum.values()), 1)).font = BOLD


def build(atom_time_p, atom_struct_p, sg_time_p, sg_struct_p, out_p, kinds, title):
    # load each (large) trace once, reuse for both shared & full
    atom_time = graph_name_avg(load(atom_time_p))
    atom_struct_evs = load(atom_struct_p)
    sg_time = _sglang_kernel_time_avg(load(sg_time_p))
    sg_struct_evs = load(sg_struct_p)

    order = [k for k in ("shared", "full") if k in kinds]
    data = {}
    for kind in order:
        ln, sgl, A, S, asum, ssum = _analyze(atom_time, atom_struct_evs, sg_time, sg_struct_evs, kind == "full")
        data[kind] = (ln, sgl, A, S, asum, ssum)
        print(f"[{kind}] ATOM layer {ln} | SGLang layer {sgl} | "
              + "ATOM " + ",".join(f"{s}={asum.get(s,0):.1f}" for s in SECTIONS)
              + " | SGLang " + ",".join(f"{s}={ssum.get(s,0):.1f}" for s in SECTIONS))

    body_rows = max(max(len(d[2]), len(d[3])) for d in data.values())
    wb = Workbook(); ws = wb.active; ws.title = "SideBySide"
    for bi, kind in enumerate(order):
        ln, sgl, A, S, asum, ssum = data[kind]
        c0 = 1 + bi * (BLOCK_W + GAP)
        _write_block(ws, c0, f"{kind.upper()}  (ATOM layer {ln} | SGLang layer {sgl})  —  {title}",
                     A, S, asum, ssum, body_rows)
    for bi in range(len(order)):
        c0 = 1 + bi * (BLOCK_W + GAP)
        for j, w in enumerate([5, 14, 46, 9, 3, 14, 46, 9]):
            ws.column_dimensions[chr(64 + c0 + j)].width = w
    ws.freeze_panes = "A3"
    wb.save(out_p); print("written", out_p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atom-time", required=True); ap.add_argument("--atom-struct", required=True)
    ap.add_argument("--sglang-time", required=True); ap.add_argument("--sglang-struct", required=True)
    ap.add_argument("--layer-kind", choices=["shared", "full", "both"], default="both",
                    help="both (default) -> one workbook with Summary + Shared + Full sheets")
    ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
    a = ap.parse_args()
    kinds = ["shared", "full"] if a.layer_kind == "both" else [a.layer_kind]
    title = a.title or "GLM-5.2 MoE decode layer (ATOM | SGLang), CALL order"
    build(a.atom_time, a.atom_struct, a.sglang_time, a.sglang_struct, a.out, kinds, title)


if __name__ == "__main__":
    main()
