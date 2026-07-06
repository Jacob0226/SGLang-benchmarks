# SGLang GLM-5.2 (DSA) fp8 MLA decode — port spec

Goal: make SGLang's GLM-5.2-MXFP4 decode use the aiter **fp8 MLA decode** kernel
(`aiter.mla.mla_decode_fwd`), matching ATOM, instead of the bf16 tilelang
`main_kernel` (×2). This is the remaining gap to ATOM after opt#1.

Measured (MI355X TP4, GLM-5.2-MXFP4, i1024/o1024 conc4, median TPOT):
- SGLang baseline (tilelang decode): 15.66 ms, GSM8K 0.940
- SGLang + opt#1 (fp8 absorbed bmm): 14.89 ms, GSM8K 0.929
- ATOM (aiter fp8 decode): 11.73 ms  ← target

Env: container `jacchang_GLM5`, image `rocm/sgl-dev:v0.5.14-rocm720-mi35x-20260628`,
SGLang editable at `/sgl-workspace/sglang` (branch main), aiter at `/sgl-workspace/aiter`.
Repro test: `~/run_sgl_test.sh <tag>` (server GLM.sh MI355X args → GSM8K 1319 → conc4 TPOT).
Trigger the path with `--nsa-decode-backend aiter` (aka `--dsa-decode-backend aiter`).

--------------------------------------------------------------------------------
## Status: aiter DSA decode path is unfinished/broken in SGLang

`dsa_backend.py::DSAAttnBackend._forward_aiter` GPU-memory-faults (both cuda-graph
capture AND eager `--disable-cuda-graph`). Instrumented args right before the
`mla_decode_fwd` call (decode warmup, bs=8, 1 tok/req):

    q_kernel = (8,16,576) float8_e4m3fn
    o_kernel = (8,16,512) float8_e4m3fn        # BUG: should be bf16
    kv_view  = (2634752,1,1,576) float8_e4m3fn
    cu_seqlens_q = (9,): [0,1,2,3,4,5,...]
    kv_indptr    = (2049,): [0,7,14,21,28,35,...]   # COMPACTED, variable length
    kv_indices   = (4194304,) , kv_max=518 , pool=2634752
    persist keys = work_meta_data, work_indptr, work_info_set, reduce_indptr,
                   reduce_final_map, reduce_partial_map, intra_batch_mode, num_kv_splits=64
    topk = 2048 , max_seq_len_q = 1
    q_scale = None      # BUG
    kv_scale = nan      # BUG (should be 1.0)

Blocker chain observed while patching:
1. q is fp8 but `q_scale=None` → aiter asm aborts: "fp8 Q requires q_scale and kv_scale".
2. Set q_scale=ones → `o=empty_like(q)`=fp8 → "kn_mla_reduce_v1 doesn't support output type Float8_e4m3fn".
3. Force o=bf16 + q_scale=ones → GPU **memory access fault** inside mla_decode_fwd
   (eager too), i.e. the index/metadata layout does not match the kernel contract.

--------------------------------------------------------------------------------
## The two incompatible schemes

### SGLang (current, faults) — dsa_backend.py
- `_forward_aiter` (~L2160-2245):
  - q_all fp8 → q_kernel fp8; `o = torch.empty_like(q)` (inherits **fp8**).
  - `q_scale=None`; `kv_scale=torch.ones(())` when fp8 (empirically prints nan).
  - `get_valid_kv_indices(page_table_1, kv_indptr, kv_indices, bs)` →
    **compacted variable-length** kv_indices + kv_indptr (drops -1).
  - `_prepare_aiter_dsa_decode_metadata` (~L513-566) → `get_mla_metadata_v1(...,
    topk=dsa_index_topk, max_split_per_batch=64, page_size=1, kv_granularity=16,
    intra_batch_mode=True)`; buffers from `_make_aiter_dsa_decode_metadata_buffer`
    (~L429) via `get_mla_metadata_info_v1`.
  - `mla_decode_fwd(q_kernel, kv.view(-1,1,1,head_dim), o_kernel, cu_seqlens_q,
    kv_indptr, kv_indices, kv_last_page_lens, max_seq_len_q, q_scale=None,
    kv_scale, **persistent)` (~L2229).

### ATOM (working) — atom/model_ops/attention_mla.py + plugin/sglang/attention_backend/sparse_mla_indexer.py
- Indexer `sparse_attn_indexer_sglang_plugin_mode` (sparse_mla_indexer.py L386):
  fills **fixed `topk_indices_buffer[:bs, :2048]` (-1 padded)** via
  `deepgemm_fp8_paged_mqa_logits` + `top_k_per_row_decode` (L497-505);
  `seq_lens_i32 = forward_batch.seq_lens[:bs]`.
- Decode `_forward_decode` (attention_mla.py L954-1119):
  - `o = torch.empty(B, padded_num_heads, kv_lora_rank, dtype=self.dtype)` → **bf16**.
  - `_q_scale = _k_scale = one_scale = torch.tensor(1.0)` (L238-240).
  - sparse: `paged_kv_indices = self.sparse_kv_indices_buffer` (L869-872) — the
    **fixed [tokens, 2048]** buffer (NOT compacted); `paged_kv_indptr = sparse_kv_indptr`.
  - `mla_decode_fwd(q, kv.view(-1,page_size,1,dim), o, cu_seqlens_q, kv_indptr,
    sparse_kv_indices_buffer, kv_last_page_lens, max_q_len, page_size=page_size,
    num_kv_splits=None if seg else 16, sm_scale, work_meta_data/work_indptr/
    work_info_set/reduce_indptr/reduce_final_map/reduce_partial_map,
    q_scale=one, kv_scale=one)` (L1092-1114).

--------------------------------------------------------------------------------
## Incompatibility points (root causes)

| # | Item | SGLang (faults) | ATOM (works) | Fix |
|---|------|-----------------|--------------|-----|
| 1 | **sparse index layout** | compacted var-len flat + indptr (`get_valid_kv_indices`) | **fixed `[tokens, 2048]` (-1 padded)** + per-token `seq_len` | pass the fixed topk buffer (SGLang `page_table_1` is already `[bs, topk]` -1-padded, used raw by tilelang) instead of compacting |
| 2 | **metadata builder** | `get_mla_metadata_v1` from compacted indptr, splits=64 | `get_mla_metadata` from fixed buffer + seq_len, splits=16 | build metadata from the fixed buffer/seq_len the way the kernel expects |
| 3 | **output dtype** | `o = empty_like(q)` → **fp8** | `o` = **bf16** (`self.dtype`) | force o bf16 (`kn_mla_reduce_v1` rejects fp8 out) |
| 4 | **q_scale** | **None** (never set) → asm abort | `one_scale` (1.0) | pass q_scale=1.0 |
| 5 | **kv_scale** | empirically **nan** | 1.0 | ensure kv_scale=1.0 |
| 6 | num_kv_splits | 64 | 16 (non-seg) | match kernel/buffer sizing |

**CORRECTED primary suspect (after reading ATOM `attention_mla.py::_forward_decode`
and `attentions/aiter_mla.py`):** the earlier "fixed `[bs,topk]` vs compacted
buffer" theory is **WRONG**. ATOM's sparse fp8 decode also feeds the aiter
`mla_decode_fwd` a **compacted CSR** buffer — `sparse_kv_indptr` +
`sparse_kv_indices` (`self.sparse_kv_indices_buffer`) — the *same* ragged format
SGLang produces via `get_valid_kv_indices`. Both call the exact same
`aiter.mla.mla_decode_fwd(q, kv, o, qo_indptr, kv_indptr, kv_indices,
kv_last_page_lens, ...)`.

The real divergence is the **persistent work-scheduling metadata**:
- ATOM builds `work_meta_data / work_indptr / work_info_set / reduce_indptr /
  reduce_final_map / reduce_partial_map` via aiter `get_mla_metadata_v1` /
  `get_mla_metadata_info_v1` (see `atom/model_ops/attentions/aiter_mla.py`),
  with `num_kv_splits=16` on the page_size=1 persistent path (or `None` on the
  seg / `ATOM_MLA_PAGE_SIZE>1` path), and `q_scale=kv_scale=1.0`.
- SGLang builds its own in `_prepare_aiter_dsa_decode_metadata`. When forced onto
  the fp8 branch it produced an OOB — i.e. its split/worker-buffer sizing for the
  **topk-clipped** sparse seqlens (`dsa_cache_seqlens_int32`) does not match what
  the asm kernel indexes. That mismatch (not the index buffer) is the fault.

--------------------------------------------------------------------------------
## Recommended fix

Keep SGLang's compacted `get_valid_kv_indices` (kv_indptr + kv_indices) — it is
already the right format. Only replace the **metadata build**:
1. Replace `_prepare_aiter_dsa_decode_metadata` with a faithful port of ATOM's
   `aiter_mla.py` persistent-buffer build: call `get_mla_metadata_v1` /
   `get_mla_metadata_info_v1` with the compacted `kv_indptr`, the topk-clipped
   `dsa_cache_seqlens_int32`, and `num_kv_splits=16` (page_size=1 path).
2. Pass the resulting `work_meta_data / work_indptr / work_info_set /
   reduce_indptr / reduce_final_map / reduce_partial_map` into `mla_decode_fwd`.
3. `q_scale=kv_scale=torch.ones((), fp32)`; `o` bf16.
4. Gate on `ATOM_MLA_PAGE_SIZE` (=1 → persistent 16-split path; >1 → seg path
   which ATOM uses with `num_kv_splits=None` and padded q row stride).

**Needs HIP-level debugging** (rocgdb / AMD_LOG_LEVEL / bounds) to confirm the
kernel's exact index contract, since the fault is an opaque asm OOB with no line
info. Reference: `atom/model_ops/attention_mla.py::_forward_decode` and
`atom/plugin/sglang/attention_backend/sparse_mla_indexer.py`.

--------------------------------------------------------------------------------
## GLM-5.1 vs 5.2 indexer / layer design (from config.json, verified)

The DSA (NSA) sparse indexer is the biggest architectural difference between the
two model revisions, and it directly changes how many topk-selection kernels run
per decode step.

| key | GLM-5.1 | GLM-5.2 | meaning |
|---|---|---|---|
| `indexer_types` | (absent → all `full`) | `['full','full','full','shared','shared','shared','full','shared', ...]` len 78 | 5.2 marks most layers `shared` |
| `index_topk_freq` | (absent → every step) | `4` | 5.2 recomputes topk only every 4th decode step |
| `index_skip_topk_offset` | (absent) | `3` | phase offset for the freq schedule |
| `index_share_for_mtp_iteration` | (absent) | `True` | MTP/nextn reuses the base step's topk |
| `mlp_layer_types` | (implicit via `first_k_dense_replace=3`) | `['dense','dense','dense','sparse', ...]` | layers 0–2 dense, 3–77 MoE (same in both) |
| `index_topk` | 2048 | 2048 | selected KV budget per query, unchanged |

**`full` vs `shared` indexer layer:**
- `full`  = the layer runs the indexer itself (wq_b / weights_proj →
  `deepgemm_fp8_paged_mqa_logits` → `top_k_per_row`/radix topk) to *produce* the
  `[bs, topk]` selection.
- `shared` = the layer does **not** recompute topk; it **reuses** the selection
  produced by the nearest preceding `full` layer. So the indexer-selection
  kernels are simply absent from a `shared` layer's decode step.

**Net effect (5.1 → 5.2):**
- 5.1: every one of the 75 sparse layers runs a full indexer topk **every decode
  step** → 75 × topk selections/step.
- 5.2: only `full` layers run it, and only every `index_topk_freq=4` steps →
  roughly an order of magnitude fewer `deepgemm_fp8_paged_mqa_logits` / topk
  kernels per step. This is the main reason 5.2's decode is cheaper on the
  indexer side while keeping the same `index_topk=2048` accuracy budget.

**Consequence for the side-by-side excel:** the profiled layer is **layer 3**,
which is `indexer_types[3]='shared'` and `mlp_layer_types[3]='sparse'` — i.e. a
**MoE layer with a shared (non-recomputing) indexer**. That is why its MLA
section shows no indexer-topk kernels. A `full` layer (e.g. layer 6) on a
recompute step would additionally show the wq_b / paged_mqa_logits / topk
kernels, adding cost that is *not* visible in the current sheet. For a port,
both paths must be handled: `full` (produce + write selection) and `shared`
(read selection).

--------------------------------------------------------------------------------
## Pinpointed divergence (the actual next experiment)

SGLang and ATOM decode call the **same** `aiter.mla.get_mla_metadata_v1` + the
**same** `mla_decode_fwd` on **compacted CSR** (`clip(context_len, topk)` cumsum).
Diffing the two runtime `get_mla_metadata_v1` invocations:

| arg | SGLang `_prepare_aiter_dsa_decode_metadata` | ATOM decode (`aiter_mla.py:1331`) |
|---|---|---|
| `intra_batch_mode` | **`True`** | **not passed (False)** |
| `topk` | **`self.dsa_index_topk` (2048)** | **not passed** |
| `nhead_kv` (arg 5) | `1` | `1` |
| `page_size` | `1` | `self.block_size` |
| `kv_granularity` | `16` | `max(block_size,16)` |
| `max_split_per_batch` | `aiter_dsa_max_split_per_batch` | `16` |
| `num_kv_splits` returned | `max_split_per_batch` | `16` (kernel arg) |

**Hypothesis:** `intra_batch_mode=True` + `topk=2048` puts the asm kernel into a
sparse "intra-batch" schedule that assumes a **topk-strided** work layout, but
the indices/indptr handed in are **compacted** → out-of-bounds. ATOM's working
decode does NOT use that mode.

**First experiment (small, not a rewrite):** in SGLang's
`_prepare_aiter_dsa_decode_metadata`, set `intra_batch_mode=False` and drop
`topk=` (match ATOM), keep `num_kv_splits=16`, `q_scale=kv_scale=1.0`, `o` bf16;
enable the fp8 decode branch and validate with GSM8K + TPOT. If the kernel needs
`intra_batch_mode=True`, then instead feed a **topk-strided** kv_indices buffer
(`page_table_1` flattened, size `tokens*topk`, `-1`→0 padded) with a fixed
`kv_indptr` stride of `topk` — the layout `intra_batch_mode` actually expects.

**Reframing:** SGLang already contains ATOM's MLA kernel + metadata builder, so
this is NOT a subsystem port — it is aligning ~2 metadata args on the fp8 sparse
decode branch. Much smaller than the spec's original "replace everything".

--------------------------------------------------------------------------------
## Done in this investigation
- opt#1 (fp8 MLA absorbed bmm for GLM) — landed on branch
  `jacob/glm-mla-fp8-absorbed-bmm` (Jacob0226/sglang). 15.66→14.89 ms, GSM8K 0.929.
- opt#3 (fused allreduce) — skipped; SGLang norm/comm already ≤ ATOM per trace.
- opt#2 diagnosis corrected: not a buffer-format port; both use compacted CSR +
  the same aiter kernel/metadata fn. Divergence isolated to `intra_batch_mode` /
  `topk` args in `_prepare_aiter_dsa_decode_metadata`. See table above.

## opt#2 RESOLVED — aiter fp8 decode now works, but is SLOWER than tilelang

The GPU-fault blocker is fixed. Enabling the aiter fp8 MLA decode core
(`--nsa-decode-backend aiter --kv-cache-dtype fp8_e4m3`) required 4 changes in
`dsa_backend.py` (saved as `opt2_aiter_fp8_decode.patch`):
1. `intra_batch_mode=True -> False` (buffer sizing, runtime `get_mla_metadata_v1`,
   and the kwarg passed into `mla_decode_fwd`) + drop `topk=` — fixes the OOB.
2. Add `q_scale=ones` when Q is fp8 (asm kernel asserts `fp8 Q requires q_scale`).
3. Force MLA output `o`/`o_kernel` to bf16 (`kn_mla_reduce_v1` rejects fp8 output).
4. `aiter_dsa_max_split_per_batch 64 -> 16` (match ATOM; no measurable effect).

**Validation (GLM-5.2-MXFP4, TP4, MI355X, isl1024/osl512, conc4):**

| decode backend | GSM8K (200q) | Median TPOT | Output tok/s |
|---|---|---|---|
| tilelang (baseline) | — | **14.52 ms** | 259.9 |
| aiter fp8 (opt#2)   | **0.945** | 24.45 ms | 158.7 |

**Conclusion:** correctness is perfect (GSM8K 0.945, 0 invalid) but the aiter fp8
decode core is ~69% **slower** than tilelang on MI355X for GLM-5.2. `num_kv_splits`
tuning (64->16) did not move it. tilelang is the InferenceX-tuned MI355X default
and wins here. ATOM's speed advantage in the side-by-side comes from its *whole*
attention subsystem (seg-MLA `page_size>1` kernel + fused projections + tuning),
not the decode core alone; swapping only the core kernel regresses. opt#2 is a
correctness/enablement result, NOT a perf win — do not merge as an optimization
on MI355X. Capturing ATOM's advantage would require the seg-MLA path, a much
larger port.
