#!/usr/bin/env bash
# Idempotent patch: full fix for AITER MLA backend with --page-size > 1
# on ROCm (MI355X).
#
# Background
# ----------
# Upstream PR sgl-project/sglang#25556 promised to fix correctness for
# AITER MLA backend with --page-size > 1, but the version that landed
# in the PR (commit 6db65bd) only contains fix #1 of the two described
# in the PR text. Fix #2 was force-pushed away during code review
# (gemini-bot pointed out the conditional missed an MLA case).
#
# This script applies BOTH fixes:
#
#   fix #1 (from upstream PR 6db65bd) -- correctness for the fp8
#          prefill attention path only:
#       make_mla_prefill_ps_meta_data:
#           block_size       = self.page_size          ->  1
#           kvlen_granularity= max(128, self.page_size)-> 128
#       The metadata is consumed by mla_prefill_ps_asm_fwd (via
#       mla_fp8_prefill_attn), which receives K/V with
#       fp8_prefill_kv_indices = arange(total_s) (linear per-token
#       layout). Passing the cache's page_size made the kernel compute
#       wrong work splits; work_info_set.kv_len flipped from `seq_len`
#       to `seq_len / page_size`, garbling prefill output.
#       Gated by:  if _use_fp8_prefill_attn (SGLANG_AITER_FP8_PREFILL_ATTN).
#       Without this fix: gsm8k accuracy collapses 0.975 -> 0.005 when
#       SGLANG_AITER_FP8_PREFILL_ATTN=1 + page_size>1.
#
#   fix #2 (originally in PR description, removed before merge, here
#          re-applied with the gemini-bot review suggestion):
#       init_cuda_graph_state:
#           cuda_graph_kv_indices was sized at BLOCK granularity
#           (max_bs * ceil(max_context_len / page_size)) but Triton
#           fills it at TOKEN granularity via cumsum(seq_lens). With
#           page_size>1 the buffer is page_size x too small ->
#           silent overrun -> "Memory access fault by GPU node-N" on
#           any non-trivial decode context.
#       Hits BOTH SGLANG_AITER_FP8_PREFILL_ATTN=0 and =1, regardless
#       of the fp8 prefill path -- it's a cuda graph decode buffer.
#
# Why this patch exists
# ---------------------
# Until upstream merges fix #2, running with --page-size 64 on AITER
# MLA reliably crashes:
#
#   page_size=64 + ISL=8192 conc=16   -> reproducible GPU memory
#                                         access fault on all 8 GPUs
#
# Verified locally (MI355X, lmsysorg/sglang:v0.5.12-rocm700-mi35x):
#   page_size=1 (InferenceX baseline)                Median TTFT 204.85 ms
#   page_size=64 + only PR upstream fix#1            CRASH
#   page_size=64 + this patch (fix#1 + fix#2)        Median TTFT 207.64 ms
#
# Within-noise match confirms the PR's own statement: "Decode is
# essentially flat across page sizes -- kernel only sees token slot
# ids" (allocator flattens page id away before the kernel sees it).
#
# Usage
# -----
#   bash /home/jacchang/local-patches/HiCachePatch/pr25556-aiter-mla-page-size.sh
#   bash /home/jacchang/local-patches/HiCachePatch/pr25556-aiter-mla-page-size.sh /path/to/sglang/repo
#
# Default target is /sgl-workspace/sglang. Do NOT patch the host edit
# repo at ~/PR/sglang (commits originate from there and the patch
# would pollute pushes).
#
# Exit codes:
#   0 = applied OR already applied (safe to re-run, idempotent)
#   2 = target block found but format differs (upstream changed,
#       inspect manually)
#   3 = target file not found
set -euo pipefail

REPO="${1:-/sgl-workspace/sglang}"
TARGET="$REPO/python/sglang/srt/layers/attention/aiter_backend.py"

if [[ ! -f "$TARGET" ]]; then
  echo "ERROR: $TARGET not found" >&2
  exit 3
fi

python3 - "$TARGET" <<'PY'
import sys, pathlib

p = pathlib.Path(sys.argv[1])
src = p.read_text()

# ---------------------------------------------------------------------
# Fix #1: make_mla_prefill_ps_meta_data: pin block_size=1 / kvlen=128
# ---------------------------------------------------------------------
fix1_old = (
    "        kvlen_granularity = max(128, self.page_size)\n"
    "        block_size = self.page_size\n"
)
fix1_new = (
    "        # HiCachePatch (PR #25556 fix #1): MLA kernel sees linear\n"
    "        # per-token layout; never pass page_size here.\n"
    "        kvlen_granularity = 128\n"
    "        block_size = 1\n"
)
fix1_applied_marker = "HiCachePatch (PR #25556 fix #1)"

if fix1_applied_marker in src:
    fix1_status = "already applied"
elif fix1_old in src:
    src = src.replace(fix1_old, fix1_new, 1)
    fix1_status = "applied"
elif "block_size = 1\n" in src and "kvlen_granularity = 128\n" in src:
    # Upstream-merged variant (PR fix #1 commit without our comment marker)
    fix1_status = "already applied (upstream)"
else:
    print(
        "[HiCachePatch] FIX #1 ERROR: expected lines not found in\n"
        f"  {p}\nExpected:\n{fix1_old}",
        file=sys.stderr,
    )
    sys.exit(2)

# ---------------------------------------------------------------------
# Fix #2: init_cuda_graph_state: token-granularity buffer for MLA
# ---------------------------------------------------------------------
fix2_old = (
    "        if kv_indices_buf is None:\n"
    "            max_num_blocks_per_seq = (\n"
    "                self.max_context_len + self.page_size - 1\n"
    "            ) // self.page_size\n"
    "            self.cuda_graph_kv_indices = torch.zeros(\n"
    "                (max_bs * max_num_blocks_per_seq),\n"
    "                dtype=torch.int32,\n"
    "                device=self.device,\n"
    "            )\n"
)
fix2_new = (
    "        if kv_indices_buf is None:\n"
    "            # HiCachePatch (PR #25556 fix #2 + gemini-bot review):\n"
    "            # AITER MLA kernels operate at token granularity (per-token\n"
    "            # slot ids), so cuda_graph_kv_indices must be sized at token\n"
    "            # granularity; otherwise create_flashinfer_kv_indices_triton\n"
    "            # silently overruns the buffer when page_size > 1 and long\n"
    "            # decode contexts are present (-> GPU memory access fault).\n"
    "            if not self.use_triton_unified_attention or self.use_mla:\n"
    "                kv_indices_buffer_size = max_bs * self.max_context_len\n"
    "            else:\n"
    "                max_num_blocks_per_seq = (\n"
    "                    self.max_context_len + self.page_size - 1\n"
    "                ) // self.page_size\n"
    "                kv_indices_buffer_size = max_bs * max_num_blocks_per_seq\n"
    "            self.cuda_graph_kv_indices = torch.zeros(\n"
    "                (kv_indices_buffer_size,),\n"
    "                dtype=torch.int32,\n"
    "                device=self.device,\n"
    "            )\n"
)
fix2_applied_marker = "HiCachePatch (PR #25556 fix #2"
# Code-shape detection: any prior application of fix #2 (regardless
# of the comment marker) will introduce this exact conditional.
fix2_code_signature = (
    "if not self.use_triton_unified_attention or self.use_mla:"
)
# Upstream alternative fix that landed in some images instead of the PR
# #25556 form: instead of branching on use_mla / use_triton_unified, they
# multiply the block-granularity allocation by page_size to recover
# token-granularity. Functionally equivalent for our purposes, so we
# treat it as "already applied" rather than overwriting it.
fix2_upstream_signature = (
    "max_bs * max_num_blocks_per_seq * self.page_size"
)
fix2_old_signature = (
    "(max_bs * max_num_blocks_per_seq),\n"
    "                dtype=torch.int32"
)

if fix2_applied_marker in src or fix2_code_signature in src:
    fix2_status = "already applied"
elif fix2_upstream_signature in src:
    fix2_status = "already applied (upstream alternative)"
elif fix2_old in src:
    src = src.replace(fix2_old, fix2_new, 1)
    fix2_status = "applied"
elif fix2_old_signature not in src:
    # Neither the buggy original nor any known fix variant is present.
    # Upstream has changed shape; bail loudly.
    print(
        "[HiCachePatch] FIX #2 ERROR: expected block not found in\n"
        f"  {p}\nExpected verbatim:\n{fix2_old}",
        file=sys.stderr,
    )
    sys.exit(2)
else:
    # Buggy original is present but doesn't exactly match (e.g. minor
    # whitespace drift). Refuse to guess.
    print(
        "[HiCachePatch] FIX #2 ERROR: buggy block detected but format\n"
        "drifted from expected. Inspect manually.",
        file=sys.stderr,
    )
    sys.exit(2)

p.write_text(src)
print(f"[HiCachePatch] {p}")
print(f"  fix #1 (make_mla_prefill_ps_meta_data): {fix1_status}")
print(f"  fix #2 (init_cuda_graph_state):         {fix2_status}")
PY
