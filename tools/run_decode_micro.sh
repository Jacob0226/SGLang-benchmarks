#!/usr/bin/env bash
# Reproducible TileLang-vs-Triton GLM-5.2 decode sparse-MLA kernel benchmark.
#
# Both kernels are imported from a real sglang checkout of PR #30575
# (https://github.com/sgl-project/sglang/pull/30575) -- NO hand-copied kernel.
# The tilelang decode kernel ("main_kernel") is unchanged by the PR, so pinning
# the whole run to the PR commit gives an apples-to-apples, self-contained repro.
#
# Usage (inside the ROCm container):
#   ./run_decode_micro.sh                 # total per-config timing (microbench)
#   ./run_decode_micro.sh --breakdown     # per-kernel split/reduce breakdown
#   ./run_decode_micro.sh --iters 300 --concs 4 8 16 32 64 --seqlens 1024 8192
#
# Env overrides:
#   SGLANG_SRC   base sglang repo to fetch the PR into (default: ~/PR/sglang)
#   PR_WORKTREE  where to checkout the PR (default: ~/PR/sglang-pr30575)
#   PR_REF       PR ref to fetch (default: pull/30575/head)
#   PR_SHA       pin commit (default: 9b8f645ea, PR #30575 head at time of writing)
#   GPUS         HIP_VISIBLE_DEVICES (default: 4,5,6,7 -- avoid busy 0-3)
set -euo pipefail

SGLANG_SRC="${SGLANG_SRC:-$HOME/PR/sglang}"
PR_WORKTREE="${PR_WORKTREE:-$HOME/PR/sglang-pr30575}"
PR_REF="${PR_REF:-pull/30575/head}"
PR_SHA="${PR_SHA:-9b8f645ea}"
GPUS="${GPUS:-4,5,6,7}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pick microbench (default) or breakdown, strip our own flag from args.
SCRIPT="glm5_decode_attn_microbench.py"
ARGS=()
for a in "$@"; do
  if [ "$a" == "--breakdown" ]; then SCRIPT="glm5_decode_attn_breakdown.py"; else ARGS+=("$a"); fi
done

# Set up the PR worktree once (idempotent).
if [ ! -f "$PR_WORKTREE/python/sglang/srt/layers/attention/dsa/triton_sparse_mla_decode.py" ]; then
  echo ">>> Setting up PR #30575 worktree at $PR_WORKTREE (pin $PR_SHA)"
  git -C "$SGLANG_SRC" fetch upstream "$PR_REF"
  git -C "$SGLANG_SRC" worktree add "$PR_WORKTREE" "$PR_SHA"
fi

echo ">>> sglang = $PR_WORKTREE (PR #30575), GPUs = $GPUS, script = $SCRIPT"
HIP_VISIBLE_DEVICES="$GPUS" CUDA_VISIBLE_DEVICES="$GPUS" \
  PYTHONPATH="$PR_WORKTREE/python:${PYTHONPATH:-}" \
  python3 "$HERE/$SCRIPT" "${ARGS[@]}"
