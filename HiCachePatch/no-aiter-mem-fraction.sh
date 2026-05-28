#!/usr/bin/env bash
# Idempotent patch: remove the AMD aiter long-context mem_fraction_static
# adjustment from sglang's server_args.py.
#
# Why: that block silently multiplies mem_fraction_static by 0.85 whenever
# attention_backend=="aiter" and model context_len > 8192, even when the
# user explicitly passed --mem-fraction-static. This makes ROCm and NV
# tests use different KV cache pool sizes for the same flag value.
#
# Usage (inside the container):
#   bash /home/jacchang/local-patches/no-aiter-mem-fraction.sh
#   bash /home/jacchang/local-patches/no-aiter-mem-fraction.sh /path/to/sglang/repo
#
# Default target is /sgl-workspace/sglang (the container's sglang checkout).
# The host edit repo at ~/PR/sglang must NOT be patched — commits originate
# from there and the patch would pollute pushes.
#
# Exit codes:
#   0 = applied OR already applied (safe to re-run)
#   2 = block found but format differs (upstream changed, inspect manually)
#   3 = target file not found
set -euo pipefail

REPO="${1:-/sgl-workspace/sglang}"
TARGET="$REPO/python/sglang/srt/server_args.py"

if [[ ! -f "$TARGET" ]]; then
  echo "ERROR: $TARGET not found" >&2
  exit 3
fi

python3 - "$TARGET" <<'PY'
import sys, pathlib

p = pathlib.Path(sys.argv[1])
src = p.read_text()

block = (
    '        # AMD platforms backends\n'
    '        if self.attention_backend == "aiter":\n'
    '            if model_config.context_len > 8192:\n'
    '                self.mem_fraction_static *= 0.85\n'
    '\n'
)

if block in src:
    p.write_text(src.replace(block, ''))
    print(f"[no-aiter-mem-fraction] removed block from {p}")
elif 'self.mem_fraction_static *= 0.85' in src:
    print(
        "[no-aiter-mem-fraction] WARNING: '*= 0.85' found but surrounding "
        "block format differs from expected. Upstream likely changed; "
        "please inspect manually.",
        file=sys.stderr,
    )
    sys.exit(2)
else:
    print(f"[no-aiter-mem-fraction] already removed in {p} (no-op)")
PY
