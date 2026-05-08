#!/usr/bin/env bash
# cascade_bench.sh — drive HiCache.sh with parameters tuned to make the
# L1 → L2 → L3 cache cascade visible across rounds, MI355X vs B200 style.
#
# Workload sizing (from tools/calc_kv_per_token.py for DSR1-0528 + FP8 KV,
# 35 KB / token / rank, MLA-replicated cache):
#
#   per-rank occupancy at end of round k = N × k × R × 35 KB
#
#   N = 300 clients, R = 4096 tokens/req, num_rounds = 10
#   → at round k the cache holds 41 GB × k per rank
#
#   Cascade events with --hicache-size 192:
#     round 2: B200 L1 (79 GB)  fills  → starts using L2
#     round 4: MI355X L1 (160)  fills  → starts using L2
#     round 7: B200 L1+L2 (271) fills  → starts using L3
#     round 9: MI355X L1+L2 (352) fills → starts using L3
#
# So the same parameters expose:
#   - Both platforms' L1 capacity advantage at rounds 1-3
#   - B200's smaller HBM (= smaller L1) hurts first
#   - B200's smaller DRAM (= smaller L2 cap) hurts again at rounds 7-8
#   - MI355X enjoys the cascade headroom from larger HBM AND larger DRAM
#
# Usage:
#   On the MI355X box:
#     ./cascade_bench.sh --tag MI355X --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507
#   On the B200 box:
#     ./cascade_bench.sh --tag B200   --docker lmsysorg/sglang:v0.5.9-cu130
#   Then locally:
#     python3 plot_cascade.py --tags MI355X B200 --hicache-size 192 \
#         --out cascade.png \
#         --title "DSR1-0528 cascade: MI355X (288 GB HBM, 3 TB DRAM) vs B200 (192 GB, 2 TB)"
#
# All flags after --tag/--docker are forwarded to HiCache.sh, so you can
# override individual cascade parameters if needed:
#   ./cascade_bench.sh --tag MI355X_aggressive --num-clients 400 ...

set -euo pipefail

TAG=""
DOCKER="untagged-docker"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --tag)    TAG="$2"; shift 2;;
    --docker) DOCKER="$2"; shift 2;;
    -h|--help) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \?//' | head -n -1 ; exit 0;;
    *) EXTRA_ARGS+=("$1"); shift 1;;
  esac
done

if [ -z "$TAG" ]; then
  echo "ERROR: --tag is required (e.g. MI355X or B200)" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Cascade-tuned parameters. See header comment for the math. Only the
# hicache_file path is run because the cascade visualization compares
# MI355X vs B200 *on the same full L1+L2+L3 stack* — which layer gets
# hit at each round is read off the per-round TTFT shape (L1 ~50 ms,
# L2 ~few-hundred ms, L3 ~sec). Pass --cache-modes-extra "no_radix
# radix hicache" if you want the Mooncake-style 4-line figure too.
CACHE_MODE="L3_file"
exec "$SCRIPT_DIR/HiCache.sh" \
  --cache-mode "$CACHE_MODE" \
  --tag "$TAG" \
  --docker "$DOCKER" \
  --num-clients     300 \
  --num-rounds      10 \
  --request-length  4096 \
  --output-length   1 \
  --max-parallel    8 \
  --request-rate    32 \
  --hicache-size    192 \
  --hicache-size-sweep "" \
  "${EXTRA_ARGS[@]}"
