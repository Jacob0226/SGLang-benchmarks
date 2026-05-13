#!/usr/bin/env bash
# Robust sequential runner for cascade_dsr1.sh: L2 + L3_file under v3 tag.
# Detaches from terminal so SIGHUP doesn't kill it.
#
# Designed to run inside jacchang_HiCache container as a fully-detached daemon:
#   docker exec jacchang_HiCache bash -c \
#     "setsid nohup bash /home/jacchang/SGLang-benchmarks/tools/chain_cascade_l1_l2_l3.sh \
#        > /home/jacchang/SGLang-benchmarks/results/.../chain.log 2>&1 < /dev/null &"
#
# L1 already ran under tag MI355X_15rounds_0512_v3 (got 11/15 rounds before
# container was rm'd). This chain only does L2 + L3_file under the SAME tag
# so all three modes land in the same parent dir for the comparison plot.

set -uo pipefail

DOCKER="rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507"
TAG="MI355X_15rounds_0512_v3"
SCRIPT_DIR=/home/jacchang/SGLang-benchmarks
SCRIPT="$SCRIPT_DIR/cascade_dsr1.sh"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

run_mode() {
  local mode="$1"
  echo
  echo "[$(ts)] ============================================================"
  echo "[$(ts)] === Launching cascade_dsr1.sh --tag $TAG --cache-mode $mode ==="
  echo "[$(ts)] ============================================================"
  cd "$SCRIPT_DIR"
  bash "$SCRIPT" --tag "$TAG" --cache-mode "$mode" --docker "$DOCKER"
  local rc=$?
  echo "[$(ts)] $mode finished, rc=$rc"
  # Belt-and-suspenders cleanup between runs (cascade_dsr1.sh has its own trap
  # but if anything escapes, kill it; sleep 30s so HBM fully releases).
  pkill -9 -f sglang.launch_server 2>/dev/null || true
  sleep 30
  return $rc
}

echo "[$(ts)] === chain_cascade_l1_l2_l3.sh started ==="
echo "[$(ts)] tag=$TAG  docker=$DOCKER"
echo "[$(ts)] sequence: L2 -> L3_file (L1 already done under same tag)"

# Sanity: kill any stale sglang server before starting
pkill -9 -f sglang.launch_server 2>/dev/null || true
sleep 5

run_mode L2       ; L2_RC=$?
run_mode L3_file  ; L3_RC=$?

echo
echo "[$(ts)] === chain_cascade_l1_l2_l3.sh DONE ==="
echo "[$(ts)] L2 rc=$L2_RC, L3_file rc=$L3_RC"
