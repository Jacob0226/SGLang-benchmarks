#!/usr/bin/env bash
# Wait for the currently-running L1 cascade_dsr1.sh to finish, then
# sequentially run L2 + L3_file with the same tag.
#
# Designed to run inside jacchang_HiCache container as a background daemon:
#   docker exec -d jacchang_HiCache bash -c \
#     "bash /home/jacchang/SGLang-benchmarks/tools/chain_cascade_l2_l3.sh \
#        > /home/jacchang/SGLang-benchmarks/results/.../chain_l2_l3.log 2>&1"

set -uo pipefail

DOCKER="rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260507"
TAG="MI355X_15rounds_0512_v3"
SCRIPT_DIR=/home/jacchang/SGLang-benchmarks
SCRIPT="$SCRIPT_DIR/cascade_dsr1.sh"

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "[$(ts)] === chain_cascade_l2_l3.sh started ==="
echo "[$(ts)] Will wait for current L1 (--tag $TAG --cache-mode L1) to finish, then run L2 + L3_file sequentially."

# ---------------- Wait for L1 ----------------
echo "[$(ts)] Polling for running cascade_dsr1.sh L1 process every 30s..."
deadline=$(( $(date +%s) + 7200 ))   # 2hr cap on L1 wait
while :; do
  # Look for any cascade_dsr1.sh process with --cache-mode L1 and our tag
  if pgrep -af "cascade_dsr1.sh.*--tag ${TAG}.*--cache-mode L1" > /dev/null \
     || pgrep -af "cascade_dsr1.sh.*--cache-mode L1.*--tag ${TAG}" > /dev/null; then
    if [ "$(date +%s)" -ge "$deadline" ]; then
      echo "[$(ts)] WARN: L1 still running after 2hr; aborting wait. Will NOT chain L2/L3 to avoid contention."
      exit 1
    fi
    sleep 30
    continue
  fi
  break
done
echo "[$(ts)] L1 process gone. Quick sanity check that no SGLang server is left over..."
sleep 10

# Belt-and-suspenders cleanup of any stale server
if pgrep -af sglang.launch_server > /dev/null; then
  echo "[$(ts)] Stale sglang.launch_server still present; killing"
  pkill -9 -f sglang.launch_server 2>/dev/null || true
  sleep 5
fi

# ---------------- Run L2 ----------------
echo
echo "[$(ts)] === Launching L2 ==="
cd "$SCRIPT_DIR"
bash "$SCRIPT" --tag "$TAG" --cache-mode L2 --docker "$DOCKER"
L2_RC=$?
echo "[$(ts)] L2 finished, rc=$L2_RC"
# Ensure server is dead between runs (script also has trap, but be safe)
pkill -9 -f sglang.launch_server 2>/dev/null || true
sleep 30

# ---------------- Run L3_file ----------------
echo
echo "[$(ts)] === Launching L3_file ==="
cd "$SCRIPT_DIR"
bash "$SCRIPT" --tag "$TAG" --cache-mode L3_file --docker "$DOCKER"
L3_RC=$?
echo "[$(ts)] L3_file finished, rc=$L3_RC"
pkill -9 -f sglang.launch_server 2>/dev/null || true

echo
echo "[$(ts)] === chain_cascade_l2_l3.sh DONE  (L2 rc=$L2_RC, L3_file rc=$L3_RC) ==="
