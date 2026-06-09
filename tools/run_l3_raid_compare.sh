#!/usr/bin/env bash
# Compare HiCache L3 on the boot drive (/tmp overlay) vs the 7-drive RAID0 (/raid),
# both with the O_DIRECT file backend. Run INSIDE the container (the one started
# with -v /raid:/raid).
#
#   bash tools/run_l3_raid_compare.sh
#
# Env overrides: DOCKER, MODEL, CLIENTS, ROUNDS, L1, L2, PAGE, TAGBASE
set -euo pipefail
cd "$(dirname "$0")/.."

DOCKER=${DOCKER:-rocm/sgl-dev:v0.5.12.post1-rocm720-mi35x-20260601}
MODEL=${MODEL:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
CLIENTS=${CLIENTS:-30}
ROUNDS=${ROUNDS:-8}          # >=5 so the working set spills into L3 (see R4+)
L1=${L1:-10}
L2=${L2:-20}
PAGE=${PAGE:-64}
TAGBASE=${TAGBASE:-0608_L3cmp}

echo "==== 1. check /raid is mounted & writable in this container ===="
if ! touch /raid/_wtest 2>/dev/null; then
  echo "FATAL: /raid not writable here. Start the container with  -v /raid:/raid"; exit 1
fi
rm -f /raid/_wtest
df -h /raid | tail -1

echo "==== 2. (re)apply O_DIRECT patch ===="
python3 tools/apply_odirect_l3.py || true
python3 -c 'import sglang.srt.mem_cache.hicache_storage as h; assert "O_DIRECT" in open(h.__file__).read(), "O_DIRECT patch missing"; print("O_DIRECT patch active")'

run_one () {  # $1 = base dir, $2 = tag suffix
  local base="$1" sfx="$2"
  echo "==== run: L3 on $base  (tag ${TAGBASE}_${sfx}) ===="
  L3_BASE_DIR="$base" ./cascade_dsr1_lite.sh \
    --tag "${TAGBASE}_${sfx}" --docker "$DOCKER" --model "$MODEL" \
    --cache-modes "L3_file" --L1-size "$L1" --hicache-size "$L2" \
    --num-rounds "$ROUNDS" --num-clients "$CLIENTS" --page-size "$PAGE"
}

kill_server () { pkill -9 -f sglang 2>/dev/null || true; sleep 10; }

echo "==== 3. RAID0 run ===="
kill_server
run_one /raid raid
echo "==== 4. /tmp (single boot drive) run ===="
kill_server
run_one /tmp  tmp
kill_server

echo
echo "==== per-round comparison (R4+ is where L3 matters) ===="
for sfx in raid tmp; do
  f=$(ls -t results/*/DeepSeek-R1-0528/bench-${TAGBASE}_${sfx}/L3file_L2_size_${L2}/bench_multiturn.log 2>/dev/null | head -1)
  echo "----- $sfx : $f -----"
  grep -E "Round [0-9]:" "$f" 2>/dev/null || echo "  (no log)"
done
