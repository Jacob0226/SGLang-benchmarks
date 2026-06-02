#!/usr/bin/env bash
# cascade_dsr1_lite.sh — DSR1-0528 cascade benchmark on ROCm aiter.
# Slim version of cascade_dsr1.sh. No NSA/GQA/MXFP4/EAGLE/CUDA dispatch.

set -euo pipefail
ulimit -n 65535

# ============================== Defaults ==============================
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TAG=""
DOCKER="untagged-docker"
TP_SIZE=8
HOST="localhost"
PORT=30000
CACHE_MODE="L3_file"
CACHE_MODES=""
HICACHE_SIZE=auto
HOST_HEADROOM_GB=400
NUM_CLIENTS=300
NUM_ROUNDS=10
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
CUDA_GRAPH_MAX_BS=0
# chunked_prefill_size = max_prefill_tokens = 32768 (DSR1 known-good for the
# HiCache cascade; matches cascade_dsr1.sh). Larger windows (the old
# "InferenceX-aligned" 131072/196608) cause two problems with page_size=64 +
# cache reuse:
#   (1) GSM8K accuracy collapse on ROCm 7.2 images (rocm720): 0.94 -> ~0.25
#       when combined with cache reuse (L1/L2/L3). ROCm 7.0 (rocm700) is NOT
#       affected. Bug needs: rocm720 + ps64 + cache + large chunked/context.
#   (2) activation OOM at high client counts (HSA_STATUS_ERROR_OUT_OF_RESOURCES).
# With prefix cache, per-round extend is only ~request_length (~4096), so a
# 32768 window still batches ~8 requests/forward — no throughput loss for the
# cascade. Override with --chunked-prefill-size / --max-prefill-tokens for
# cold-start throughput experiments.
# NOTE: context_length stays at model native (163840) by default. If you hit
# the ps64+cache accuracy bug on a rocm720 image, also pass --context-length
# 65536 (the cascade workload maxes ~40-60K anyway), or use a rocm700 image.
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
MEM_FRACTION_STATIC=0.8
MEM_FRACTION_EXPLICIT=false   # flipped true when --mem-fraction-static is passed
L1_SIZE=""                    # GB/rank; when set, mem-fraction-static is auto-derived
                              # via compute_profile_params.py (mutually exclusive
                              # with --mem-fraction-static)
# PAGE_SIZE default = 64 to match the L3_file prefetch granularity that the
# file backend was tuned for. ps=1 (sglang auto for aiter MLA legacy) is
# correct/safe but cripples L3 prefetch efficiency: in our 30-client / 8-round
# / L1=10 / L2=20 cascade, ps=1 capped round-8 hit rate at 20% with 261k
# prefetched tokens, vs ps=64 reaching 50% hit / 1.37M tokens prefetched
# (round 8 TTFT 7.44s -> 4.60s).
#
# IMPORTANT: ps=64 requires HiCachePatch fix #2 applied to the sglang
# checkout, otherwise cuda_graph_kv_indices is undersized by 64x and the
# server gets a "Memory access fault by GPU node-N" the moment any
# longer-context decode hits the cuda graph (PR sgl-project/sglang#25556
# fix #2 was force-pushed away during review; see HiCachePatch/README.md
# and pr25556-explained.md).
#
# To force the safe-without-patch ps=1 path, pass --page-size 1.
# CONTEXT_LENGTH default stays empty: sglang uses model native (163840 for
# DSR1).
PAGE_SIZE=64
CONTEXT_LENGTH=""
CONTEXT_LENGTH_EXPLICIT=false
# AITER fp8 prefill attention toggle (AMD only, gated by gfx95). Tristate:
#   ""    -> let the existing env var win, falling back to 0 (script-historical
#            default, OFF / bf16 prefill).
#   "0"   -> force OFF (bf16 prefill).
#   "1"   -> force ON  (fp8 prefill, matches InferenceX default for gfx95).
# CLI: --aiter-fp8-prefill-attn N  (N in 0|1)
AITER_FP8_PREFILL_ATTN=""
# L3 prefetch threshold (tokens). After local L1+L2 match, sglang queries L3
# for the next continuous matching span; a prefetch is only triggered if the
# L3-hit length >= this threshold. Default empty = use sglang code default
# (256). Configured via --hicache-storage-backend-extra-config JSON, only
# meaningful for cache_mode in {L3_file}.
PREFETCH_THRESHOLD=""
HICACHE_WRITE_POLICY="write_through"
# Layout × io backend compatibility matrix (server_args.py:3108-3125
# silently rewrites incompatible pairs, so we pin the recommended one).
#
#   layout              kernel (rec.)   direct
#   ----------------    -------------   ------
#   layer_first         OK              OK
#   page_first          OK              -> forced to page_first_direct
#   page_first_direct   -> forced direct OK
#
# We want page-first organisation (better for L3 file backend, which
# reads / writes whole pages) AND kernel io backend (GPU-assisted KV
# transfer kernels, recommended in SGLang docs over plain cudaMemcpy).
# That intersection is `page_first + kernel`, NOT `page_first_direct
# + kernel` (which silently downgrades io to direct).
HICACHE_MEM_LAYOUT="page_first"
HICACHE_IO_BACKEND="kernel"
GSM8K_PRECHECK="true"
GSM8K_NUM_QUESTIONS=1200
GSM8K_PARALLEL=1200
WAIT_FOR_SERVER_SEC=1500
# Scheduler watchdog timeout (sec). Empty = sglang default (300). Bump this
# for profiling runs: torch.profiler capturing a large prefill can stall a
# forward pass past 300s and crash the server (SIGQUIT). CLI: --watchdog-timeout
WATCHDOG_TIMEOUT=""
ATTENTION_BACKEND=""   # "" = auto-detect by vendor: NV->trtllm_mla, AMD->aiter
OUTPUT_DIR_OVERRIDE=""

ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)               MODEL_PATH="$2"; shift 2;;
    --tag)                 TAG="$2"; shift 2;;
    --docker)              DOCKER="$2"; shift 2;;
    --cache-mode)          CACHE_MODE="$2"; shift 2;;
    --cache-modes)         CACHE_MODES="$2"; shift 2;;
    --tp)                  TP_SIZE="$2"; shift 2;;
    --port)                PORT="$2"; shift 2;;
    --hicache-size)        HICACHE_SIZE="$2"; shift 2;;
    --host-headroom-gb)    HOST_HEADROOM_GB="$2"; shift 2;;
    --num-clients)         NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)          NUM_ROUNDS="$2"; shift 2;;
    --request-length)      REQUEST_LENGTH="$2"; shift 2;;
    --max-parallel)        MAX_PARALLEL="$2"; shift 2;;
    --request-rate)        REQUEST_RATE="$2"; shift 2;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; MEM_FRACTION_EXPLICIT=true; shift 2;;
    --L1-size)             L1_SIZE="$2"; shift 2;;
    --page-size)           PAGE_SIZE="$2"; shift 2;;
    --context-length)      CONTEXT_LENGTH="$2"; CONTEXT_LENGTH_EXPLICIT=true; shift 2;;
    --hicache-mem-layout)  HICACHE_MEM_LAYOUT="$2"; shift 2;;
    --hicache-io-backend)  HICACHE_IO_BACKEND="$2"; shift 2;;
    --aiter-fp8-prefill-attn)
                           AITER_FP8_PREFILL_ATTN="$2"; shift 2;;
    --chunked-prefill-size)
                           CHUNKED_PREFILL_SIZE="$2"; shift 2;;
    --max-prefill-tokens)  MAX_PREFILL_TOKENS="$2"; shift 2;;
    --prefetch-threshold)  PREFETCH_THRESHOLD="$2"; shift 2;;
    --attention-backend)   ATTENTION_BACKEND="$2"; shift 2;;
    --watchdog-timeout)    WATCHDOG_TIMEOUT="$2"; shift 2;;
    --output-dir)          OUTPUT_DIR_OVERRIDE="$2"; shift 2;;
    --gsm8k-num-questions) GSM8K_NUM_QUESTIONS="$2"; shift 2;;
    --no-gsm8k-precheck)   GSM8K_PRECHECK="false"; shift 1;;
    -h|--help)
      cat <<EOF
Usage: $0 --tag TAG --docker DOCKER [--cache-mode MODE | --cache-modes 'MODE1 MODE2 ...']
Modes: none | L1 | L2 | L3_file
Output:
  --output-dir DIR                 override default results/<docker>/<model>/bench-<tag>
                                   root. Per-mode subdirs are still created
                                   under this directory.

Memory sizing (pick one):
  --mem-fraction-static F        explicit fraction (default 0.85)
  --L1-size N                    GB/rank for L1 (GPU radix); auto-derives
                                 --mem-fraction-static via
                                 compute_profile_params.py. Requires
                                 --hicache-size when cache-mode is L2/L3_file.
                                 Mutually exclusive with --mem-fraction-static.
EOF
      exit 0
      ;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

[ -z "$TAG" ] && { echo "ERROR: --tag required" >&2; exit 1; }
[ ! -d "$MODEL_PATH" ] && { echo "ERROR: bad model path: $MODEL_PATH" >&2; exit 1; }
case "$CACHE_MODE" in
  none|L1|L2|L3_file) ;;
  *) echo "ERROR: bad --cache-mode: $CACHE_MODE (none|L1|L2|L3_file)" >&2; exit 1;;
esac
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# ============================== Chain dispatcher ==============================
# --cache-modes "none L1 L2 L3_file" → re-exec self per mode, pkill+sleep between.
if [ -n "$CACHE_MODES" ]; then
  # Pre-create base log dir so we can capture chain dispatcher stdout to it.
  # Mirrors the single-mode BASE_LOG_DIR computation below (line 219); kept
  # in sync manually because we need it before any child runs.
  if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
    CHAIN_BASE_LOG_DIR="$OUTPUT_DIR_OVERRIDE"
  else
    CHAIN_DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
    CHAIN_BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$CHAIN_DOCKER_FILENAME/${MODEL_NAME}/bench-${TAG}"
  fi
  mkdir -p "$CHAIN_BASE_LOG_DIR"
  CHAIN_LOG="$CHAIN_BASE_LOG_DIR/chain.log"
  : > "$CHAIN_LOG"
  exec > >(tee -a "$CHAIN_LOG") 2>&1

  FORWARD_ARGS=()
  i=0
  while [ "$i" -lt "${#ORIG_ARGS[@]}" ]; do
    case "${ORIG_ARGS[$i]}" in
      --cache-modes) i=$((i + 2)) ;;
      *) FORWARD_ARGS+=("${ORIG_ARGS[$i]}"); i=$((i + 1)) ;;
    esac
  done
  for MODE in $CACHE_MODES; do
    echo ""
    echo ">>> ============================================================"
    echo ">>> chain: starting cache_mode=${MODE}  (t=$(date +%H:%M:%S))"
    echo ">>> ============================================================"
    if ! "$0" --cache-mode "$MODE" "${FORWARD_ARGS[@]}"; then
      rc=$?
      echo ">>> chain: cache_mode=${MODE} FAILED with exit code ${rc}; continuing to next mode"
    fi
    pkill -9 sglang 2>/dev/null || true
    sleep 10
  done
  echo ">>> chain done (${CACHE_MODES})  (t=$(date +%H:%M:%S))"

  # Cross-mode summary: combine all per-mode bench_multiturn.jsonl +
  # cache_tiers.csv into a single $BASE_LOG_DIR/cascade_summary.csv. Re-
  # derive BASE_LOG_DIR here because the chain dispatcher exits before
  # the single-mode path computes it.
  CHAIN_MODEL_NAME=$(basename "${MODEL_PATH%/}")
  if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
    CHAIN_BASE_LOG_DIR="$OUTPUT_DIR_OVERRIDE"
  else
    CHAIN_DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
    CHAIN_BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$CHAIN_DOCKER_FILENAME/${CHAIN_MODEL_NAME}/bench-${TAG}"
  fi
  CHAIN_SUMMARIZER="$(dirname "$(readlink -f "$0")")/summarize_cascade.py"
  if [ -f "$CHAIN_SUMMARIZER" ] && [ -d "$CHAIN_BASE_LOG_DIR" ]; then
    python3 "$CHAIN_SUMMARIZER" cross_mode "$CHAIN_BASE_LOG_DIR" || true
  fi
  exit 0
fi

# ============================== Auto-derive mem-fraction-static ==============================
# When --L1-size is given, run compute_profile_params.py to get the
# mem-fraction-static that exactly fits weights/rank + L1 + buffer on
# the local GPU. Lets you run identical L1 sizes across MI355X/B200
# without hand-computing the fraction (HBM/rank differs per platform).
#
# Placed after the chain dispatcher: in chain mode the parent exits
# above without ever using MEM_FRACTION_STATIC, while each re-exec'd
# child hits this block exactly once for its own --cache-mode.
if [ -n "$L1_SIZE" ]; then
  if [ "$MEM_FRACTION_EXPLICIT" = true ]; then
    echo "ERROR: pass either --L1-size (auto mem-fraction) OR --mem-fraction-static, not both" >&2
    exit 1
  fi
  # Helper validates L2 >= L1 and uses L2 only for PROFILE_TARGET math
  # (which we don't consume here). For none/L1 modes L2 is irrelevant,
  # so we alias L2 = L1 just to pass validation. For L2/L3_file modes
  # we need a real --hicache-size.
  HELPER_L2="$L1_SIZE"
  if [ "$CACHE_MODE" = "L2" ] || [ "$CACHE_MODE" = "L3_file" ]; then
    if [ "$HICACHE_SIZE" = "auto" ]; then
      echo "ERROR: --L1-size with --cache-mode=$CACHE_MODE also requires explicit --hicache-size (no 'auto')" >&2
      exit 1
    fi
    HELPER_L2="$HICACHE_SIZE"
  fi
  HELPER="$(dirname "$(readlink -f "$0")")/compute_profile_params.py"
  [ -f "$HELPER" ] || { echo "ERROR: $HELPER not found" >&2; exit 1; }
  echo ">>> deriving --mem-fraction-static from --L1-size=${L1_SIZE} (L2-placeholder=${HELPER_L2})"
  PARAMS=$(python3 "$HELPER" \
    --model "$MODEL_PATH" \
    --tp "$TP_SIZE" \
    --L1-size "$L1_SIZE" \
    --L2-size "$HELPER_L2" \
    --num-clients "$NUM_CLIENTS" \
    --request-length "$REQUEST_LENGTH" 2>&1) || { echo "$PARAMS" >&2; exit 1; }
  echo "$PARAMS" | grep -E '^(WARN|ERROR|INFO)' >&2 || true
  # Whitelist only the vars we actually consume below. Helper also prints
  # NUM_ROUNDS=..., which would silently shadow the user's --num-rounds.
  # See bench_meta.json mismatch postmortem (cascade-FairCompare_0520).
  ALLOW='^(MEM_FRACTION_STATIC|PROFILE_TARGET_ROUND_1IDX|WEIGHTS_GB_PER_RANK|HBM_GB_PER_RANK)='
  eval "$(echo "$PARAMS" | grep -E "$ALLOW")"
  echo "    weights/rank=${WEIGHTS_GB_PER_RANK}GB  HBM/rank=${HBM_GB_PER_RANK}GB"
  echo "    derived: --mem-fraction-static=${MEM_FRACTION_STATIC}"
fi

# ============================== Auto-size hicache ==============================
if [ "$CACHE_MODE" = "none" ] || [ "$CACHE_MODE" = "L1" ]; then
  HICACHE_SIZE=0
elif [ "$HICACHE_SIZE" = "auto" ]; then
  sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  # Wait for MemAvailable to settle (prior pinned-host pool may still be releasing).
  prev=0
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    cur=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
    if [ "$prev" -gt 0 ] && [ $((cur - prev)) -le 5 ] && [ $((prev - cur)) -le 5 ]; then
      echo ">>> MemAvail settled at ${cur} GB"
      break
    fi
    prev=$cur
    sleep 5
  done
  MEM_AVAIL_GB=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
  USABLE_GB=$(( MEM_AVAIL_GB - HOST_HEADROOM_GB ))
  [ "$USABLE_GB" -le 0 ] && { echo "ERROR: only ${MEM_AVAIL_GB} GB avail, need >${HOST_HEADROOM_GB} headroom" >&2; exit 1; }
  PER_RANK=$(( USABLE_GB / TP_SIZE / 32 * 32 ))
  [ "$PER_RANK" -lt 32 ] && PER_RANK=32
  [ "$PER_RANK" -gt 512 ] && PER_RANK=512
  HICACHE_SIZE="$PER_RANK"
  echo ">>> auto hicache-size: ${HICACHE_SIZE} GB/rank (total $(( HICACHE_SIZE * TP_SIZE )) GB)"
fi

# ============================== Host snapshot helpers ==============================
# One-shot CPU / DRAM / NVMe / GPU snapshot, ported verbatim from
# cascade_dsr1.sh. Captured per cache-mode into $LOG_DIR/host_info.log so
# every lite run records the box's CPU model/speed, DRAM speed, and the
# NVMe (L3 backing disk) PCIe bandwidth ceiling alongside its results.
#
# PCIe link speed string → GB/s per lane (single direction, post-encoding):
# Gen1 0.25, Gen2 0.5, Gen3 0.985, Gen4 1.969, Gen5 3.938, Gen6 7.877.
# Returns 0 for unknown speeds.
pcie_lane_gbps() {
  case "$1" in
    "2.5 GT/s PCIe"|"2.5 GT/s")   echo "0.250" ;;
    "5.0 GT/s PCIe"|"5.0 GT/s")   echo "0.500" ;;
    "8.0 GT/s PCIe"|"8.0 GT/s")   echo "0.985" ;;
    "16.0 GT/s PCIe"|"16.0 GT/s") echo "1.969" ;;
    "32.0 GT/s PCIe"|"32.0 GT/s") echo "3.938" ;;
    "64.0 GT/s PCIe"|"64.0 GT/s") echo "7.877" ;;
    *)                            echo "0"     ;;
  esac
}

# collect_host_info prints a one-shot CPU / DRAM / NVMe / GPU snapshot of
# the box to stdout. Caller decides whether to tee it to a log file. Uses
# TAG and HICACHE_SIZE (with sensible fallbacks) only for the predicted
# L3 file-backend path string.
collect_host_info() {
  local tag_for_path="${TAG:-host_info_only}"
  local size_for_path="${HICACHE_SIZE:-N}"
  echo "=== cascade_dsr1_lite.sh host snapshot @ $(date '+%F %T %Z') ==="
  echo "--- lscpu ---"
  lscpu | grep -E "Architecture|Vendor|Model name|CPU\(s\)|Socket|Core|Thread|NUMA|^CPU max MHz|^CPU min MHz"
  echo "--- /proc/meminfo ---"
  grep -E "^MemTotal:|^MemAvailable:|^MemFree:|^Cached:" /proc/meminfo
  echo "--- NUMA per-node DRAM ---"
  for n in /sys/devices/system/node/node[0-9]*; do
    [ -d "$n" ] || continue
    local nid mem cpus
    nid=$(basename "$n" | sed 's/node//')
    mem=$(awk '/MemTotal/{print int($4/1024/1024)" GB"}' "$n/meminfo")
    cpus=$(cat "$n/cpulist" 2>/dev/null)
    printf "  node %s: DRAM=%s, CPUs=%s\n" "$nid" "$mem" "$cpus"
  done

  # DIMM-level detail (model, rated + configured speed, manufacturer). Needs
  # dmidecode + /sys/firmware/dmi/tables (root in container is typical).
  # Auto-install dmidecode on debian/ubuntu bases since the binary is tiny
  # and the bench image rarely ships with it; silently no-op if apt fails.
  echo "--- DRAM DIMMs (dmidecode -t memory) ---"
  if ! command -v dmidecode >/dev/null 2>&1 \
       && command -v apt-get >/dev/null 2>&1 && [ "$(id -u)" -eq 0 ]; then
    echo "  (auto-installing dmidecode...)"
    apt-get -qq update >/dev/null 2>&1 || true
    apt-get -qq install -y dmidecode >/dev/null 2>&1 || true
  fi
  if command -v dmidecode >/dev/null 2>&1 && [ -r /sys/firmware/dmi/tables/DMI ]; then
    dmidecode -t memory 2>/dev/null | awk '
      # stem_of strips the trailing-digit suffix from a Locator string.
      # Useful for Intel-style "CPU0_DIMM_A0" / "CPU0_DIMM_A1" where the
      # trailing digit is the slot index WITHIN a channel — stripping
      # collapses 2DPC slot pairs back to the channel. AMD-style "A1",
      # "A2", ..., "A12" packs the channel ID into that trailing digit
      # instead, so this same stripping over-collapses; we disambiguate
      # in END by checking the populated/stem ratio.
      function stem_of(s,   c) { c = s; sub(/[0-9]+$/, "", c); return c }
      # parse_mts pulls the numeric MT/s value out of "5600 MT/s" /
      # "4400 MT/s" strings; returns 0 for "Unknown" / empty.
      function parse_mts(s,   v) {
        v = s
        if (v !~ /MT\/s/) return 0
        sub(/[[:space:]]*MT\/s.*/, "", v)
        gsub(/[^0-9.]/, "", v)
        return v + 0
      }
      BEGIN { populated=0; total_gb=0 }
      /^Physical Memory Array$/ { in_arr=1; max_cap=""; num_dev=""; next }
      in_arr && /^[[:space:]]*Maximum Capacity:/   { sub(/^[[:space:]]*Maximum Capacity: /,""); max_cap=$0 }
      in_arr && /^[[:space:]]*Number Of Devices:/  { sub(/^[[:space:]]*Number Of Devices: /,""); num_dev=$0 }
      in_arr && /^$/ {
        if (max_cap!="") printf "  array max=%s, slots=%s\n", max_cap, num_dev
        in_arr=0
      }
      /^Memory Device$/ {
        in_md=1; size=""; type=""; speed=""; cfg=""; mfr=""; part=""; loc=""; next
      }
      in_md && /^[[:space:]]*Size:/                       { sub(/^[[:space:]]*Size: /,""); size=$0 }
      in_md && /^[[:space:]]*Type: /                      { sub(/^[[:space:]]*Type: /,""); type=$0 }
      in_md && /^[[:space:]]*Speed:/ && !/Configured/      { sub(/^[[:space:]]*Speed: /,""); speed=$0 }
      in_md && /^[[:space:]]*Configured Memory Speed:/     { sub(/^[[:space:]]*Configured Memory Speed: /,""); cfg=$0 }
      in_md && /^[[:space:]]*Manufacturer:/                { sub(/^[[:space:]]*Manufacturer: /,""); mfr=$0 }
      in_md && /^[[:space:]]*Part Number:/                 { sub(/^[[:space:]]*Part Number: /,""); gsub(/[[:space:]]+$/,"",$0); part=$0 }
      in_md && /^[[:space:]]*Locator:/ && !/Bank Locator/  { sub(/^[[:space:]]*Locator: /,""); loc=$0 }
      in_md && /^$/ {
        if (size != "" && size !~ /No Module/) {
          printf "  %-14s %-9s %-5s rated=%-10s cfg=%-10s %s %s\n",
                 loc, size, type, speed, cfg, mfr, part
          populated += 1
          if (size ~ /GB$/)      { gb=size; sub(/[[:space:]]*GB.*/,"",gb); total_gb += gb + 0 }
          else if (size ~ /MB$/) { mb=size; sub(/[[:space:]]*MB.*/,"",mb); total_gb += (mb+0)/1024 }
          # Record BOTH the full locator and the trailing-digit-stripped
          # stem; END uses populated/n_stem ratio to pick between Intel
          # 2DPC (stem count = channels) and AMD 1DPC (full count =
          # channels). Also stash the most recent configured / rated
          # MT/s values for the bandwidth calc.
          full_locs[loc] = 1
          stem_locs[stem_of(loc)] = 1
          cmts = parse_mts(cfg);   if (cmts > 0) last_cmts = cmts
          rmts = parse_mts(speed); if (rmts > 0) last_rmts = rmts
        }
        in_md=0
      }
      END {
        if (!populated) exit
        printf "  populated DIMMs: %d, total %d GB\n", populated, total_gb
        # DRAM peak BW = MT/s x 8 bytes/transfer x num_channels.
        # MT/s is megatransfers/sec (DDR transfers twice per clock, so e.g.
        # 4400 MT/s -> 2200 MHz). DDR5 channel width is 64-bit = 8 bytes,
        # giving 35.2 GB/s/ch at 4400 MT/s or 48.0 GB/s/ch at 6000 MT/s.
        # We prefer "Configured Memory Speed" (the actual running speed
        # set by BIOS; can be downclocked vs the DIMM SPD) and fall back
        # to "Speed" (the rated max) only when cfg is Unknown.
        n_full = 0; for (k in full_locs) n_full++
        n_stem = 0; for (k in stem_locs) n_stem++
        # Pick channel count based on locator format. Intel boards label
        # slots "CPU0_DIMM_A0" / "CPU0_DIMM_A1" — trailing digit is the
        # SLOT index, so stripping it collapses 2DPC pairs to one channel
        # per stem (n_ch = n_stem). AMD EPYC boards label slots "A1",
        # "A2", ..., "A12", "B1", ..., "B12" — trailing digit is the
        # CHANNEL ID, so each populated locator is already one channel
        # (n_ch = n_full). Disambiguate by populated/n_stem ratio: 1 or
        # 2 = plausible Intel 1DPC/2DPC; anything larger means stem
        # stripping over-collapsed (e.g. AMD A1..A12 -> stem "A") and we
        # use the full count instead.
        ratio = (n_stem > 0 ? int((populated / n_stem) + 0.5) : 0)
        if (ratio == 1 || ratio == 2) {
          n_ch = n_stem
        } else {
          n_ch = n_full
        }
        dpc = (n_ch > 0 ? int((populated / n_ch) + 0.5) : 1)
        use_mts = (last_cmts > 0 ? last_cmts : last_rmts)
        src     = (last_cmts > 0 ? "configured" : "rated (cfg unavailable)")
        if (n_ch == 0 || use_mts == 0) exit
        per_ch_gbs = use_mts * 8 / 1000
        agg_gbs    = per_ch_gbs * n_ch
        printf "  channels populated: %d (DPC=%d) @ %s %d MT/s\n",
               n_ch, dpc, src, use_mts
        printf "  -> DRAM peak BW: %.1f GB/s/ch x %d ch = %.0f GB/s aggregate (this node)\n",
               per_ch_gbs, n_ch, agg_gbs
        if (last_cmts > 0 && last_rmts > 0 && last_cmts < last_rmts) {
          rated_per_ch = last_rmts * 8 / 1000
          rated_agg    = rated_per_ch * n_ch
          dpc_note = (dpc == 2 ? " -- typical for 2DPC" : "")
          printf "    (rated %d MT/s would be %.0f GB/s; cfg downclocked %.0f%%%s)\n",
                 last_rmts, rated_agg, 100*(last_rmts-last_cmts)/last_rmts, dpc_note
        }
      }'
  else
    echo "  (dmidecode unavailable -- install with: apt-get update && apt-get install -y dmidecode)"
  fi

  # NVMe drives via sysfs (works without the `nvme` userspace tool). We list
  # every controller, its model, firmware, total size, current PCIe link
  # speed × width, and the resulting theoretical max bandwidth. The host
  # may have multiple drives on different PCIe generations (e.g. Samsung
  # negotiated at Gen3 next to KIOXIA at Gen5 in this lab), and the L3 file
  # backend's actual disk throughput is capped by whichever drive backs the
  # docker overlay / mount it lands on — see the L3-path section below.
  echo "--- NVMe drives (/sys/class/nvme) ---"
  printf "  %-7s %-34s %-10s %-7s %-28s %s\n" "name" "model" "firmware" "size" "PCIe cur / max" "~GB/s cur/max"
  local c name model fw size cls clw mls mlw cur_lane max_lane bw_cur bw_max
  for c in /sys/class/nvme/nvme*; do
    [ -d "$c" ] || continue
    name=$(basename "$c")
    model=$(cat "$c/model" 2>/dev/null | xargs)
    fw=$(cat "$c/firmware_rev" 2>/dev/null | xargs)
    size=$(lsblk -dn -o SIZE "/dev/${name}n1" 2>/dev/null | head -1 | xargs)
    cls=$(cat "$c/device/current_link_speed" 2>/dev/null)
    clw=$(cat "$c/device/current_link_width" 2>/dev/null)
    mls=$(cat "$c/device/max_link_speed"     2>/dev/null)
    mlw=$(cat "$c/device/max_link_width"     2>/dev/null)
    cur_lane=$(pcie_lane_gbps "$cls")
    max_lane=$(pcie_lane_gbps "$mls")
    bw_cur=$(awk -v l="$cur_lane" -v w="${clw:-0}" 'BEGIN{printf "%.1f", l*w}')
    bw_max=$(awk -v l="$max_lane" -v w="${mlw:-0}" 'BEGIN{printf "%.1f", l*w}')
    printf "  %-7s %-34s %-10s %-7s %-28s %s / %s\n" \
           "$name" "$model" "$fw" "$size" "${cls:-?} x${clw:-?} / ${mls:-?} x${mlw:-?}" \
           "$bw_cur" "$bw_max"
  done

  # L3 file backend lands at /tmp inside the container per the script's
  # HICACHE_FILE_STORE_DIR formula. Inside docker /tmp is on the overlay,
  # so df shows "overlay" rather than the underlying NVMe — readers should
  # cross-reference the drives listed above to figure out which physical
  # disk the host's /var/lib/docker actually sits on.
  echo "--- L3 file backend path & backing fs ---"
  local l3_pred detect
  l3_pred="/tmp/cascade_dsr1_l3_${tag_for_path}_${size_for_path}"
  detect="$l3_pred"; [ ! -e "$detect" ] && detect="/tmp"
  echo "  expected L3 dir: $l3_pred"
  df -h "$detect" 2>/dev/null | tail -1 \
    | awk '{printf "  mount=%-12s fs=%-10s size=%s used=%s avail=%s\n", $6, $1, $2, $3, $4}'
  echo "  note: container /tmp is on the docker overlay; the backing NVMe is whichever"
  echo "        drive holds /var/lib/docker on the host (correlate with the NVMe list)."

  echo "--- GPU info ---"
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showid 2>&1 | grep "Device Name" | head -1
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name --format=csv,noheader | head -1
  fi

  # Per-GPU PCIe link spec — the HiCache L2 / L3 -> GPU upload ceiling.
  # Each rank pulls its KV-cache shard via its own PCIe link, so aggregate
  # upload BW is sum(per-link cur BW) across all 3D controllers. We walk
  # /sys/bus/pci by class 0x030200 (3D controller) which catches both
  # NVIDIA and AMD compute GPUs and skips the management VGA at 0x030000.
  # GPU friendly name is best-effort: nvidia-smi first, fall back to a
  # short lspci description, else vendor:device IDs.
  echo "--- GPU PCIe links (/sys/bus/pci, class 0x030200) ---"
  printf "  %-13s %-15s %-30s %s\n" "bus_id" "name" "PCIe cur / max" "~GB/s cur/max"
  local gpu_name_map
  gpu_name_map=$(mktemp 2>/dev/null || echo "/tmp/gpu_name_map.$$")
  : > "$gpu_name_map"
  if command -v nvidia-smi >/dev/null 2>&1; then
    # nvidia-smi BDF is "00000000:1B:00.0"; normalize to sysfs format
    # "0000:1b:00.0" (lowercase, 4-char domain).
    nvidia-smi --query-gpu=pci.bus_id,name --format=csv,noheader 2>/dev/null \
      | awk -F, '{
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1)
          gsub(/^[[:space:]]+|[[:space:]]+$/, "", $2)
          bdf = tolower($1); sub(/^[0-9a-f]{4}/, "", bdf)
          printf "%s\t%s\n", bdf, $2
        }' > "$gpu_name_map"
  fi
  local n_gpu=0 total_bw_cur=0 total_bw_max=0
  local dev bdf name cls_str cw mls_str mw cur_lane max_lane bw_cur bw_max
  for dev in /sys/bus/pci/devices/*; do
    [ -d "$dev" ] || continue
    [ "$(cat "$dev/class" 2>/dev/null)" = "0x030200" ] || continue
    bdf=$(basename "$dev")
    name=$(awk -F'\t' -v b="$bdf" '$1 == b {print $2; exit}' "$gpu_name_map")
    if [ -z "$name" ] && command -v lspci >/dev/null 2>&1; then
      name=$(lspci -s "$bdf" 2>/dev/null | sed 's/.*: //; s/ (rev.*//' | head -c 30)
    fi
    [ -z "$name" ] && name="(unknown)"
    cls_str=$(cat "$dev/current_link_speed" 2>/dev/null)
    cw=$(cat "$dev/current_link_width"      2>/dev/null)
    mls_str=$(cat "$dev/max_link_speed"     2>/dev/null)
    mw=$(cat "$dev/max_link_width"          2>/dev/null)
    cur_lane=$(pcie_lane_gbps "$cls_str")
    max_lane=$(pcie_lane_gbps "$mls_str")
    bw_cur=$(awk -v l="$cur_lane" -v w="${cw:-0}" 'BEGIN{printf "%.1f", l*w}')
    bw_max=$(awk -v l="$max_lane" -v w="${mw:-0}" 'BEGIN{printf "%.1f", l*w}')
    printf "  %-13s %-15s %-30s %s / %s\n" \
           "$bdf" "$name" \
           "${cls_str:-?} x${cw:-?} / ${mls_str:-?} x${mw:-?}" \
           "$bw_cur" "$bw_max"
    total_bw_cur=$(awk -v t="$total_bw_cur" -v b="$bw_cur" 'BEGIN{printf "%.1f", t+b}')
    total_bw_max=$(awk -v t="$total_bw_max" -v b="$bw_max" 'BEGIN{printf "%.1f", t+b}')
    n_gpu=$((n_gpu+1))
  done
  rm -f "$gpu_name_map"
  if [ "$n_gpu" -gt 0 ]; then
    printf "  -> aggregate GPU PCIe upload BW: %.0f GB/s cur / %.0f GB/s max across %d GPUs\n" \
           "$total_bw_cur" "$total_bw_max" "$n_gpu"
    echo "     (HiCache L2 / L3 -> GPU upload ceiling; one PCIe link per TP rank)"
  fi
}

# ============================== Output dir + meta ==============================
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
if [ -n "$OUTPUT_DIR_OVERRIDE" ]; then
  BASE_LOG_DIR="$OUTPUT_DIR_OVERRIDE"
else
  BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}/bench-${TAG}"
fi
case "$CACHE_MODE" in
  none|L1)  LOG_DIR="${BASE_LOG_DIR}/${CACHE_MODE}" ;;
  L2)       LOG_DIR="${BASE_LOG_DIR}/L2_size_${HICACHE_SIZE}" ;;
  # L3_file mode also uses HICACHE_SIZE as its L2 (host) pool size, so we
  # encode it as "L3file_L2_size_<N>" to disambiguate from a plain L2 run.
  L3_file)  LOG_DIR="${BASE_LOG_DIR}/L3file_L2_size_${HICACHE_SIZE}" ;;
esac
mkdir -p "$LOG_DIR"

{
  printf '%s' "$0"
  for a in "${ORIG_ARGS[@]}"; do
    printf ' %q' "$a"
  done
  printf '\n'
} > "$LOG_DIR/cmdline.txt"

META_HICACHE=$([ "$CACHE_MODE" = "none" ] || [ "$CACHE_MODE" = "L1" ] && echo "null" || echo "$HICACHE_SIZE")
# Record the fp8-prefill resolution: explicit CLI flag wins, then env var,
# then script default 0. Captured before the AMD env block runs (so this
# only reflects intent; the actual exported value is decided there).
META_FP8_PREFILL=$([ -n "$AITER_FP8_PREFILL_ATTN" ] && echo "$AITER_FP8_PREFILL_ATTN" || echo "${SGLANG_AITER_FP8_PREFILL_ATTN:-0}")
cat > "$LOG_DIR/bench_meta.json" <<EOF
{
  "cache_mode": "$CACHE_MODE",
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "page_size": $PAGE_SIZE,
  "tp_size": $TP_SIZE,
  "hicache_size_gb": $META_HICACHE,
  "hicache_write_policy": "$HICACHE_WRITE_POLICY",
  "hicache_mem_layout": "$HICACHE_MEM_LAYOUT",
  "hicache_io_backend": "$HICACHE_IO_BACKEND",
  "aiter_fp8_prefill_attn": $META_FP8_PREFILL,
  "chunked_prefill_size": $CHUNKED_PREFILL_SIZE,
  "max_prefill_tokens": $MAX_PREFILL_TOKENS,
  "prefetch_threshold": $([ -n "$PREFETCH_THRESHOLD" ] && echo "$PREFETCH_THRESHOLD" || echo "null"),
  "mem_fraction_static": $MEM_FRACTION_STATIC,
  "num_clients": $NUM_CLIENTS,
  "num_rounds": $NUM_ROUNDS,
  "request_length": $REQUEST_LENGTH,
  "max_parallel": $MAX_PARALLEL,
  "request_rate": $REQUEST_RATE,
  "tag": "$TAG",
  "docker": "$DOCKER",
  "gsm8k_precheck_accuracy": null
}
EOF

# Host hardware snapshot (CPU model/speed, DRAM speed, NVMe/L3-disk PCIe BW
# ceiling, GPU PCIe links). One per cache-mode subdir, mirroring bench_meta.json.
collect_host_info | tee "$LOG_DIR/host_info.log" >/dev/null

# ============================== L3 file store + disk check ==============================
HICACHE_FILE_STORE_DIR=""
if [[ "$CACHE_MODE" == L3_* ]]; then
  HICACHE_FILE_STORE_DIR="/tmp/cascade_dsr1_l3_${TAG}_${HICACHE_SIZE}"
  rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null
  mkdir -p "$HICACHE_FILE_STORE_DIR"
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_FILE_STORE_DIR"

  L2_TOTAL_GB=$(( HICACHE_SIZE * TP_SIZE ))
  L3_FREE_GB=$(df -BG --output=avail "$HICACHE_FILE_STORE_DIR" | tail -1 | tr -dc '0-9')
  echo ">>> L3 disk: free=${L3_FREE_GB} GB, L2 total=${L2_TOTAL_GB} GB"
  if [ "${L3_FREE_GB:-0}" -le "$L2_TOTAL_GB" ]; then
    echo "ERROR: L3 free (${L3_FREE_GB} GB) <= L2 total (${L2_TOTAL_GB} GB)" >&2
    exit 1
  fi
fi

# ============================== Trap ==============================
trap '
  rm -rf "${HICACHE_FILE_STORE_DIR:-}" 2>/dev/null
  [ -n "${CACHE_MONITOR_PID:-}" ] && kill "${CACHE_MONITOR_PID}" 2>/dev/null
  pkill -9 sglang 2>/dev/null || true
  sleep 10
  true' EXIT

# ============================== Platform detection ==============================
# attention_backend / moe_runner_backend / env vars are vendor-specific.
# We auto-detect once and gate every NV-only or AMD-only flag below.
VENDOR="unknown"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  VENDOR="nvidia"
fi
if command -v rocm-smi >/dev/null 2>&1 && rocm-smi --showid >/dev/null 2>&1; then
  # Pick AMD only when nvidia-smi isn't there OR ROCM_PATH is set (i.e.
  # we're inside an ROCm container). Same heuristic as the profile scripts.
  if [ "$VENDOR" = "unknown" ] || [ -n "${ROCM_PATH:-}" ]; then
    VENDOR="amd"
  fi
fi
echo ">>> platform: ${VENDOR}"

if [ -z "$ATTENTION_BACKEND" ]; then
  case "$VENDOR" in
    nvidia) ATTENTION_BACKEND="trtllm_mla";;   # MLA-aware NV backend (DSR1)
    amd)    ATTENTION_BACKEND="aiter";;        # AMD ROCm aiter
    *)      ATTENTION_BACKEND="aiter";;        # legacy default
  esac
fi

# ============================== Server cmd ==============================
export PYTHONUNBUFFERED=1
export SAFETENSORS_FAST_GPU=1
if [ "$VENDOR" = "amd" ]; then
  # Aligned to InferenceX dsr1_fp8_mi355x.sh:
  #   SGLANG_USE_AITER=1                       aiter prefill/decode path
  #   RCCL_MSCCL_ENABLE=0                      pin RCCL path (no MSCCL)
  #   ROCM_QUICK_REDUCE_QUANTIZATION=INT4      quick-allreduce quantized to INT4
  #                                            (was NONE; InferenceX uses INT4
  #                                            with no accuracy regression)
  #   SGLANG_AITER_FP8_PREFILL_ATTN=0          bf16 prefill kernel
  #                                            (gfx95 default is True; InferenceX
  #                                            relies on it but on this hardware
  #                                            bf16 and fp8 prefill are within
  #                                            noise -- see 8combinations sweep)
  export SGLANG_USE_AITER=1
  export RCCL_MSCCL_ENABLE=0
  export ROCM_QUICK_REDUCE_QUANTIZATION=INT4
  # fp8 prefill toggle, in priority order:
  #   1. --aiter-fp8-prefill-attn N      (CLI flag, highest)
  #   2. caller-exported SGLANG_AITER_FP8_PREFILL_ATTN
  #   3. fall back to "0" (OFF, bf16 prefill — script-historical default
  #      that matches the OLD 0527 reference baseline)
  # NOTE: sglang's own code-default for this env var is "True" (fp8 ON).
  # Because we always export here, that code-default never kicks in unless
  # the caller deliberately unsets the env var AND removes this export.
  if [ -n "$AITER_FP8_PREFILL_ATTN" ]; then
    export SGLANG_AITER_FP8_PREFILL_ATTN="$AITER_FP8_PREFILL_ATTN"
  else
    export SGLANG_AITER_FP8_PREFILL_ATTN=${SGLANG_AITER_FP8_PREFILL_ATTN:-0}
  fi
fi

SERVER_CMD=(
  python3 -u -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --tp "$TP_SIZE"
    --host "$HOST" --port "$PORT"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --enable-metrics
    --trust-remote-code
    --kv-cache-dtype fp8_e4m3
    --page-size "$PAGE_SIZE"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    --attention-backend "$ATTENTION_BACKEND"
)
# --context-length is opt-in (sglang uses model native by default).
[ "$CONTEXT_LENGTH_EXPLICIT" = true ] && SERVER_CMD+=(--context-length "$CONTEXT_LENGTH")
# Opt-in longer scheduler watchdog (sglang default 300s). Needed for profiling.
[ -n "$WATCHDOG_TIMEOUT" ] && SERVER_CMD+=(--watchdog-timeout "$WATCHDOG_TIMEOUT")
if [ "$VENDOR" = "nvidia" ]; then
  # Matches the previously-validated B200 cascade (May-12 run): use
  # FlashInfer's TRT-LLM kernels for MoE + fused allreduce.
  SERVER_CMD+=(
    --moe-runner-backend flashinfer_trtllm
    --enable-flashinfer-allreduce-fusion
  )
fi
# Only pass --cuda-graph-max-bs if user explicitly overrode (>0).
[ "$CUDA_GRAPH_MAX_BS" -gt 0 ] && SERVER_CMD+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")

case "$CACHE_MODE" in
  none) SERVER_CMD+=(--disable-radix-cache);;
  L1)   :;;
  L2)
    # page_first_direct: page-contiguous in L2 (zero-copy L2<->L3) while keeping
    # same-layer tokens grouped within a page for aggregated L2->GPU transfers.
    # Requires sgl-kernel with PR #10339+ on ROCm (>= v0.5.4); v0.5.11 covers both
    # MI355X (rocm720 image) and B200 (cu130 image).
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-io-backend "$HICACHE_IO_BACKEND"
      --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
    );;
  L3_file)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      --hicache-io-backend "$HICACHE_IO_BACKEND"
      --hicache-mem-layout "$HICACHE_MEM_LAYOUT"
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
      --hicache-storage-backend file
      --hicache-storage-prefetch-policy best_effort
    )
    # Override the L3 prefetch trigger threshold (sglang code default 256
    # tokens). Empty = leave default. Translated to the JSON extra-config
    # the storage backend pops in hiradix_cache.py:587.
    if [ -n "$PREFETCH_THRESHOLD" ]; then
      SERVER_CMD+=(--hicache-storage-backend-extra-config \
        "{\"prefetch_threshold\":$PREFETCH_THRESHOLD}")
    fi
    ;;
esac

# ============================== Launch + pre-check + health ==============================
SERVER_LOG="$LOG_DIR/server.log"
echo ">>> launching SGLang cache_mode=${CACHE_MODE}"
echo "${SERVER_CMD[*]}" | tee "$SERVER_LOG"

if pgrep sglang >/dev/null 2>&1; then
  echo "ERROR: sglang already running. 'pkill -9 sglang && sleep 10' first." >&2
  pgrep -a sglang >&2
  exit 1
fi
if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
  echo "ERROR: port ${PORT} already serves /health 200" >&2
  exit 1
fi

"${SERVER_CMD[@]}" 2>&1 | tee -a "$SERVER_LOG" &
SERVER_BG_PID=$!

deadline=$(( $(date +%s) + WAIT_FOR_SERVER_SEC ))
echo ">>> wait /health up to ${WAIT_FOR_SERVER_SEC}s..."
while [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; do
  if ! kill -0 "$SERVER_BG_PID" 2>/dev/null; then
    echo "ERROR: server pipeline died. tail of $SERVER_LOG:" >&2
    tail -n 50 "$SERVER_LOG" >&2
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERROR: /health timeout, see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
echo ">>> server ready (pid=$SERVER_BG_PID)"

# ============================== Warmup ==============================
echo ">>> warmup"
# random-range-ratio 0.8 matches InferenceX (input lengths uniform in
# [0.8*N, N]); previous 1.0 was a fixed length that didn't reflect the
# distribution InferenceX uses.
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --model "$MODEL_PATH" --dataset-name random \
  --random-input 1024 --random-output 128 --random-range-ratio 0.8 \
  --max-concurrency 4 --num-prompt 8 --output-file /dev/null \
  2>&1 | tee "$LOG_DIR/warmup.log"

# ============================== GSM8K precheck ==============================
if [ "$GSM8K_PRECHECK" = "true" ]; then
  GSM8K_SCRIPT=""
  for c in /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
           "$HOME/work-space/sglang/benchmark/gsm8k/bench_sglang.py"; do
    [ -f "$c" ] && GSM8K_SCRIPT="$c" && break
  done
  if [ -n "$GSM8K_SCRIPT" ]; then
    echo ">>> GSM8K precheck (${GSM8K_NUM_QUESTIONS} q, parallel=${GSM8K_PARALLEL})"
    ( cd "$LOG_DIR" && python3 "$GSM8K_SCRIPT" \
        --host "$HOST" --port "$PORT" \
        --num-questions "$GSM8K_NUM_QUESTIONS" \
        --parallel "$GSM8K_PARALLEL" \
        --result-file "$LOG_DIR/Accuracy_GSM8K.jsonl" \
        2>&1 | tee "$LOG_DIR/Accuracy_GSM8K.log" ) || true
    ACC=$(grep -oP '^Accuracy:\s+\K[0-9.]+' "$LOG_DIR/Accuracy_GSM8K.log" | tail -1 || true)
    echo ">>> GSM8K accuracy: ${ACC:-NA}"
    python3 - "$LOG_DIR/bench_meta.json" "${ACC:-null}" <<'PY'
import json, sys
p, acc = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["gsm8k_precheck_accuracy"] = float(acc) if acc != "null" else None
json.dump(d, open(p, "w"), indent=2)
PY
  else
    echo ">>> WARNING: bench_sglang.py not found, skipping GSM8K"
  fi
fi

# ============================== Bench multiturn ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
[ -z "$BENCH_SCRIPT" ] && { echo "ERROR: bench_multiturn.py not found" >&2; exit 1; }

# Flush radix tree + OS page cache so cascade starts cold.
curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
sleep 2
sync
echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
sleep 2

# Optional cache_monitor sidecar (per-tier hit rate per round → cache_tiers.csv).
CACHE_MONITOR_SCRIPT="$(dirname "$(readlink -f "$0")")/cache_monitor.py"
CACHE_MONITOR_PID=""
if [ -f "$CACHE_MONITOR_SCRIPT" ]; then
  python3 "$CACHE_MONITOR_SCRIPT" \
      --url "http://${HOST}:${PORT}/metrics" \
      --interval 10 \
      --num-clients "$NUM_CLIENTS" \
      --num-rounds "$NUM_ROUNDS" \
      --csv "$LOG_DIR/cache_tiers.csv" \
      > "$LOG_DIR/cache_monitor.log" 2>&1 &
  CACHE_MONITOR_PID=$!
fi

echo ">>> bench multiturn N=${NUM_CLIENTS} R=${REQUEST_LENGTH} rounds=${NUM_ROUNDS}"
python3 "$BENCH_SCRIPT" \
  --host "$HOST" --port "$PORT" \
  --model-path "$MODEL_PATH" \
  --num-clients "$NUM_CLIENTS" \
  --num-rounds "$NUM_ROUNDS" \
  --request-length "$REQUEST_LENGTH" \
  --output-length "$OUTPUT_LENGTH" \
  --max-parallel "$MAX_PARALLEL" \
  --request-rate "$REQUEST_RATE" \
  --ready-queue-policy random \
  --log-file "$LOG_DIR/bench_multiturn.jsonl" \
  --tag "${MODEL_NAME}-${TAG}" \
  --disable-random-sample \
  --disable-auto-run \
  --enable-round-barrier \
  2>&1 | tee "$LOG_DIR/bench_multiturn.log"

if [ -n "$CACHE_MONITOR_PID" ] && kill -0 "$CACHE_MONITOR_PID" 2>/dev/null; then
  kill "$CACHE_MONITOR_PID" 2>/dev/null || true
  wait "$CACHE_MONITOR_PID" 2>/dev/null || true
fi

# ============================== Per-mode round summary ==============================
# Joins this mode's bench_multiturn.jsonl (per-round TTFT + hit rate)
# with cache_tiers.csv (per-round L2 fill + L3 prefetch BW p50/p99) into
# a single round_summary.csv. Tolerates missing cache_tiers.csv (older
# runs) and missing/partial bench_multiturn.jsonl (failed runs).
SUMMARIZER="$(dirname "$(readlink -f "$0")")/summarize_cascade.py"
if [ -f "$SUMMARIZER" ]; then
  python3 "$SUMMARIZER" per_mode "$LOG_DIR" || true
fi

# ============================== Cleanup ==============================
echo ">>> stopping server"
pkill -9 sglang 2>/dev/null || true
sleep 10
[ -n "$HICACHE_FILE_STORE_DIR" ] && [ -d "$HICACHE_FILE_STORE_DIR" ] && rm -rf "$HICACHE_FILE_STORE_DIR"

echo ">>> done. results in: $LOG_DIR"
