#!/usr/bin/env bash
# cascade_dsr1.sh — self-contained DSR1-0528 cascade benchmark.
#
# Launch SGLang on DSR1-0528 with a multi-turn workload tuned to walk the
# cache hierarchy round by round, so MI355X vs B200 differences in HBM
# and DRAM capacity show up as TTFT / cache-hit-rate inflection points.
# Same script supports the three cache hierarchy depths via --cache-mode:
#
#   L1       GPU radix cache only (default SGLang, no HiCache flags).
#            No host pool, no disk — baseline for "GPU memory only" runs.
#   L2       L1 + host DRAM pool (--enable-hierarchical-cache + --hicache-size).
#            No external storage backend.
#   L3_file  L1 + L2 + local-file external store at /tmp (NVMe-backed in
#            most docker images). The default; this is the full cascade
#            run that gives you the L1→L2→L3 spike pattern.
#
# Doesn't depend on HiCache.sh or any other helper script — just bash +
# sglang inside the docker image. Writes everything (logs, metadata,
# plot) into one bench folder so a copy of that folder is reproducible.
#
# Workload (hard-coded to land cache-fill events in observable rounds):
#   N=300 clients × R=4096 tokens/req × num_rounds (default 10)
#   per-rank cache occupancy ≈ 41 GB × round  (DSR1 MLA replicated, FP8 KV)
#   write_through to all tiers; L3 store goes under /tmp (auto-cleaned)
#
# All workload defaults below match the original a0367ca cascade_dsr1.sh
# config (the one that produced the 2026-05-08 reference jsonl). Pass
# --num-rounds 15 / --hicache-size N / --chunked-prefill-size N etc. on
# the CLI to override for ablation runs (e.g. extending the cascade past
# the L2 ceiling, sweeping host-pool size, or stress-testing the AMD
# scheduler with longer chunked-prefills).
#
# Override common knobs via flags below. Anything else: edit the
# constants in this file.
#
# GSM8K precheck (ON by default; --no-gsm8k-precheck to opt out): runs a
# 1200-question GSM8K accuracy bench AFTER warmup and BEFORE the cascade
# bench. Log file name + workload match GLM.sh's accuracy_test()
# (--num-questions 1200 --parallel 1200, log -> $LOG_DIR/Accuracy_GSM8K.log)
# so the cascade and GLM benchmark trees produce comparable accuracy
# artifacts. The parsed accuracy is also recorded in bench_meta.json so
# each cascade run carries a model-correctness signature. After the GSM8K
# phase, /flush_cache + drop_caches (best-effort) clean SGLang radix tree
# + OS page cache so the cascade bench still starts cold. Adds ~70-90 sec
# per run on MI355X (parallel=1200 finishes the 1200 questions in ~70 s wall).
#
# Usage on MI355X — preferred: one command via the chain dispatcher
# (re-execs self per mode, with pkill -9 sglang + sleep 10 between
# iterations, GLM.sh-style). Saves you from manually chaining 4 shells:
#   ./cascade_dsr1.sh --tag MI355X_cascade \
#       --cache-modes "none L1 L2 L3_file" \
#       --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260514
#
# Or run a single mode for ad-hoc / re-run scenarios:
#   ./cascade_dsr1.sh --tag MI355X_cascade --cache-mode L3_file \
#       --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260514
#
# Usage on B200 (mirror): same commands with --tag B200_cascade and
#   --docker lmsysorg/sglang:v0.5.9-cu130.
#
# All three modes write under the same parent dir
#   results/<docker>/<MODEL>-cascade-<TAG>/{L1,L2/size_<N>,L3_file/size_<N>}/
# so plot_cascade.py can pick them up by absolute path easily.
#
# Plot once both done (matplotlib needed; runs inside docker if local
# python doesn't have it). Pass each platform's bench_multiturn.jsonl
# explicitly via --MI355X / --B200, and use absolute paths so the output
# PNG lands in a predictable spot regardless of cwd:
#   python3 plot_cascade.py \
#       --Title "DSR1-0528 cascade L3_file: MI355X vs B200" \
#       --MI355X $HOME/SGLang-benchmarks/results/<rocm-docker>/DeepSeek-R1-0528-cascade-MI355X_cascade/L3_file/size_<N>/bench_multiturn.jsonl \
#       --B200   $HOME/SGLang-benchmarks/results/<cuda-docker>/DeepSeek-R1-0528-cascade-B200_cascade/L3_file/size_<N>/bench_multiturn.jsonl \
#       --out    $HOME/SGLang-benchmarks/results/cascade_dsr1.png

set -euo pipefail
set -x
ulimit -n 65535
sh -c 'echo 0 > /proc/sys/kernel/numa_balancing' || true

# ============================== Defaults ==============================
MODEL_PATH=${MODEL_PATH:-/data/huggingface/hub/deepseek-ai/DeepSeek-R1-0528}
TAG=""
DOCKER="untagged-docker"
TP_SIZE=8
HOST="localhost"
PORT="30000"
# --cache-mode picks one point in the cache hierarchy:
#   L1       GPU radix cache only (no --enable-hierarchical-cache)
#   L2       + host DRAM pool      (no external storage)
#   L3_file  + local file backend  (full cascade; the default)
CACHE_MODE="L3_file"
# --hicache-size: per-rank host KV pool in GB. Default "auto" lets each
# platform max out its own DRAM (MI355X ~320 GB/rank on a 3 TB box,
# B200 ~192 GB/rank on a 2 TB box) — same command line on both. Pass an
# explicit number for cross-platform fairness (same host pool size on
# both MI355X and B200), e.g. --hicache-size 192 to reproduce the
# original a0367ca config.
#   "auto"   = pick the largest value that fits in this box's MemAvailable
#              minus host headroom, divided across TP ranks. Use this for
#              "let each platform win on its own DRAM" runs.
#   <number> = explicit per-rank GB (cross-platform fairness — same host
#              pool size on both MI355X and B200).
HICACHE_SIZE=auto
# HOST_HEADROOM_GB: how much DRAM to reserve OUTSIDE the HiCache host pool.
# On a 3 TB MI355X box with TP=8, the auto sizer does:
#     PER_RANK = (MemAvail - HOST_HEADROOM) / 8, aligned down to 32 GB
# i.e. raising headroom by 256 GB drops PER_RANK by 32 GB.
#
# Headroom must absorb (write_through hot path, MI355X DSR1-FP8):
#   ~671 GB DSR1-FP8 weights briefly held in host RAM during model load
#   ~ 80 GB HiCache pinned-memory staging buffers (TP=8, page_first_direct)
#   ~ 50 GB OS page cache for /data/huggingface/...
#   ~ 30 GB SGLang process group anonymous memory (per-rank workers)
#   + safety margin for write_through bursts that briefly double-allocate.
#
# Default 400 GB was chosen after the 200 GB run on jacchang_HiCache pinned
# 2560/3024 GB and slowed L2 prefill 10-200x (cascade_dsr1.sh L2 v3 run,
# 2026-05-12). On a 3 TB box this still gives PER_RANK=320 GB / total 2560
# GB pool, because raising headroom 200->400 only shaved 25 GB/rank pre-
# alignment. Bump higher (e.g. 900) if write_through is starving the host.
HOST_HEADROOM_GB=400
NUM_CLIENTS=300
NUM_ROUNDS=10
REQUEST_LENGTH=4096
OUTPUT_LENGTH=1
MAX_PARALLEL=8
REQUEST_RATE=32
# --cuda-graph-max-bs: passthrough to SGLang's --cuda-graph-max-bs. 0
# (default) means "don't pass it; let SGLang pick its own default
# (~512)". Set explicitly only if you have a memory-constrained reason.
# NOTE: bench_multiturn.py's --max-parallel is a client-side concurrency
# limit, NOT a server-side cap — the server scheduler batches as many
# requests as fit in max-running-requests + chunked-prefill-size, so in
# practice #running-req climbs well above MAX_PARALLEL.
CUDA_GRAPH_MAX_BS=0
# Prefill chunking. Default 32768 matches the original a0367ca config
# (SGLang's own default is 8192). a0c0522 bumped this to 65536 to save
# chunk-switch overhead on the round-11+ 60K-token prompts, but that
# change has been linked to AMD scheduler instability under high
# concurrency — keep at 32768 by default and pass --chunked-prefill-size
# 65536 explicitly for ablation runs.
CHUNKED_PREFILL_SIZE=32768
MAX_PREFILL_TOKENS=32768
# 30 min cap; large host pools (e.g. 320 GB × 8 ranks = 2.56 TB pinned)
# can take 12-15 min just to fault in + register with the GPU driver,
# so 900s was too tight and caused L3_file timeouts on MI355X.
WAIT_FOR_SERVER_SEC=1800
# GSM8K precheck. ON by default so every cascade run carries a model-
# correctness signature in bench_meta.json (paired with the per-run
# Accuracy_GSM8K.log under $LOG_DIR). Adds ~70-90 sec per cascade run;
# pass --no-gsm8k-precheck to opt out when you only care about cache /
# perf numbers. Defaults match GLM.sh's accuracy_test() (1200 / 1200)
# for cross-bench parity. Higher parallel for cascade-style 4096-token
# prompts can crash the AMD scheduler, but GSM8K prompts are short
# (~600 tokens) so parallel=1200 is safe and matches what we already
# use in GLM benchmarks.
GSM8K_PRECHECK="true"
GSM8K_NUM_QUESTIONS=1200
GSM8K_PARALLEL=1200
# --gsm8k-only: exit cleanly after GSM8K precheck (skip the cascade
# multi-turn benchmark). Useful for quick accuracy verification of a
# server config change (~10-15 min vs ~30+).
GSM8K_ONLY="false"

# Hardware-spec-only mode: prints the host_info snapshot (CPU / DIMMs /
# NVMe drives / L3 path) to stdout and exits before validation, server
# launch, or any benchmark. Useful for capturing a box's spec sheet
# without spinning up SGLang. No --model / --tag / --cache-mode needed.
HOST_INFO_ONLY="false"

# --mem-fraction-static: SGLang's --mem-fraction-static. Lower it
# (e.g. 0.7) for smaller models where weights are tiny; raise it
# (e.g. 0.9) only when KV pool pressure is the bottleneck. Affects
# both device KV-pool size and the "host > device" HiCache assertion.
MEM_FRACTION_STATIC=0.85

# --page-size: overrides the model-family-detected PAGE_SIZE. Family
# defaults are page_size=64 across the board (MLA / NSA / GQA); pass
# --page-size 1 here to A/B against the legacy contiguous-KV layout for
# DSR1 / Llama / Qwen runs. Leave empty to use family default.
PAGE_SIZE_OVERRIDE=""

# --no-cache: legacy alias for --cache-mode none. Disables both radix
# cache and HiCache (--disable-radix-cache + no --enable-hierarchical-cache).
NO_CACHE="false"

# --hicache-write-policy: write_through (default) or write_back. Hubert's
# PR #16531 for DSR1-MXFP4 + page_size=64 + HiCache used write_through;
# write_back may be faster for cascade workloads where eviction is rare.
HICACHE_WRITE_POLICY="write_through"

# --hicache-ratio: host pool / device pool ratio (when not using
# explicit --hicache-size). Empty = SGLang default (2). Hubert used 1.
HICACHE_RATIO=""

# --max-running-requests: cap on concurrent requests in scheduler.
# Empty = SGLang default. Hubert used 128 for DSR1-MXFP4 + EAGLE.
MAX_RUNNING_REQUESTS=""

# --enable-eagle + companions: turn on EAGLE speculative decoding.
# Hubert tested DSR1-MXFP4 with steps=3, topk=1, draft_tokens=4 +
# lmsys/DeepSeek-R1-NextN draft model. SGLANG_ENABLE_SPEC_V2=1 is
# auto-set in the ROCm block when --enable-eagle is on.
ENABLE_EAGLE="false"
SPECULATIVE_DRAFT_MODEL=""
SPECULATIVE_NUM_STEPS=3
SPECULATIVE_EAGLE_TOPK=1
SPECULATIVE_NUM_DRAFT_TOKENS=4

# --hubert-preset: apply env vars + flag combinations from Hubert's
# PR #16531 ("[AMD] Fix aiter page-size handling, DeepSeek MLA tuple
# inputs, and HiCache/FA3 decode-backend override"). Verified config
# for DSR1-MXFP4 + aiter + HiCache + page_size=64 → GSM8K 0.942.
# Sets SGLANG_AITER_MLA_PERSIST=1, SGLANG_ROCM_FUSED_DECODE_MLA=1,
# SGLANG_USE_AITER_AR=0, SGLANG_INT4_WEIGHT=0, SGLANG_MOE_PADDING=1,
# SGLANG_SET_CPU_AFFINITY=1, RCCL_MSCCL_ENABLE=0; AITER_MXFP4_MOE_SF=1
# only when model is MXFP4.
HUBERT_PRESET="false"

# --cache-modes (plural): space-separated list of modes to run as a GLM.sh-
# style chain. With this set, the script re-execs itself once per mode
# (each as a separate cascade_dsr1.sh process so all per-mode state is
# fresh) with pkill -9 sglang + sleep 10 between iterations, then exits.
# Without this flag, the script behaves as before (single mode via
# --cache-mode). Example:
#   ./cascade_dsr1.sh --tag PS64 --cache-modes "none L1 L2 L3_file" \
#       --docker rocm/sgl-dev:v0.5.11-rocm720-mi35x-20260514
CACHE_MODES=""

# Save the original argv so the chain dispatcher can re-exec self with the
# user's flags forwarded (minus --cache-modes itself, which would cause an
# infinite recursion).
ORIG_ARGS=("$@")

while [[ $# -gt 0 ]]; do
  case $1 in
    --model)          MODEL_PATH="$2"; shift 2;;
    --tag)            TAG="$2"; shift 2;;
    --docker)         DOCKER="$2"; shift 2;;
    --cache-mode)     CACHE_MODE="$2"; shift 2;;
    --cache-modes)    CACHE_MODES="$2"; shift 2;;
    --tp)             TP_SIZE="$2"; shift 2;;
    --port)           PORT="$2"; shift 2;;
    --hicache-size)   HICACHE_SIZE="$2"; shift 2;;
    --host-headroom-gb) HOST_HEADROOM_GB="$2"; shift 2;;
    --num-clients)    NUM_CLIENTS="$2"; shift 2;;
    --num-rounds)     NUM_ROUNDS="$2"; shift 2;;
    --request-length) REQUEST_LENGTH="$2"; shift 2;;
    --max-parallel)   MAX_PARALLEL="$2"; shift 2;;
    --request-rate)   REQUEST_RATE="$2"; shift 2;;
    --cuda-graph-max-bs)   CUDA_GRAPH_MAX_BS="$2"; shift 2;;
    --chunked-prefill-size) CHUNKED_PREFILL_SIZE="$2"; shift 2;;
    --max-prefill-tokens)   MAX_PREFILL_TOKENS="$2"; shift 2;;
    --gsm8k-precheck) GSM8K_PRECHECK="true"; shift 1;;
    --no-gsm8k-precheck) GSM8K_PRECHECK="false"; shift 1;;
    --gsm8k-only) GSM8K_PRECHECK="true"; GSM8K_ONLY="true"; shift 1;;
    --gsm8k-num-questions) GSM8K_NUM_QUESTIONS="$2"; shift 2;;
    --gsm8k-parallel) GSM8K_PARALLEL="$2"; shift 2;;
    --host-info-only) HOST_INFO_ONLY="true"; shift 1;;
    --mem-fraction-static) MEM_FRACTION_STATIC="$2"; shift 2;;
    --page-size) PAGE_SIZE_OVERRIDE="$2"; shift 2;;
    --no-cache) NO_CACHE="true"; shift 1;;
    --hicache-write-policy) HICACHE_WRITE_POLICY="$2"; shift 2;;
    --hicache-ratio) HICACHE_RATIO="$2"; shift 2;;
    --max-running-requests) MAX_RUNNING_REQUESTS="$2"; shift 2;;
    --enable-eagle) ENABLE_EAGLE="true"; shift 1;;
    --speculative-draft-model-path) SPECULATIVE_DRAFT_MODEL="$2"; shift 2;;
    --speculative-num-steps) SPECULATIVE_NUM_STEPS="$2"; shift 2;;
    --speculative-eagle-topk) SPECULATIVE_EAGLE_TOPK="$2"; shift 2;;
    --speculative-num-draft-tokens) SPECULATIVE_NUM_DRAFT_TOKENS="$2"; shift 2;;
    --hubert-preset) HUBERT_PRESET="true"; shift 1;;
    -h|--help) sed -n '1,/^set -euo pipefail/p' "$0" | sed 's/^# \?//' | head -n -1; exit 0;;
    *) echo "Unknown option: $1" >&2; exit 1;;
  esac
done

# ============================== Chain dispatcher (GLM.sh-style) ==============================
# If --cache-modes was passed (space-separated list), re-exec self once
# per mode with pkill + sleep 10 between iterations. Same idea as the
# `for PROF_MODE in ...; do ...; pkill -9 python; sleep 10; done` loop
# in GLM.sh, except scoped to sglang processes only. This makes manually
# chaining `./cascade_dsr1.sh --cache-mode L1; ./cascade_dsr1.sh ...`
# unnecessary AND eliminates the race where the previous invocation's
# server is still alive when the next one launches (root cause of the
# 2026-05-16 "L3_file curve == L2 curve" bug).
if [ -n "$CACHE_MODES" ]; then
  # Strip --cache-modes and its value from the argv we forward — keeping
  # it would cause infinite self-exec recursion.
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
    echo ">>> cascade_dsr1.sh chain: starting cache_mode=${MODE}"
    echo ">>> ============================================================"
    "$0" --cache-mode "$MODE" "${FORWARD_ARGS[@]}"
    echo ">>> chain: ${MODE} done; killing any leftover sglang + sleeping 10s"
    pkill -9 sglang 2>/dev/null || true
    sleep 10
  done
  echo ">>> chain: all modes complete (${CACHE_MODES})"
  exit 0
fi

# ============================== Host snapshot helpers ==============================
# Defined up-here (before validation) so --host-info-only can call them
# without needing --tag / --model / --cache-mode. The benchmark path also
# uses these once it's done with validation.
#
# PCIe link speed string → GB/s per lane (single direction, post-encoding):
# Gen1 0.25, Gen2 0.5, Gen3 0.985, Gen4 1.969, Gen5 3.938, Gen6 7.877.
# Used to derive each NVMe drive's theoretical max bandwidth ceiling so the
# host_info log makes the SSD speed envelope explicit instead of leaving the
# reader to look up "what is PCIe Gen5 x4". Returns 0 for unknown speeds.
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
# L3 file-backend path string, so --host-info-only callers don't have to
# supply them.
collect_host_info() {
  local tag_for_path="${TAG:-host_info_only}"
  local size_for_path="${HICACHE_SIZE:-N}"
  echo "=== cascade_dsr1.sh host snapshot @ $(date '+%F %T %Z') ==="
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

# --host-info-only: print the snapshot and exit BEFORE any validation,
# auto-sizing, server launch, or benchmark step. Useful for capturing a
# box's spec sheet without spinning up SGLang. No --model / --tag /
# --cache-mode required in this mode. Disable `set -x` first so the
# snapshot stdout is pure spec data (no shell trace noise mixed in) —
# users typically tee this straight into a host_info.log.
if [ "$HOST_INFO_ONLY" = "true" ]; then
  set +x
  collect_host_info
  exit 0
fi

if [ -z "$TAG" ]; then
  echo "ERROR: --tag is required (e.g. MI355X_cascade or B200_cascade)" >&2
  exit 1
fi
if [ ! -d "$MODEL_PATH" ]; then
  echo "ERROR: model path doesn't exist: $MODEL_PATH" >&2
  exit 1
fi
# --no-cache (legacy) overrides CACHE_MODE to "none".
if [ "$NO_CACHE" = "true" ]; then
  CACHE_MODE="none"
fi
case "$CACHE_MODE" in
  none|L1|L2|L3_file) ;;
  *) echo "ERROR: --cache-mode must be one of: none, L1, L2, L3_file (got: '$CACHE_MODE')" >&2; exit 1;;
esac
MODEL_NAME=$(basename "${MODEL_PATH%/}")

# ============================== Model-family detection ==============================
# Server launch args (page_size, reasoning parser, context length, NSA
# backend) depend on the model family. We detect from the model dir
# basename so the same script can run DSR1/DSV3 (MLA), GLM-5/DSV3.2 (NSA),
# Qwen3/Llama/Mistral (GQA) without per-model flag wrangling.
#
# Note: NSA cases must come BEFORE the more generic MLA glob below, or
# DeepSeek-V3.2-Exp matches the wrong arm.
case "$MODEL_NAME" in
  DeepSeek-V3.2*)
    MODEL_FAMILY="NSA"
    REASONING_PARSER_FLAGS=(--reasoning-parser deepseek-r1)
    PAGE_SIZE=64
    CONTEXT_LENGTH=65536
    echo ">>> NSA model (DSV3.2 DSA); routing attention through tilelang" >&2
    ;;
  GLM-5*|glm-5*|GLM-5.1*|glm-5.1*)
    MODEL_FAMILY="NSA"
    # GLM-5/5.1 uses its own reasoning + tool-call parser (matches GLM.sh).
    # Without these, bench_sglang.py GSM8K answer extraction picks up
    # intermediate numbers from un-stripped <think>...</think> →
    # accuracy 0.94 → 0.82.
    REASONING_PARSER_FLAGS=(
      --reasoning-parser glm45
      --tool-call-parser glm47
    )
    PAGE_SIZE=64
    CONTEXT_LENGTH=65536
    echo ">>> NSA model (GLM-5/5.1); reasoning=glm45, tool-call=glm47, attn via tilelang" >&2
    ;;
  DeepSeek-R1*|DeepSeek-V3*|DeepSeek-V2*|Kimi-K2*|deepseek-*|DeepSeek-R1-0528*)
    MODEL_FAMILY="MLA"
    # Drop --reasoning-parser here on purpose: see ROCm-block comment in
    # SERVER_CMD section about parser eating cascade output text.
    REASONING_PARSER_FLAGS=()
    # page_size=64 is now the script-wide default. For MLA on ROCm aiter
    # this combo only works once the FP8-prefill kernel from PR #18528 is
    # disabled (SGLANG_AITER_FP8_PREFILL_ATTN=0, exported below); otherwise
    # GSM8K collapses to ~0.0 on MI355X. Pair with --page-size 1 override
    # if you specifically need to A/B against the legacy contiguous layout.
    PAGE_SIZE=64
    CONTEXT_LENGTH=65536
    ;;
  Qwen3-32B|Qwen3-*|Qwen2.5-*|Qwen2-*)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=64
    CONTEXT_LENGTH=40960
    ;;
  Llama-3.*|Llama3-*|Meta-Llama-3*)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=64
    CONTEXT_LENGTH=65536
    ;;
  Mixtral-*|Mistral-*|mistral-*)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=64
    CONTEXT_LENGTH=32768
    ;;
  *)
    MODEL_FAMILY="GQA"
    REASONING_PARSER_FLAGS=()
    PAGE_SIZE=64
    CONTEXT_LENGTH=32768
    echo ">>> NOTE: unrecognized model name '$MODEL_NAME', defaulting to GQA family" >&2
    ;;
esac

# Apply --page-size override after family detection.
if [ -n "$PAGE_SIZE_OVERRIDE" ]; then
  echo ">>> page_size override: model-family default $PAGE_SIZE → $PAGE_SIZE_OVERRIDE" >&2
  PAGE_SIZE="$PAGE_SIZE_OVERRIDE"
fi

# Detect MXFP4 weight format from model name. AITER_MXFP4_MOE_SF=1 must be
# set on ROCm + MXFP4 to enable the scale-factor MoE kernel path.
IS_MXFP4="false"
case "$MODEL_NAME" in
  *MXFP4*|*mxfp4*) IS_MXFP4="true" ;;
esac

# NSA on ROCm + radix cache: tilelang lacks the page_table_1_flattened
# fixup that flashmla_sparse has on CUDA. With radix cache on, prefix-
# shared KV reads come from wrong physical slots → GSM8K 0.94 → ~0.82.
# flashmla_sparse can't replace it on ROCm (sgl_kernel.flashmla_ops is
# CUDA-only). For accurate runs on NSA + ROCm use --cache-mode none.
CORRECTNESS_AT_RISK="false"
if { [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; } \
   && [ "$MODEL_FAMILY" = "NSA" ] && [ "$CACHE_MODE" != "none" ]; then
  CORRECTNESS_AT_RISK="true"
  echo ">>> WARNING: NSA family on ROCm + cache_mode=$CACHE_MODE is correctness-broken." >&2
  echo "    tilelang NSA prefill on ROCm lacks the prefix-sharing fixup that" >&2
  echo "    flashmla_sparse has on CUDA. GSM8K typically drops 0.94 → ~0.82." >&2
  echo "    For accurate runs use --cache-mode none. This run is throughput-only." >&2
fi

# Auto-clamp NUM_ROUNDS so per-client cumulative input doesn't overflow
# context. Cascade workload appends REQUEST_LENGTH tokens each round, so
# total ≈ REQUEST_LENGTH * NUM_ROUNDS at the last round. Leave 4K slack
# for output + sysprompt.
_total_seq=$(( REQUEST_LENGTH * NUM_ROUNDS + 4096 ))
if [ "$_total_seq" -gt "$CONTEXT_LENGTH" ]; then
  _max_rounds=$(( (CONTEXT_LENGTH - 4096) / REQUEST_LENGTH ))
  echo ">>> WARNING: NUM_ROUNDS=$NUM_ROUNDS * REQUEST_LENGTH=$REQUEST_LENGTH + 4K slack" >&2
  echo "    = $_total_seq tokens exceeds CONTEXT_LENGTH=$CONTEXT_LENGTH for $MODEL_FAMILY" >&2
  echo "    clamping NUM_ROUNDS from $NUM_ROUNDS to $_max_rounds" >&2
  NUM_ROUNDS=$_max_rounds
fi

echo ">>> model family: $MODEL_FAMILY (page_size=$PAGE_SIZE, context_length=$CONTEXT_LENGTH, cache_mode=$CACHE_MODE)"

# ============================== Auto-size hicache ==============================
# L1 doesn't use a host pool at all (just GPU radix cache), so any
# --hicache-size value is meaningless. We skip sizing entirely and set
# HICACHE_SIZE=0 as a sentinel (treated as null in bench_meta.json).
#
# When --hicache-size is "auto" for L2 / L3_file, measure the box's
# actual MemAvailable and size the host KV pool to its safe maximum.
# This is the whole point of the cross-platform comparison: MI355X
# (3 TB DRAM) and B200 (~2 TB DRAM) each get to use their own DRAM
# ceiling, so the cascade reflects true platform capacity. Headroom
# (default 400 GB; see HOST_HEADROOM_GB block above for breakdown)
# covers DSR1-FP8 weights held briefly in host DRAM during model load,
# OS page cache, HiCache pinned-memory staging buffers, NUMA frag slack,
# and the SGLang process group's anonymous memory.
if [ "$CACHE_MODE" = "none" ] || [ "$CACHE_MODE" = "L1" ]; then
  HICACHE_SIZE=0
  echo ">>> CACHE_MODE=$CACHE_MODE: skipping host pool sizing (no HiCache layer)"
elif [ "$HICACHE_SIZE" = "auto" ]; then
  # Wait for any prior run's pinned host pool to actually come back to
  # MemAvailable. After SIGKILL on the previous sglang server the kernel
  # async-releases the cudaMallocHost pages, so /proc/meminfo MemAvailable
  # can lag by 10-30s. Without this wait, chained runs (e.g. L2 → L3 back
  # to back) auto-size L3 against the still-pinned L2 pool and end up with
  # ~32 GB/rank less host pool than L2 had, making cross-mode comparisons
  # unfair. We sync + drop_caches once (clears OS page cache), then sample
  # MemAvail every 5s and break as soon as two consecutive samples are
  # within 5 GB (signal: kernel is done reclaiming for now). Cap at 60s.
  sync
  echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true
  prev=0
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    cur=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
    if [ "$prev" -gt 0 ] && [ $((cur - prev)) -le 5 ] && [ $((prev - cur)) -le 5 ]; then
      echo ">>> MemAvailable settled at ${cur} GB after $((i*5)) s"
      break
    fi
    echo ">>> waiting for MemAvailable to settle (sample $i: ${cur} GB; prev ${prev} GB)"
    prev=$cur
    sleep 5
  done

  MEM_AVAIL_GB=$(awk '/^MemAvailable:/ {print int($2/1024/1024)}' /proc/meminfo)
  USABLE_GB=$(( MEM_AVAIL_GB - HOST_HEADROOM_GB ))
  if [ "$USABLE_GB" -le 0 ]; then
    echo "ERROR: only ${MEM_AVAIL_GB} GB MemAvailable, can't reserve" \
         "${HOST_HEADROOM_GB} GB headroom. Use --host-headroom-gb or" \
         "--hicache-size N." >&2
    exit 1
  fi
  PER_RANK=$(( USABLE_GB / TP_SIZE ))
  PER_RANK=$(( PER_RANK / 32 * 32 ))   # 32 GB align for clean numbers
  [ "$PER_RANK" -lt 32  ] && PER_RANK=32
  [ "$PER_RANK" -gt 512 ] && PER_RANK=512   # diminishing returns past this
  HICACHE_SIZE="$PER_RANK"
  echo ">>> auto --hicache-size: ${HICACHE_SIZE} GB per rank" \
       "(MemAvailable=${MEM_AVAIL_GB} GB, headroom=${HOST_HEADROOM_GB} GB," \
       "TP=${TP_SIZE} ranks; total host pool = $(( HICACHE_SIZE * TP_SIZE )) GB)"
else
  echo ">>> manual --hicache-size: ${HICACHE_SIZE} GB per rank" \
       "(total host pool = $(( HICACHE_SIZE * TP_SIZE )) GB)"
fi

# ============================== Output dir ==============================
# none / L1 have no hicache-size, so dir is just .../<TAG>/<mode>/.
# L2 / L3_file include the size so a later size sweep doesn't overwrite.
DOCKER_FILENAME=$(echo "$DOCKER" | sed 's/\//_/g; s/:/-/g')
BASE_LOG_DIR="$HOME/SGLang-benchmarks/results/$DOCKER_FILENAME/${MODEL_NAME}-cascade-${TAG}"
case "$CACHE_MODE" in
  none|L1) LOG_DIR="${BASE_LOG_DIR}/${CACHE_MODE}" ;;
  L2|L3_file) LOG_DIR="${BASE_LOG_DIR}/${CACHE_MODE}/size_${HICACHE_SIZE}" ;;
esac
mkdir -p "$LOG_DIR"
echo ">>> bench folder: $LOG_DIR"

# ============================== Platform detection ==============================
is_rocm() { [ -e /dev/kfd ] || command -v rocm-smi >/dev/null 2>&1; }

# Detect HBM size for sanity messaging.
get_gpu_hbm_gb() {
  if command -v rocm-smi >/dev/null 2>&1; then
    rocm-smi --showmeminfo vram 2>/dev/null \
      | awk '/VRAM Total Memory/ {print int($NF/1024/1024/1024); exit}'
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
      | head -1 | awk '{print int($1/1024)}'
  else
    echo 0
  fi
}
HBM_GB=$(get_gpu_hbm_gb)
echo ">>> detected HBM: ${HBM_GB} GB per GPU, TP=${TP_SIZE}"

# ============================== NUMA interleave ==============================
NUMA_NODES=$(ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null \
              | sed 's|.*/node||' | sort -n | paste -sd,)
NUMACTL_PREFIX=()
if command -v numactl >/dev/null 2>&1 && [[ "$NUMA_NODES" == *,* ]]; then
  NUMACTL_PREFIX=(numactl --interleave="$NUMA_NODES")
  echo ">>> NUMA interleave: --interleave=${NUMA_NODES}"
fi

# ============================== Host snapshot ==============================
# pcie_lane_gbps() / collect_host_info() are defined up-top so
# --host-info-only can call them before validation. Here in the benchmark
# path we just call collect_host_info and tee it to the run's log dir.
collect_host_info | tee "$LOG_DIR/host_info.log" >/dev/null

# ============================== Bench meta ==============================
# Emit hicache_size_gb as JSON null when cache_mode=L1 (the field is
# meaningless without a host pool), otherwise the integer per-rank GB.
# We inject Python's None (not the JSON literal `null`) since the value
# is interpolated into a Python source heredoc; json.dumps then emits
# the proper JSON null.
if [ "$CACHE_MODE" = "L1" ]; then
  META_HICACHE_VAL="None"
else
  META_HICACHE_VAL="$HICACHE_SIZE"
fi
# GSM8K precheck fields. Initialized with accuracy=None; updated in-place
# after the GSM8K phase actually runs (see update_bench_meta_accuracy).
if [ "$GSM8K_PRECHECK" = "true" ]; then
  META_GSM8K_ENABLED="True"
  META_GSM8K_NQ="$GSM8K_NUM_QUESTIONS"
else
  META_GSM8K_ENABLED="False"
  META_GSM8K_NQ="None"
fi
python3 - "$LOG_DIR/bench_meta.json" <<PY
import json, sys
data = {
    "cache_mode": "$CACHE_MODE",
    "correctness_at_risk": $([ "$CORRECTNESS_AT_RISK" = "true" ] && echo "True" || echo "False"),
    "model_path": "$MODEL_PATH",
    "model_name": "$MODEL_NAME",
    "model_family": "$MODEL_FAMILY",
    "is_mxfp4": $([ "$IS_MXFP4" = "true" ] && echo "True" || echo "False"),
    "hubert_preset": $([ "$HUBERT_PRESET" = "true" ] && echo "True" || echo "False"),
    "hicache_write_policy": "$HICACHE_WRITE_POLICY",
    "hicache_ratio": $([ -n "$HICACHE_RATIO" ] && echo "\"$HICACHE_RATIO\"" || echo "None"),
    "max_running_requests": $([ -n "$MAX_RUNNING_REQUESTS" ] && echo "$MAX_RUNNING_REQUESTS" || echo "None"),
    "enable_eagle": $([ "$ENABLE_EAGLE" = "true" ] && echo "True" || echo "False"),
    "page_size": $PAGE_SIZE,
    "context_length": $CONTEXT_LENGTH,
    "tp_size": $TP_SIZE,
    "kv_cache_dtype": "fp8_e4m3",
    "mem_fraction_static": $MEM_FRACTION_STATIC,
    "host_headroom_gb": $HOST_HEADROOM_GB,
    "hbm_gb": $HBM_GB,
    "device_pool_gb": int($HBM_GB * $MEM_FRACTION_STATIC - 84),
    "hicache_size_gb": $META_HICACHE_VAL,
    "bench_mode": "multiturn",
    "num_clients": $NUM_CLIENTS,
    "num_rounds": $NUM_ROUNDS,
    "request_length": $REQUEST_LENGTH,
    "output_length": $OUTPUT_LENGTH,
    "max_parallel": $MAX_PARALLEL,
    "request_rate": $REQUEST_RATE,
    "enable_round_barrier": True,
    "disable_random_sample": True,
    "gsm8k_precheck_enabled": $META_GSM8K_ENABLED,
    "gsm8k_precheck_num_questions": $META_GSM8K_NQ,
    "gsm8k_precheck_accuracy": None,  # filled in after the GSM8K phase
    "tag": "$TAG",
    "docker": "$DOCKER",
}
with open(sys.argv[1], "w") as f:
    json.dump(data, f, indent=2)
PY

# Helper: update gsm8k_precheck_accuracy in bench_meta.json after GSM8K
# runs. Takes one argument: a Python literal (number or None).
update_bench_meta_accuracy() {
  local acc_pyval="$1"
  python3 - <<PY
import json
p = "$LOG_DIR/bench_meta.json"
with open(p) as f:
    data = json.load(f)
data["gsm8k_precheck_accuracy"] = $acc_pyval
with open(p, "w") as f:
    json.dump(data, f, indent=2)
PY
}

# ============================== L3 file store (in /tmp, container-local) ==============================
# Only set up the on-disk L3 store for L3_* modes — L1 / L2 don't need it.
HICACHE_FILE_STORE_DIR=""
if [[ "$CACHE_MODE" == L3_* ]]; then
  HICACHE_FILE_STORE_DIR="/tmp/cascade_dsr1_l3_${TAG}_${HICACHE_SIZE}"
  rm -rf "$HICACHE_FILE_STORE_DIR" 2>/dev/null
  mkdir -p "$HICACHE_FILE_STORE_DIR"
  export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR="$HICACHE_FILE_STORE_DIR"

  # Sanity check: L3 NVMe free space must be > total L2 host pool size
  # (HICACHE_SIZE per rank × TP_SIZE ranks). If L3 can't even hold a
  # single L2 worth of evictions, the L3_file run will collapse onto the
  # L2 curve (no meaningful extra capacity to spill to) and the cascade
  # plot becomes uninformative. df -BG prints free space in GB chunks
  # rounded down, which is exactly the conservative comparison we want.
  L2_TOTAL_GB=$(( HICACHE_SIZE * TP_SIZE ))
  L3_FREE_GB=$(df -BG --output=avail "$HICACHE_FILE_STORE_DIR" | tail -1 | tr -dc '0-9')
  echo ">>> L3 disk check: free=${L3_FREE_GB} GB at $(dirname "$HICACHE_FILE_STORE_DIR")," \
       "L2 total host pool=${L2_TOTAL_GB} GB (=${HICACHE_SIZE} GB/rank × ${TP_SIZE} ranks)"
  if [ "${L3_FREE_GB:-0}" -le "$L2_TOTAL_GB" ]; then
    echo "ERROR: L3 NVMe free space (${L3_FREE_GB} GB) <= L2 host pool" >&2
    echo "       total (${L2_TOTAL_GB} GB). L3 can't even hold one L2 worth of" >&2
    echo "       evictions — the L3_file curve will look identical to L2 and" >&2
    echo "       the cascade plot will be uninformative. Free up /tmp space" >&2
    echo "       or shrink --hicache-size before re-running." >&2
    exit 1
  fi
fi
# GLM.sh-style cleanup: SIGKILL anything matching "sglang", sleep 10 so
# the HIP/CUDA driver can reap VRAM before the next invocation. Same
# pattern as GLM.sh line 466 (`pkill -9 python; sleep 10`), scoped to
# sglang since this script runs inside a dedicated sglang container.
# Uses :- so none/L1/L2 (HICACHE_FILE_STORE_DIR="") doesn't trip set -u.
trap '
  rm -rf "${HICACHE_FILE_STORE_DIR:-}" 2>/dev/null
  [ -n "${CACHE_MONITOR_PID:-}" ] && kill "${CACHE_MONITOR_PID}" 2>/dev/null
  pkill -9 sglang 2>/dev/null || true
  sleep 10
  true' EXIT

# ============================== Server launch ==============================
SERVER_LOG="$LOG_DIR/server.log"

# Force Python unbuffered stdout/stderr so server.log captures every
# Prefill batch / POST /generate line in real time. Default block-
# buffering loses up to ~8 KB of trailing logs when killed at end of
# run and makes tail -f look frozen during the run.
export PYTHONUNBUFFERED=1
if [ "$CUDA_GRAPH_MAX_BS" -gt 0 ]; then
  echo ">>> cuda-graph-max-bs=${CUDA_GRAPH_MAX_BS} (user override)"
else
  echo ">>> cuda-graph-max-bs: SGLang default (~512)"
fi
echo ">>> chunked-prefill-size=${CHUNKED_PREFILL_SIZE}, max-prefill-tokens=${MAX_PREFILL_TOKENS}"

# Common cmd (everything that doesn't depend on cache mode).
# python3 -u also forces unbuffered output (belt + suspenders with
# PYTHONUNBUFFERED=1 above).
# --page-size and --context-length are model-family dependent, set above.
# REASONING_PARSER_FLAGS expands to () for DSR1/GQA (parser drops answer
# text), to (--reasoning-parser glm45 --tool-call-parser glm47) for GLM-5.
SERVER_CMD=(
  "${NUMACTL_PREFIX[@]}"
  python3 -u -m sglang.launch_server
    --model-path "$MODEL_PATH"
    --tp "$TP_SIZE"
    --host "$HOST" --port "$PORT"
    --mem-fraction-static "$MEM_FRACTION_STATIC"
    --watchdog-timeout 2400
    --enable-metrics
    --enable-cache-report
    --trust-remote-code
    "${REASONING_PARSER_FLAGS[@]}"
    --kv-cache-dtype fp8_e4m3
    --page-size "$PAGE_SIZE"
    --context-length "$CONTEXT_LENGTH"
    --chunked-prefill-size "$CHUNKED_PREFILL_SIZE"
    --max-prefill-tokens "$MAX_PREFILL_TOKENS"
)
# Only pass --cuda-graph-max-bs when user explicitly overrode it
# (CUDA_GRAPH_MAX_BS > 0). Otherwise rely on SGLang's own default (~512).
if [ "$CUDA_GRAPH_MAX_BS" -gt 0 ]; then
  SERVER_CMD+=(--cuda-graph-max-bs "$CUDA_GRAPH_MAX_BS")
fi
# Cache-mode-specific flags. none = --disable-radix-cache (no radix, no
# HiCache); L1 = GPU-only radix; L2 = + host DRAM pool; L3_file = + file.
case "$CACHE_MODE" in
  none)
    SERVER_CMD+=(--disable-radix-cache)
    ;;
  L1)
    : # nothing — default SGLang behavior (GPU-only RadixCache)
    ;;
  L2)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      # --hicache-mem-layout page_first_direct # Docker Jan-10 haven't supported
      --hicache-io-backend kernel
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
    )
    ;;
  L3_file)
    SERVER_CMD+=(
      --enable-hierarchical-cache
      --hicache-size "$HICACHE_SIZE"
      # --hicache-mem-layout page_first_direct # Docker Jan-10 haven't supported
      --hicache-io-backend kernel
      --hicache-write-policy "$HICACHE_WRITE_POLICY"
      --hicache-storage-backend file
      --hicache-storage-prefetch-policy best_effort
    )
    ;;
esac

# Optional CLI overrides (Hubert PR #16531 style knobs).
if [ -n "$HICACHE_RATIO" ]; then
  SERVER_CMD+=(--hicache-ratio "$HICACHE_RATIO")
fi
if [ -n "$MAX_RUNNING_REQUESTS" ]; then
  SERVER_CMD+=(--max-running-requests "$MAX_RUNNING_REQUESTS")
fi
if [ "$ENABLE_EAGLE" = "true" ]; then
  if [ -z "$SPECULATIVE_DRAFT_MODEL" ]; then
    echo "ERROR: --enable-eagle requires --speculative-draft-model-path" >&2
    exit 1
  fi
  SERVER_CMD+=(
    --speculative-algorithm EAGLE
    --speculative-draft-model-path "$SPECULATIVE_DRAFT_MODEL"
    --speculative-num-steps "$SPECULATIVE_NUM_STEPS"
    --speculative-eagle-topk "$SPECULATIVE_EAGLE_TOPK"
    --speculative-num-draft-tokens "$SPECULATIVE_NUM_DRAFT_TOKENS"
  )
  # SGLang spec scheduler v2 (used by Hubert's PR + GLM.sh MTP path).
  export SGLANG_ENABLE_SPEC_V2=1
fi

if is_rocm; then
  # MI355X DSR1-FP8 server config — aligned with InferenceX
  # benchmarks/single_node/dsr1_fp8_mi355x.sh + AMD ROCm 7.0 official doc
  # (https://rocm.docs.amd.com/en/docs-7.0-docker/benchmark-docker/inference-sglang-deepseek-r1-fp8.html).
  #
  # SGLANG_USE_AITER=1                       enable aiter kernels (docker
  #                                          default already sets this; pinned
  #                                          here so behavior is explicit).
  # ROCM_QUICK_REDUCE_QUANTIZATION=NONE      do NOT quantize AllReduce. INT4
  #                                          (the value the SKILL.md notes for
  #                                          GLM-5) is too lossy for DSR1 and
  #                                          tanks GSM8K accuracy from 0.93+
  #                                          to ~0.01 (verified 2026-05-12 in
  #                                          tools/gsm8k_dsr1_minfix_test.sh).
  # Also dropped from SERVER_CMD common args:
  #   --reasoning-parser deepseek-r1   strips <think>...</think> server-side;
  #                            for cascade workload (output_length=1) and
  #                            GSM8K precheck the parser can eat the actual
  #                            answer text. Not needed for these benches.
  #
  # NOTE on --page-size 64: previously avoided because page>1 + aiter MLA
  # produced garbled outputs / HSA faults. Root-cause was the new FP8
  # prefill kernel from PR #18528 (KV layout assumption). With
  # SGLANG_AITER_FP8_PREFILL_ATTN=0 exported below the kernel falls back
  # to the contiguous path and page_size=64 is safe. Family default is
  # now PAGE_SIZE=64; override with --page-size 1 for A/B sanity checks.
  export SAFETENSORS_FAST_GPU=1
  export SGLANG_USE_AITER=1
  export ROCM_QUICK_REDUCE_QUANTIZATION=NONE

  # MXFP4 model: always export the env vars aiter needs to take the
  # scale-factor MoE kernel path. Without AITER_MXFP4_MOE_SF=1 the model
  # either crashes at MoE init or silently produces garbage outputs.
  # SGLANG_DISABLE_FUSED_AR_MXFP4_QUANT=false enables the fused all-reduce
  # path for MXFP4 quant; Hubert's PR #16531 verified config sets both.
  # Independent of --hubert-preset because these are *required* for MXFP4
  # to work at all (not optional perf tweaks like the rest of the preset).
  if [ "$IS_MXFP4" = "true" ]; then
    export AITER_MXFP4_MOE_SF=1
    export SGLANG_DISABLE_FUSED_AR_MXFP4_QUANT=false
    echo ">>> MXFP4 model detected: exported AITER_MXFP4_MOE_SF=1," \
         "SGLANG_DISABLE_FUSED_AR_MXFP4_QUANT=false" >&2
  fi

  # Disable PR #18528 (Fp8 prefill attn kernel integration, merged 2026-02-11)
  # on MI355X. The new mla_prefill_ps_asm_fwd kernel default-enabled by
  # is_gfx95_supported() collapses GSM8K accuracy from 0.94 → 0.02 when
  # --page-size > 1 with radix cache (or HiCache, which forces radix on).
  # Bisect window: sglang d0d387dea (good 0.949) → dcc63dc54 (bad 0.750)
  # = 19 commits, only PR #18528 touched aiter_backend.py. Verified
  # workaround: setting this env var to 0 restores acc to 0.954 on the
  # latest May 13 docker. Track upstream fix; until then keep this off
  # for any DSR1 + page>1 workload.
  # See results/_bisect/ for full per-docker accuracy table.
  export SGLANG_AITER_FP8_PREFILL_ATTN=0

  # --hubert-preset: Hubert PR #16531 ("Fix aiter page-size handling,
  # DeepSeek MLA tuple inputs, and HiCache/FA3 decode-backend override")
  # tested DSR1-MXFP4 + aiter + HiCache + page_size=64 → GSM8K 0.942
  # with the env vars below. Apply the same set so we can reproduce
  # that result on FP8 / MXFP4 DSR1 variants.
  if [ "$HUBERT_PRESET" = "true" ]; then
    export SGLANG_AITER_MLA_PERSIST=1
    export SGLANG_ROCM_FUSED_DECODE_MLA=1
    export SGLANG_USE_AITER_AR=0
    export SGLANG_INT4_WEIGHT=0
    export SGLANG_MOE_PADDING=1
    export SGLANG_SET_CPU_AFFINITY=1
    export RCCL_MSCCL_ENABLE=0
    echo ">>> --hubert-preset: applied Hubert PR #16531 env vars" >&2
    echo "    SGLANG_AITER_MLA_PERSIST=1, SGLANG_ROCM_FUSED_DECODE_MLA=1," >&2
    echo "    SGLANG_USE_AITER_AR=0, SGLANG_INT4_WEIGHT=0," >&2
    echo "    SGLANG_MOE_PADDING=1, SGLANG_SET_CPU_AFFINITY=1," >&2
    echo "    RCCL_MSCCL_ENABLE=0" >&2
    # AITER_MXFP4_MOE_SF=1 + SGLANG_DISABLE_FUSED_AR_MXFP4_QUANT=false are
    # set unconditionally for MXFP4 models in the block above (required,
    # not part of the optional hubert preset).
  fi

  # NSA family (GLM-5/5.1, DSV3.2): route attention through tilelang.
  # On ROCm, aiter's dense MHA fallback path inside _concat_and_cast_mha_k
  # trips a Triton `arange's range must be a power of 2` error when
  # qk_nope_head_dim isn't power-of-2 (GLM-5 ships qk_nope_head_dim=192).
  # tilelang skips that fallback entirely — same workaround as GLM.sh.
  # We also drop --attention-backend aiter so the NSA dispatcher picks
  # the tilelang path.
  if [ "$MODEL_FAMILY" = "NSA" ]; then
    SERVER_CMD+=(
      --nsa-prefill-backend tilelang
      --nsa-decode-backend tilelang
    )
  else
    SERVER_CMD+=(--attention-backend aiter)
  fi
else
  export SGL_ENABLE_JIT_DEEPGEMM=1
  SERVER_CMD+=(
    --attention-backend trtllm_mla
    --moe-runner-backend flashinfer_trtllm
    --enable-flashinfer-allreduce-fusion
  )
fi

echo ">>> launching SGLang (cache_mode=${CACHE_MODE}, hicache-size=${HICACHE_SIZE} GB)"
echo "${SERVER_CMD[*]}" | tee "$SERVER_LOG"

# Pre-launch safety: refuse to start if any sglang process is alive in
# this container, or if port 30000 is already serving /health. Without
# these guards, a leftover server from the previous cascade_dsr1.sh
# invocation (whose EXIT trap missed it for any reason) would (a) make
# our new server OOM during weight load, then (b) /health below would
# still return 200 from the *old* server and we'd silently run the
# entire bench against the wrong cache_mode (root cause of the 2026-05-16
# "L3_file curve == L2 curve" bug — see size_320.invalid_l2_rerun_*).
# With the new --cache-modes chain dispatcher this race is largely gone,
# but keep the check as a belt-and-suspenders safety net for manual
# multi-shell invocations.
if pgrep sglang >/dev/null 2>&1; then
  echo "ERROR: another sglang process is already running in this container." >&2
  echo "       Run 'pkill -9 sglang && sleep 10' first, or wait for the" >&2
  echo "       previous cascade_dsr1.sh invocation to exit." >&2
  pgrep -a sglang >&2
  exit 1
fi
if curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null | grep -q '^200$'; then
  echo "ERROR: port ${PORT} already has something serving /health 200." >&2
  echo "       Refusing to launch — would otherwise silently run bench" >&2
  echo "       against the wrong server." >&2
  exit 1
fi

"${SERVER_CMD[@]}" 2>&1 | tee -a "$SERVER_LOG" &
SERVER_BG_PID=$!

# ============================== Wait for /health ==============================
# Wait for the *launched* server to come up. Cross-check that:
#   1) SERVER_BG_PID is still alive (bg pipeline didn't crash), AND
#   2) /health returns 200.
# Without (1), an OOM at weight-load time would silently let us proceed
# against whatever other server happened to answer on PORT — see the
# 2026-05-16 L3_file post-mortem above.
deadline=$(( $(date +%s) + WAIT_FOR_SERVER_SEC ))
echo ">>> waiting up to ${WAIT_FOR_SERVER_SEC}s for /health 200..."
while [ "$(curl -s -o /dev/null -w '%{http_code}' "http://${HOST}:${PORT}/health" 2>/dev/null)" != "200" ]; do
  if ! kill -0 "$SERVER_BG_PID" 2>/dev/null; then
    echo "ERROR: server background pipeline (pid=$SERVER_BG_PID) exited before" >&2
    echo "       /health became ready. Inspect $SERVER_LOG (likely OOM, port" >&2
    echo "       conflict, weight-load crash, or aiter kernel failure)." >&2
    tail -n 80 "$SERVER_LOG" >&2 || true
    exit 1
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "ERROR: server didn't come up within ${WAIT_FOR_SERVER_SEC}s — see $SERVER_LOG" >&2
    exit 1
  fi
  sleep 5
done
echo ">>> server ready (bg pid=$SERVER_BG_PID)"

# ============================== Warmup ==============================
echo ">>> warmup (random 1024/128 × 8 prompts × 4 concurrency)"
python3 -m sglang.bench_serving \
  --backend sglang --host "$HOST" --port "$PORT" \
  --model "$MODEL_PATH" --dataset-name random \
  --random-input 1024 --random-output 128 --random-range-ratio 1.0 \
  --max-concurrency 4 --num-prompt 8 --output-file /dev/null \
  2>&1 | tee "$LOG_DIR/warmup.log"

# ============================== Optional: GSM8K precheck ==============================
# Runs SGLang's built-in GSM8K accuracy bench against the SAME server,
# logs the accuracy into bench_meta.json so each cascade run carries a
# correctness signature. Both warmup and GSM8K are subsequently flushed
# (radix tree + OS page cache) before the cascade bench, so cascade
# still starts from a cold cache.
if [ "$GSM8K_PRECHECK" = "true" ]; then
  GSM8K_SCRIPT=""
  for c in /sgl-workspace/sglang/benchmark/gsm8k/bench_sglang.py \
           "$HOME/work-space/sglang/benchmark/gsm8k/bench_sglang.py"; do
    [ -f "$c" ] && GSM8K_SCRIPT="$c" && break
  done
  if [ -z "$GSM8K_SCRIPT" ]; then
    echo ">>> WARNING: --gsm8k-precheck requested but bench_sglang.py not found; skipping" >&2
  else
    # File names match GLM.sh's accuracy_test() so the cascade tree and
    # the GLM benchmark tree produce comparable accuracy artifacts.
    GSM8K_LOG="$LOG_DIR/Accuracy_GSM8K.log"
    GSM8K_RESULT_JSONL="$LOG_DIR/Accuracy_GSM8K.jsonl"
    echo ">>> GSM8K precheck: ${GSM8K_NUM_QUESTIONS} questions, parallel=${GSM8K_PARALLEL}"
    # cd into LOG_DIR so the bench script's tmp_output_*.txt and any other
    # cwd-relative outputs land beside the rest of this run's artifacts.
    if (
        cd "$LOG_DIR"
        # NOTE: bench_sglang.py's --backend is a *frontend* selector
        # (srt / srt-no-parallel / srt-raw / gpt-*) defined in
        # python/sglang/test/test_utils.py:select_sglang_backend(). It is NOT
        # the server-side --backend used by bench_serving.py (sglang/vllm/tgi).
        # Don't pass --backend here — the default "srt" is what we want and
        # what GLM.sh's accuracy_test() uses.
        python3 "$GSM8K_SCRIPT" \
          --host "$HOST" --port "$PORT" \
          --num-questions "$GSM8K_NUM_QUESTIONS" \
          --parallel "$GSM8K_PARALLEL" \
          --result-file "$GSM8K_RESULT_JSONL" \
          2>&1 | tee "$GSM8K_LOG"
    ); then
      # bench_sglang.py prints "Accuracy: 0.930" on stdout. Grep is robust
      # enough; falls back to None if the line isn't there for any reason.
      GSM8K_ACC=$(grep -oP '^Accuracy:\s+\K[0-9.]+' "$GSM8K_LOG" | tail -1 || true)
      if [ -n "$GSM8K_ACC" ]; then
        echo ">>> GSM8K accuracy: ${GSM8K_ACC}"
        update_bench_meta_accuracy "$GSM8K_ACC"
      else
        echo ">>> WARNING: GSM8K finished but no Accuracy line parsed; leaving accuracy=null in bench_meta.json" >&2
        update_bench_meta_accuracy "None"
      fi
    else
      echo ">>> WARNING: GSM8K precheck failed (rc=$?); continuing with cascade bench, accuracy=null in bench_meta.json" >&2
      update_bench_meta_accuracy "None"
    fi
  fi
fi

# Early-exit when --gsm8k-only was passed: skip the cascade benchmark
# and let the EXIT trap tear the server down (with graceful SIGTERM).
if [ "${GSM8K_ONLY:-false}" = "true" ]; then
  echo ">>> --gsm8k-only set; skipping cascade multiturn benchmark."
  echo ">>> done. results in: $LOG_DIR"
  if [ "${GSM8K_PRECHECK:-false}" = "true" ]; then
    echo "    accuracy log: $LOG_DIR/Accuracy_GSM8K.log"
    echo "    bench_meta:   $LOG_DIR/bench_meta.json"
  fi
  exit 0
fi

# ============================== Bench: multiturn ==============================
BENCH_SCRIPT=""
for c in /sgl-workspace/sglang/benchmark/hicache/bench_multiturn.py \
         "$HOME/work-space/sglang/benchmark/hicache/bench_multiturn.py"; do
  [ -f "$c" ] && BENCH_SCRIPT="$c" && break
done
if [ -z "$BENCH_SCRIPT" ]; then
  echo "ERROR: bench_multiturn.py not found" >&2
  exit 1
fi

# Flush radix tree so each run starts cold (HiCache itself controls L2/L3
# tier eviction). This wipes any KV pages left over from warmup or the
# optional GSM8K precheck.
curl -s -X POST "http://${HOST}:${PORT}/flush_cache" >/dev/null || true
sleep 2

# Best-effort drop of OS page cache so L3 NVMe reads are truly cold on
# round 1. Inside SGLang docker images we are usually root and can write
# /proc/sys/vm/drop_caches directly; on stripped-down hosts we fall back
# to passwordless sudo, then to a warning if neither works (cascade
# correctness is unaffected — only L3 cold-read latency may be slightly
# inflated by warmup/GSM8K page cache hits).
sync
if echo 3 > /proc/sys/vm/drop_caches 2>/dev/null; then
  echo ">>> dropped OS page cache"
elif command -v sudo >/dev/null 2>&1 && echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1; then
  echo ">>> dropped OS page cache (via sudo)"
else
  echo ">>> WARNING: could not drop OS page cache (need root or passwordless sudo);" \
       "L3 cold reads in round 1 may be slightly inflated by leftover page cache" >&2
fi
sleep 2

# Start the per-tier cache hit-rate monitor as a sidecar. It detects
# bench_multiturn.py's round-barrier crossings via sglang:num_requests_total
# and emits one line + one CSV row per finished round, with L1 / L2 / L3
# token-hit rates attributable to THAT round only (delta vs the previous
# round's snapshot — first round's delta uses the baseline captured at
# monitor start, so warmup / GSM8K precheck hits never pollute the cascade
# rates). No waiting for bench_multiturn.jsonl at the very end.
CACHE_MONITOR_SCRIPT="$(dirname "$(readlink -f "$0")")/cache_monitor.py"
if [ -f "$CACHE_MONITOR_SCRIPT" ]; then
  echo ">>> starting cache_monitor sidecar (time-driven 10s sample;"
  echo "    one CSV row per sample fuses /metrics cache hits + /proc/meminfo"
  echo "    host RAM, with round_index column to group rows by round)"
  python3 "$CACHE_MONITOR_SCRIPT" \
      --url "http://${HOST}:${PORT}/metrics" \
      --interval 10 \
      --num-clients "$NUM_CLIENTS" \
      --num-rounds "$NUM_ROUNDS" \
      --csv "$LOG_DIR/cache_tiers.csv" \
      > "$LOG_DIR/cache_monitor.log" 2>&1 &
  CACHE_MONITOR_PID=$!
  echo ">>> cache_monitor pid=${CACHE_MONITOR_PID}; tail -f ${LOG_DIR}/cache_monitor.log"
else
  echo ">>> WARNING: ${CACHE_MONITOR_SCRIPT} not found; skipping cache+host monitor" >&2
  CACHE_MONITOR_PID=""
fi

echo ">>> running cascade multiturn (N=${NUM_CLIENTS} × R=${REQUEST_LENGTH} × ${NUM_ROUNDS} rounds)"
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

if [ -n "${CACHE_MONITOR_PID:-}" ] && kill -0 "$CACHE_MONITOR_PID" 2>/dev/null; then
  echo ">>> stopping cache_monitor (pid=${CACHE_MONITOR_PID})"
  kill "$CACHE_MONITOR_PID" 2>/dev/null || true
  wait "$CACHE_MONITOR_PID" 2>/dev/null || true
fi

# ============================== Cleanup ==============================
# GLM.sh-style: pkill -9 sglang + sleep 10. Simple and reliable. The
# EXIT trap also fires after this, so anything that survives gets a
# second kill on the way out.
echo ">>> stopping server (pkill -9 sglang)"
pkill -9 sglang 2>/dev/null || true
sleep 10

if [ -n "$HICACHE_FILE_STORE_DIR" ] && [ -d "$HICACHE_FILE_STORE_DIR" ]; then
  sz=$(du -sh "$HICACHE_FILE_STORE_DIR" 2>/dev/null | cut -f1)
  rm -rf "$HICACHE_FILE_STORE_DIR"
  echo ">>> cleaned L3 file store (${sz:-?} reclaimed)"
fi

# Auto-suggest the right --MI355X-dir / --B200-dir flag for plot_cascade.py
# based on --tag. plot_cascade.py now takes cascade root dirs (one per
# platform) and auto-discovers L1/L2/L3_file jsonls inside.
case "$TAG" in
  *MI355X*|*MI300*|*MI325*|*MI250*|*MI210*|*ROCm*|*rocm*|*amd*) PLATFORM_FLAG="--MI355X-dir" ;;
  *B200*|*H200*|*H100*|*A100*|*L40*|*nvidia*|*NVIDIA*)          PLATFORM_FLAG="--B200-dir"  ;;
  *)                                                            PLATFORM_FLAG="--MI355X-dir" ;;
esac

echo ">>> done. results in: $LOG_DIR"
echo "    plot (after all cache modes done; auto-discovers L1/L2/L3_file):"
echo "      python3 plot_cascade.py \\"
echo "          ${PLATFORM_FLAG} ${BASE_LOG_DIR} \\"
echo "          $([ "$PLATFORM_FLAG" = "--MI355X-dir" ] && echo "--B200-dir <B200-cascade-root>" || echo "--MI355X-dir <MI355X-cascade-root>") \\"
echo "          --max-rounds 10"
echo "    Writes 3 PNGs into the MI355X dir: MI355X.png, B200.png, MI355X_VS_B200.png"
