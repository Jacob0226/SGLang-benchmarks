#!/usr/bin/env bash
# Prefill A/B between two images with GLM.sh: one lane per container, same args.
# usage (inside the container): prefill_reg_lane_1119.sh <gpus> <port> <image> [variant...]
#   variant: pgoff (--cuda-graph-backend-prefill disabled) | bcg (breakable, max-bs 2560)
# Each variant runs a TTFT bench (1K/8K/70K in, conc 1/4/8) and a conc1 EXTEND profile.
set -uo pipefail
GPUS=$1; PORT=$2; IMAGE=$3; shift 3
VARIANTS=("$@")
[ ${#VARIANTS[@]} -eq 0 ] && VARIANTS=(pgoff bcg)

[ -e /root/SGLang-benchmarks ] || ln -s /home/jacchang/SGLang-benchmarks /root/SGLang-benchmarks
cd /home/jacchang/SGLang-benchmarks/BenchFixedLength
MODEL=/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4

export HIP_VISIBLE_DEVICES=$GPUS PORT SKIP_GSM8K=1
for v in "${VARIANTS[@]}"; do
  case $v in
    pgoff) export SERVER_EXTRA_ARGS="--cuda-graph-backend-prefill disabled" ;;
    bcg)   export SERVER_EXTRA_ARGS="--cuda-graph-backend-prefill breakable --cuda-graph-max-bs-prefill 2560" ;;
    *) echo "unknown variant $v"; exit 1 ;;
  esac
  echo "##### $(date '+%F %T') variant=$v bench"
  IN_OUT_OVERRIDE="1024:64 8192:64 70000:64" CONC_OVERRIDE="1 4 8" \
    bash ./GLM.sh --model $MODEL --mtp --golden-al --docker "$IMAGE" --tag "prefill1119-$v"
  echo "##### $(date '+%F %T') variant=$v prof"
  PROF_SERVER_MODES_OVERRIDE=default PROF_IN_OUT_OVERRIDE="8192:16 70000:16" \
  PROF_CONC_OVERRIDE="1" PROF_MTP_OUTPUT_LEN=64 \
    bash ./GLM.sh --model $MODEL --mtp --golden-al --prof --docker "$IMAGE" --tag "prefill1119-$v"
done
echo "##### $(date '+%F %T') LANE_DONE"
