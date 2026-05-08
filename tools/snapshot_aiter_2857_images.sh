#!/usr/bin/env bash
# snapshot_aiter_2857_images.sh
# ------------------------------------------------------------------------------
# Capture two side-by-side docker images for the aiter#2857 investigation:
#
#   jacchang/shared:sglang-aiter2857-rocm722-mi35x-aiter011post1-20260427
#       = ROCm 7.2.2 base image + aiter v0.1.11.post1 (PRE-regression)
#         GSM8K accuracy on Qwen3-235B-A22B-Instruct-2507-mxfp4 = 0.9401  (PASS)
#
#   jacchang/shared:sglang-aiter2857-rocm722-mi35x-aiter012post1-20260427
#       = ROCm 7.2.2 base image + aiter v0.1.12.post1 (POST-regression)
#         GSM8K accuracy on Qwen3-235B-A22B-Instruct-2507-mxfp4 = 0.909   (FAIL)
#
# The watchdog race (graph capture crash) is fixed in BOTH images by the
# ROCm 7.2.2 base bump. Only the aiter pin differs, so any A/B you do
# with these two images isolates the aiter v0.1.11.post1 -> v0.1.12.post1
# numerics regression while keeping the rest of the stack constant.
#
# Strategy:
#   - The base image jacchang/shared:sglang-aiter2857-rocm722-mi35x-20260424
#     already contains aiter v0.1.12.post1 -> just `docker tag` it for the
#     post-regression name (instant, no extra disk).
#   - The currently-running container ci_sglang_aiter2857ci has been
#     downgraded to aiter v0.1.11.post1 (rebuilt via setup.py develop) ->
#     `docker commit` it for the pre-regression name.
#
# Optional `--push` flag pushes both tags to Docker Hub.
#
# Usage:
#   bash snapshot_aiter_2857_images.sh           # commit + tag locally
#   bash snapshot_aiter_2857_images.sh --push    # also docker push both
set -euo pipefail

BASE_IMAGE="jacchang/shared:sglang-aiter2857-rocm722-mi35x-20260424"
CONTAINER="ci_sglang_aiter2857ci"
DATE_TAG="20260427"

IMG_011POST1="jacchang/shared:sglang-aiter2857-rocm722-mi35x-aiter011post1-${DATE_TAG}"
IMG_012POST1="jacchang/shared:sglang-aiter2857-rocm722-mi35x-aiter012post1-${DATE_TAG}"

PUSH=0
[[ "${1:-}" == "--push" ]] && PUSH=1

log() { printf '[snapshot-2857] %s\n' "$*" >&2; }

# ----------------------------------------------------------------------------
# 1. Sanity checks: container exists with v0.1.11.post1; base image exists.
# ----------------------------------------------------------------------------
if ! docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
  log "ERROR: base image $BASE_IMAGE not found locally"
  log "       (this is the ROCm 7.2.2 + aiter v0.1.12.post1 build)"
  exit 1
fi

if ! docker inspect -f '{{.State.Status}}' "$CONTAINER" >/dev/null 2>&1; then
  log "ERROR: container $CONTAINER not found. It must be running with"
  log "       aiter v0.1.11.post1 already rebuilt inside it."
  log "       (this script does not do the rebuild itself)"
  exit 1
fi

log "verifying aiter version inside $CONTAINER (expect 0.1.11.post*) ..."
got_aiter=$(docker exec "$CONTAINER" bash -lc 'pip show amd-aiter 2>/dev/null | awk "/^Version:/{print \$2}"' || echo "")
if [[ "$got_aiter" != 0.1.11.post* ]]; then
  log "WARN: container's amd-aiter version is '$got_aiter' (expected 0.1.11.post*)."
  log "      If you have not rebuilt to v0.1.11.post1, the snapshot will be wrong."
  log "      Continue anyway? Re-run with FORCE=1 to skip this check."
  if [[ "${FORCE:-0}" != "1" ]]; then
    exit 1
  fi
fi
log "container aiter version: $got_aiter"

# ----------------------------------------------------------------------------
# 2. Snapshot the v0.1.11.post1 state via docker commit.
# ----------------------------------------------------------------------------
log "committing $CONTAINER -> $IMG_011POST1"
docker commit \
  --message "Snapshot of $BASE_IMAGE with aiter rebuilt to v0.1.11.post1 ($got_aiter); $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  --change 'LABEL aiter_version=0.1.11.post1' \
  --change 'LABEL aiter_2857_role=pre-regression-baseline' \
  --change 'LABEL aiter_2857_gsm8k_accuracy=0.9401' \
  --change "LABEL aiter_2857_base_image=${BASE_IMAGE}" \
  "$CONTAINER" "$IMG_011POST1"

# ----------------------------------------------------------------------------
# 3. Tag the base image as the v0.1.12.post1 image (it already is one).
# ----------------------------------------------------------------------------
log "tagging $BASE_IMAGE as $IMG_012POST1 (no rebuild needed; base already has v0.1.12.post1)"
docker tag "$BASE_IMAGE" "$IMG_012POST1"

# ----------------------------------------------------------------------------
# 4. Show both tags.
# ----------------------------------------------------------------------------
log "local images:"
docker images --filter=reference='jacchang/shared:sglang-aiter2857-rocm722-mi35x-aiter*' \
              --format 'table {{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}'

# ----------------------------------------------------------------------------
# 5. Optional push.
# ----------------------------------------------------------------------------
if [[ "$PUSH" -eq 1 ]]; then
  if ! grep -q '"auths"' "$HOME/.docker/config.json" 2>/dev/null; then
    log "WARN: ~/.docker/config.json has no auth entries; run 'docker login' first if push fails"
  fi
  log "pushing $IMG_011POST1"
  docker push "$IMG_011POST1"
  log "pushing $IMG_012POST1"
  docker push "$IMG_012POST1"
  log "pushed both."
fi

log "done."
log "  $IMG_011POST1  (aiter v0.1.11.post1, GSM8K 0.9401, pre-regression)"
log "  $IMG_012POST1  (aiter v0.1.12.post1, GSM8K 0.909, post-regression)"
