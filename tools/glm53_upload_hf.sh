#!/usr/bin/env bash
# Upload the packaged TP-0 profile archive to the Hub.
# Runs inside the container: the hf CLI's venv points at /opt/venv/bin/python3,
# which only exists there, and the cached token is under the container's root
# $HOME/.cache/huggingface.
set -uo pipefail
exec > /home/jacchang/SGLang-benchmarks/tmp/logs/glm53_upload.log 2>&1

export PATH=/home/jacchang/.local/bin:$PATH
REPO="${REPO:-JacobChang/GLM5.3-Flash-FP4}"
ZIP="${ZIP:-/home/jacchang/SGLang-benchmarks/tmp/Docker0914-10PR.zip}"

hf auth whoami
hf repos create "$REPO" --type dataset --exist-ok
hf upload "$REPO" "$ZIP" "$(basename "$ZIP")" --type dataset \
    --commit-message "GLM-5.3-Flash Quark-MXFP4 TP4 profile traces: docker 0914 image + 10 Day-0 PRs"
echo "=== upload exit $? ==="
hf datasets list "$REPO" --tree
