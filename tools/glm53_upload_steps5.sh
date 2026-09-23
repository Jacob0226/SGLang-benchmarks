#!/usr/bin/env bash
# Upload MI355X steps=5 profile zip. Must run inside the 0914 container:
# hf CLI venv python lives at /opt/venv, and the Hub token is cached there.
set -uo pipefail
exec > /home/jacchang/SGLang-benchmarks/tmp/logs/glm53_upload_steps5.log 2>&1

export PATH=/home/jacchang/.local/bin:$PATH
REPO="${REPO:-JacobChang/GLM5.3-Flash-FP4}"
ZIP="${ZIP:-/home/jacchang/SGLang-benchmarks/tmp/MI355X_Docker0914-10PR-steps5.zip}"

echo "=== $(date -Is) start ==="
ls -lh "$ZIP"
hf auth whoami
hf upload "$REPO" "$ZIP" "$(basename "$ZIP")" --type dataset \
    --commit-message "Add MI355X MXFP4 profile with --profile-num-steps 5, all 4 TP ranks (graph-on + no-cuda-graph)"
echo "=== upload exit $? ==="
hf datasets list "$REPO" --tree
echo "=== $(date -Is) done ==="
