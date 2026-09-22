#!/usr/bin/env bash
# Fetch amd/GLM-5.3-Flash-Quark-MXFP4 onto this node. Node crsuse2-m2m-172 only
# carries GLM-5.2 checkpoints, and job 147020 (which had it under /data) is gone.
#
# Laid out as <org>/<name> rather than the node's flat <org>__<name> convention
# on purpose: GLM.sh derives MODEL_NAME from the last TWO path components, so a
# flat directory resolves to "hub_amd__GLM-5.3-Flash-Quark-MXFP4", misses every
# per-model recipe block, and silently falls back to triton DSA -- which this
# model cannot use at all (index_kpool=4).
#
# Output: ~/glm53_fetch.log
set -uo pipefail
exec > /home/jacchang/glm53_fetch.log 2>&1

REPO="${REPO:-amd/GLM-5.3-Flash-Quark-MXFP4}"
DEST="${DEST:-/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4}"

python3 -c 'import huggingface_hub' 2>/dev/null \
    || python3 -m pip install -q --break-system-packages "huggingface_hub>=1.0" \
    || python3 -m pip install -q "huggingface_hub>=1.0"
python3 -c 'import huggingface_hub; print("huggingface_hub", huggingface_hub.__version__)'

mkdir -p "$DEST"
echo "=== downloading $REPO -> $DEST at $(date '+%F %T') ==="

python3 - "$REPO" "$DEST" <<'PY'
import sys
from huggingface_hub import snapshot_download

repo, dest = sys.argv[1], sys.argv[2]
path = snapshot_download(
    repo_id=repo,
    local_dir=dest,
    max_workers=16,
    # Weights + config only; no need for the .bin duplicates if any exist.
    allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.md", "*.txt", "LICENSE"],
)
print("snapshot at", path)
PY

echo "=== done at $(date '+%F %T') ==="
du -sh "$DEST"
ls "$DEST" | wc -l
