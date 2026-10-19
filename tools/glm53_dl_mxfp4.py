"""Download amd/GLM-5.3-Flash-Quark-MXFP4 into the container's HF hub mount."""

import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

from huggingface_hub import snapshot_download

snapshot_download(
    "amd/GLM-5.3-Flash-Quark-MXFP4",
    revision="b5688f25491202978c19c4d036eef579f61bbe07",
    local_dir="/data/huggingface/hub/amd/GLM-5.3-Flash-Quark-MXFP4",
    max_workers=16,
)
print("DONE")
