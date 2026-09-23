#!/usr/bin/env bash
# Probe the 0922 ROCm 10 image before porting the Day-0 PR stack onto it.
# Output: ~/glm53_image0922.txt
exec > /home/jacchang/SGLang-benchmarks/tmp/logs/glm53_image0922.txt 2>&1

echo "=== in-tree sglang ==="
git config --global --add safe.directory /sgl-workspace/sglang 2>/dev/null
git -C /sgl-workspace/sglang log --oneline -3
git -C /sgl-workspace/sglang rev-parse HEAD
echo "--- dirty files ---"
git -C /sgl-workspace/sglang status --short | head

echo "=== python sglang package ==="
python3 -c 'import sglang, os; print(sglang.__version__, os.path.dirname(sglang.__file__))'

echo "=== aiter ==="
git config --global --add safe.directory /sgl-workspace/aiter 2>/dev/null
git -C /sgl-workspace/aiter rev-parse HEAD 2>&1

echo "=== rocm / torch ==="
cat /opt/rocm/.info/version 2>/dev/null
python3 -c 'import torch; print("torch", torch.__version__, "| hip", torch.version.hip)'
python3 -c 'import triton; print("triton", triton.__version__)'

echo "=== is #39200 (fused mHC attn->MLP boundary) in the image tree? ==="
grep -c 'hc_ffn_post_pre' /sgl-workspace/sglang/python/sglang/srt/models/glm5_next.py 2>/dev/null

echo "=== home visible? ==="
ls -d /home/jacchang/SGLang-benchmarks/tools >/dev/null && echo "yes"
ls -d /home/jacchang/PR/glm53-day0-stack >/dev/null && echo "0914 stack tree visible"

echo "=== model mount ==="
ls /data/huggingface/hub/ 2>&1 | head
df -h /data/huggingface/hub | tail -1

echo "=== hf cli ==="
export PATH=/home/jacchang/.local/bin:$PATH
command -v hf && hf auth whoami 2>&1 | head -3
