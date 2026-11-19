#!/bin/bash
# Node 152: two containers on rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261118 (SGLang 3a41a59919,
# aiter 581579ab5 which already has ROCm/aiter#6435, sgl-amd-mono a04ceab) for the stock vs
# stock + PR stack AgentX. jacchang_stock1118 is the image as shipped; jacchang_pr1118 also gets
# ROCm/aiter#8006 (KDA in-proj / o_proj BF16 tuned CSV) applied to the image's aiter.
set -e
IMG=rocm/sgl-dev:v0.5.23-rocm10-mi35x-20261118
for C in jacchang_stock1118 jacchang_pr1118; do
    docker rm -f $C >/dev/null 2>&1 || true
    docker run -d --name $C --network host --ipc host --shm-size 32g \
      --device /dev/kfd --device /dev/dri --group-add video --cap-add SYS_PTRACE --cap-add SYS_NICE \
      --security-opt seccomp=unconfined --security-opt label=disable \
      -v /shared_nfs/models/:/data/huggingface/hub -v /home/jacchang:/home/jacchang \
      -v /shared_nfs/jacchang:/shared_nfs/jacchang -w /home/jacchang/SGLang-benchmarks \
      $IMG sleep infinity
    docker exec $C bash -c '
git config --global --add safe.directory "*"
for d in sglang sgl-amd-mono aiter; do echo "'$C'" $d $(git -C /sgl-workspace/$d log -1 --format=%h); done
strings -a /sgl-workspace/aiter/aiter/jit/module_moe_mxfp4_aux.so 2>/dev/null | grep -c aux_scatter_H4096_TOPK9_NT1 || echo "no prebuilt aux .so with TOPK9 scatter"
(apt-get install -y -q psmisc >/tmp/apt.log 2>&1 || (apt-get update -q >/dev/null 2>&1 && apt-get install -y -q psmisc >/tmp/apt.log 2>&1))
which fuser'
done
docker exec jacchang_pr1118 bash -c '
cd /sgl-workspace/aiter && git apply /shared_nfs/jacchang/agentx_1119_img1118/aiter8006.patch && git status --short'
