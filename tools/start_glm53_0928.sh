#!/bin/bash
# Detached, not -it --rm: an interactive container dies with the srun session.
DATA=/mnt/m2m_nobackup/models/
docker rm -f jacchang_GLM53-Flash-1 2>/dev/null
docker run -d --privileged --name=jacchang_GLM53-Flash-1 --network=host --device=/dev/kfd \
    --device=/dev/dri --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
    --ipc=host --shm-size=32g -w $HOME/SGLang-benchmarks/ \
    -v $DATA/:/data/huggingface/hub -v $HOME:$HOME -e USER=$(whoami) -e HOME=$HOME -e TERM=xterm \
    rocm/sgl-dev:v0.5.20-rocm10-mi35x-20260928 \
    tail -f /dev/null
docker ps --filter name=jacchang_GLM53-Flash-1 --format "{{.Names}} | {{.Image}} | {{.Status}}"
