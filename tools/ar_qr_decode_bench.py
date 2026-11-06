#!/usr/bin/env python3
"""Custom allreduce vs quick reduce at decode-sized messages, inside CUDA graphs, through
SGLang's own TP group communicators (the objects the server dispatches to).

Each method is captured as REPS back-to-back allreduces of a [rows, hidden] bf16 tensor and
replayed; reported is the per-allreduce time on rank 0 and the relative error against the
custom allreduce result.

usage (inside the ROCm container, GPUs via HIP_VISIBLE_DEVICES):
  ROCM_QUICK_REDUCE_QUANTIZATION=INT4 python3 ar_qr_decode_bench.py --tp 4 \
      --rows 32,64,96,128,192,256,384,512 --hidden 4096
"""
import argparse
import os
import socket

import torch
import torch.multiprocessing as mp


def free_port():
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def worker(rank, a, port):
    from sglang.srt.distributed import init_distributed_environment
    from sglang.srt.distributed.parallel_state import graph_capture, initialize_model_parallel
    from sglang.srt.runtime_context import get_parallel
    from sglang.test.test_utils import publish_build_topology

    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=a.tp, rank=rank, distributed_init_method=f"tcp://localhost:{port}",
        local_rank=rank,
    )
    publish_build_topology(tp_size=a.tp, world_rank=rank)
    initialize_model_parallel()
    tp = get_parallel().tp_group
    data = torch.zeros(1, device="cuda")
    torch.distributed.all_reduce(data, group=tp.device_group)
    torch.cuda.synchronize()
    ca, qr = tp.ca_comm, tp.qr_comm
    if rank == 0:
        print(f"ca={ca is not None and not ca.disabled} qr={qr is not None and not qr.disabled} "
              f"qr_level={getattr(qr, 'qr_quant_level', None)} fp16={getattr(qr, 'use_fp16_kernels', None)}",
              flush=True)
        print(f"{'rows':>6}{'MB':>7}{'ca_us':>9}{'qr_us':>9}{'qr_gate':>9}{'rel_err':>10}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(1234 + rank)
    for rows in [int(r) for r in a.rows.split(",")]:
        x = torch.randn(rows, a.hidden, generator=g, device="cuda", dtype=torch.bfloat16)
        res = {}
        outs = {}
        for name in ("ca", "qr"):
            if name == "qr" and (qr is None or qr.disabled):
                continue
            with graph_capture() as ctx:
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=ctx.stream):
                    for _ in range(a.reps):
                        out = (ca.custom_all_reduce(x) if name == "ca"
                               else qr.quick_all_reduce(x))
            for _ in range(3):
                graph.replay()
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(a.iters):
                graph.replay()
            end.record()
            torch.cuda.synchronize()
            res[name] = start.elapsed_time(end) * 1e3 / (a.iters * a.reps)
            outs[name] = out.float().clone()
            del graph
        if rank == 0:
            gate = qr.should_quick_allreduce(x) if qr is not None and not qr.disabled else False
            err = ((outs["qr"] - outs["ca"]).norm() / outs["ca"].norm()).item() if "qr" in outs else float("nan")
            mb = rows * a.hidden * 2 / 2**20
            print(f"{rows:6d}{mb:7.2f}{res['ca']:9.1f}{res.get('qr', float('nan')):9.1f}"
                  f"{str(bool(gate)):>9}{err:10.4f}", flush=True)
    torch.distributed.barrier()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--rows", default="32,64,96,128,192,256,384,512")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--iters", type=int, default=50)
    a = ap.parse_args()
    mp.spawn(worker, args=(a, free_port()), nprocs=a.tp, join=True)


if __name__ == "__main__":
    main()
