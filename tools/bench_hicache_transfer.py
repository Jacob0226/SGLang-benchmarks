#!/usr/bin/env python3
"""Portable HiCache KV-transfer benchmark for MI355X (ROCm) and B200 (CUDA).

Outputs, in one run:
  [HOPS]   L1<->L2 (GPU HBM <-> pinned host), L2->L3 / L3->L2 (host <-> NVMe file)
  [LAYOUT] MLA transfer kernels for layer-first vs page-first:
             load   lf->lf, pf->lf      (pf->lf = real serving load path)
             backup lf->lf, lf->pf      (lf->pf = real serving backup path)

Run the same command on both boxes and diff the numbers:
  python tools/bench_hicache_transfer.py --nvme-dir /data \
      --layer-num 61 --item-bytes 1152 --tokens 4096 \
      --host-tokens 131072 --dev-tokens 131072 --blobs 64 --iters 50

Notes:
  * device side is always layer-first (attention reads it directly); we vary the
    HOST layout (lf vs pf), matching hicache_mem_layout.
  * pf<->lf are registered in torch.ops.sgl_kernel even when not re-exported by
    sgl_kernel's __init__; we call them via torch.ops with explicit block_quota /
    num_warps_per_block (HIP=16, CUDA=32).
  * --item-bytes 1152 = DeepSeek MLA latent (512 + 64) * 2 bytes.
"""
import argparse
import os
import time

import torch


def cuda_time(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters / 1000.0


def wall_time(fn, iters, warmup=1):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def gbps(nbytes, sec):
    return nbytes / sec / 1e9


def bench_hops(args):
    print("\n========== [HOPS]  L1 <-> L2 <-> L3(NVMe) ==========")
    nbytes = args.num_layers * args.page_tokens * args.item_bytes * args.blobs
    print(f"blob = {args.num_layers}L x {args.page_tokens}tok x {args.item_bytes}B "
          f"x {args.blobs}pages = {nbytes/1e6:.1f} MB/transfer")
    dgpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    hpin = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    hpin2 = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    path = os.path.join(args.nvme_dir, f"_kvhop_{os.getpid()}.bin")

    t = cuda_time(lambda: hpin.copy_(dgpu, non_blocking=True), args.iters)
    print(f"  L1->L2  HBM  -> host(pin) : {gbps(nbytes,t):7.2f} GB/s  ({t*1e3:7.2f} ms)")
    t = cuda_time(lambda: dgpu.copy_(hpin, non_blocking=True), args.iters)
    print(f"  L2->L1  host -> HBM       : {gbps(nbytes,t):7.2f} GB/s  ({t*1e3:7.2f} ms)")

    np_view = hpin.numpy()

    def wr():
        with open(path, "wb", buffering=0) as f:
            f.write(np_view.tobytes())
            f.flush()
            os.fsync(f.fileno())
    t = wall_time(wr, max(3, args.iters // 4))
    print(f"  L2->L3  host -> NVMe file : {gbps(nbytes,t):7.2f} GB/s  ({t*1e3:7.2f} ms) fsync")
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except Exception:
        pass

    def rd():
        with open(path, "rb", buffering=0) as f:
            data = f.read()
        hpin2.numpy()[:len(data)] = torch.frombuffer(bytearray(data), dtype=torch.uint8).numpy()
    t = wall_time(rd, max(3, args.iters // 4))
    print(f"  L3->L2  NVMe -> host(pin) : {gbps(nbytes,t):7.2f} GB/s  ({t*1e3:7.2f} ms)")
    os.remove(path)


def bench_layout(args):
    print("\n========== [LAYOUT]  MLA transfer kernels (host lf vs pf) ==========")
    try:
        import sgl_kernel  # noqa: F401  registers torch.ops.sgl_kernel
        ops = torch.ops.sgl_kernel
        ops.transfer_kv_per_layer_mla_pf_lf  # probe
    except Exception as exc:
        print(f"  [skip] sgl_kernel transfer ops unavailable: {str(exc)[:80]}")
        return
    nw = 16 if torch.version.hip is not None else 32
    bq = 2
    L, item = args.num_layers, args.item_bytes
    layout_dim = item * L
    moved = args.tokens * item * L  # bytes for all-layer

    dev = [torch.empty(args.dev_tokens * item, dtype=torch.uint8, device="cuda") for _ in range(L)]
    dev2 = [torch.empty(args.dev_tokens * item, dtype=torch.uint8, device="cuda") for _ in range(L)]
    host = torch.empty(args.host_tokens * L * item, dtype=torch.uint8, pin_memory=True)
    g = torch.Generator(); g.manual_seed(0)
    hidx = torch.randperm(args.host_tokens, generator=g)[:args.tokens].cuda().long()
    didx = torch.randperm(args.dev_tokens, generator=g)[:args.tokens].cuda().long()
    dptr = torch.tensor([t.data_ptr() for t in dev], dtype=torch.uint64, device="cuda")
    dptr2 = torch.tensor([t.data_ptr() for t in dev2], dtype=torch.uint64, device="cuda")

    def load_lf():
        for li in range(L):
            ops.transfer_kv_per_layer_mla.default(dev2[li], dev[li], didx, didx, item, bq, nw)

    def load_pf():
        for li in range(L):
            ops.transfer_kv_per_layer_mla_pf_lf.default(host, dev[li], hidx, didx, li, item, layout_dim, bq, nw)

    def backup_lf():
        ops.transfer_kv_all_layer_mla.default(dptr, dptr2, didx, didx, item, L, bq, nw)

    def backup_pf():
        ops.transfer_kv_all_layer_mla_lf_pf.default(dptr, host, didx, hidx, item, layout_dim, L, bq, nw)

    print(f"  {args.tokens} tok/call, {moved/1e6:.0f} MB/all-layer")
    for label, fn in [("load   lf->lf", load_lf), ("load   pf->lf (REAL)", load_pf),
                      ("backup lf->lf", backup_lf), ("backup lf->pf (REAL)", backup_pf)]:
        t = cuda_time(fn, args.iters)
        print(f"  {label:22} : {gbps(moved,t):7.2f} GB/s  ({t*1e3:8.3f} ms)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nvme-dir", default="/data")
    p.add_argument("--num-layers", type=int, default=61)
    p.add_argument("--item-bytes", type=int, default=1152)
    p.add_argument("--tokens", type=int, default=4096, help="scattered tokens/call for LAYOUT")
    p.add_argument("--host-tokens", type=int, default=131072)
    p.add_argument("--dev-tokens", type=int, default=131072)
    p.add_argument("--page-tokens", type=int, default=64, help="contiguous tokens/page for HOPS blob")
    p.add_argument("--blobs", type=int, default=64)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--skip-hops", action="store_true")
    p.add_argument("--skip-layout", action="store_true")
    args = p.parse_args()

    assert torch.cuda.is_available(), "no CUDA/HIP device"
    print(f"=== HiCache transfer bench on {torch.cuda.get_device_name()} "
          f"(hip={torch.version.hip}, cuda={torch.version.cuda}) ===")
    if not args.skip_hops:
        bench_hops(args)
    if not args.skip_layout:
        bench_layout(args)


if __name__ == "__main__":
    main()
