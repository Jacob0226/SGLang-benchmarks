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

import numpy as np
import torch

O_DIRECT = getattr(os, "O_DIRECT", 0)
ALIGN = 4096  # O_DIRECT buffer/offset/length alignment


def aligned_buf(nbytes, align=ALIGN):
    """Page-aligned uint8 numpy buffer (required for O_DIRECT)."""
    raw = np.empty(nbytes + align, dtype=np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes]


def drop_file_cache(path):
    try:
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except (AttributeError, OSError):
        pass


def write_buffered(path, mv, fsync):
    with open(path, "wb") as f:
        f.write(mv)
        f.flush()
        if fsync:
            os.fsync(f.fileno())


def write_odirect(path, mv):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | O_DIRECT, 0o644)
    try:
        n = 0
        while n < len(mv):
            n += os.write(fd, mv[n:])
        os.fsync(fd)
    finally:
        os.close(fd)


def read_into(path, mv, odirect):
    flags = os.O_RDONLY | (O_DIRECT if odirect else 0)
    fd = os.open(path, flags)
    try:
        off = 0
        while off < len(mv):
            n = os.preadv(fd, [mv[off:]], off)
            if n == 0:
                break
            off += n
    finally:
        os.close(fd)


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
    path = os.path.join(args.nvme_dir, f"_kvhop_{os.getpid()}.bin")

    t = cuda_time(lambda: hpin.copy_(dgpu, non_blocking=True), args.iters)
    print(f"  L1->L2  HBM  -> host(pin) : {gbps(nbytes,t):7.2f} GB/s  ({t*1e3:7.2f} ms)")
    t = cuda_time(lambda: dgpu.copy_(hpin, non_blocking=True), args.iters)
    print(f"  L2->L1  host -> HBM       : {gbps(nbytes,t):7.2f} GB/s  ({t*1e3:7.2f} ms)")

    # L2<->L3 file I/O. Use pre-allocated, page-aligned host buffers and low-level
    # os.write/os.preadv so the timed region is pure file I/O (no tobytes()/bytearray
    # copies). write/fsync are measured separately, and an O_DIRECT variant gives the
    # true drive bandwidth (bypasses the page cache, the only honest read/write number).
    nb_a = (nbytes // ALIGN) * ALIGN  # align length for O_DIRECT
    src = aligned_buf(nb_a)
    dst = aligned_buf(nb_a)
    src[:] = 0
    mv_src, mv_dst = memoryview(src), memoryview(dst)
    n4 = max(3, args.iters // 4)
    print(f"  --- L2->L3  host -> NVMe (host buffer already in RAM, single stream) ---")

    t = wall_time(lambda: write_buffered(path, mv_src, fsync=False), n4)
    print(f"  write buffered, no fsync  : {gbps(nb_a,t):7.2f} GB/s  ({t*1e3:7.2f} ms)  [page cache only]")
    t = wall_time(lambda: write_buffered(path, mv_src, fsync=True), n4)
    print(f"  write buffered + fsync    : {gbps(nb_a,t):7.2f} GB/s  ({t*1e3:7.2f} ms)  [HiCache-like path]")
    if O_DIRECT:
        t = wall_time(lambda: write_odirect(path, mv_src), n4)
        print(f"  write O_DIRECT + fsync    : {gbps(nb_a,t):7.2f} GB/s  ({t*1e3:7.2f} ms)  [TRUE drive write]")

    print(f"  --- L3->L2  NVMe -> host (read into pre-alloc buffer) ---")
    write_buffered(path, mv_src, fsync=True)  # ensure file exists
    t = wall_time(lambda: read_into(path, mv_dst, odirect=False), n4)
    print(f"  read cached (warm)        : {gbps(nb_a,t):7.2f} GB/s  ({t*1e3:7.2f} ms)  [served from page cache]")
    if O_DIRECT:
        def rd_cold():
            drop_file_cache(path)
            read_into(path, mv_dst, odirect=True)
        t = wall_time(rd_cold, n4, warmup=0)
        print(f"  read O_DIRECT (cold)      : {gbps(nb_a,t):7.2f} GB/s  ({t*1e3:7.2f} ms)  [TRUE drive read]")
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

    rows = [("AOT load   lf->lf", load_lf), ("AOT load   pf->lf (REAL)", load_pf),
            ("AOT backup lf->lf", backup_lf), ("AOT backup lf->pf (REAL)", backup_pf)]

    if args.jit:
        try:
            from sglang.jit_kernel.hicache import (
                can_use_hicache_jit_kernel,
                transfer_hicache_all_layer_mla as jit_all,
                transfer_hicache_one_layer_mla as jit_one,
            )
            assert can_use_hicache_jit_kernel(element_size=item), \
                "JIT not usable for this element_size (compile failed? apply PR #25154 on ROCm)"
            d2 = [d.view(args.dev_tokens, item) for d in dev]
            d2b = [d.view(args.dev_tokens, item) for d in dev2]
            h3 = host.view(args.host_tokens, L, item)
            hlptr = torch.tensor([host.data_ptr() + li * item for li in range(L)],
                                 dtype=torch.uint64, device="cuda")

            def jit_load_lf():
                for li in range(L):
                    jit_one(cache_dst=d2[li], indices_dst=didx,
                            cache_src=d2b[li], indices_src=didx, element_dim=item)

            def jit_load_pf():
                for li in range(L):
                    jit_one(cache_dst=d2[li], indices_dst=didx,
                            cache_src=h3[:, li, :], indices_src=hidx, element_dim=item)

            def jit_backup_lf():
                jit_all(ptr_dst=dptr2, indices_dst=didx, ptr_src=dptr, indices_src=didx,
                        cache_src_stride_bytes=item, cache_dst_stride_bytes=item, element_size=item)

            def jit_backup_pf():
                jit_all(ptr_dst=hlptr, indices_dst=hidx, ptr_src=dptr, indices_src=didx,
                        cache_src_stride_bytes=item, cache_dst_stride_bytes=layout_dim, element_size=item)

            rows += [("JIT load   lf->lf", jit_load_lf), ("JIT load   pf->lf (REAL)", jit_load_pf),
                     ("JIT backup lf->lf", jit_backup_lf), ("JIT backup lf->pf (REAL)", jit_backup_pf)]
        except Exception as exc:
            print(f"  [JIT skip] {str(exc)[:110]}")

    print(f"  {args.tokens} tok/call, {moved/1e6:.0f} MB/all-layer")
    for label, fn in rows:
        try:
            t = cuda_time(fn, args.iters)
            print(f"  {label:26} : {gbps(moved,t):7.2f} GB/s  ({t*1e3:8.3f} ms)")
        except Exception as exc:
            print(f"  {label:26} : [error] {str(exc)[:60]}")


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
    p.add_argument("--jit", action="store_true",
                   help="also bench the JIT kernels (apply PR #25154 first on ROCm)")
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
