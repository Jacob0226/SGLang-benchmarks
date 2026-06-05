#!/usr/bin/env python3
"""PoC: L2->L3 (host RAM -> NVMe file) write-bandwidth under different
strategies, to quantify the headroom over the current per-page + fsync path.

Strategies:
  per_page_fsync   : many small writes, fsync after EACH  (≈ current worst case)
  per_page_nofsync : many small writes, fsync ONCE at end
  bulk_nofsync     : one big sequential write, fsync once
  bulk_fsync       : one big sequential write + fsync
  bulk_odirect     : one big O_DIRECT write (bypass page cache)
  parallel_files   : --par files written concurrently (simulate striping; put
                     --dir on a RAID0 / different mounts to see multi-drive gain)

Usage:
  python tools/bench_l3_write.py --dir /data --total-mb 1024 --page-kb 256 --par 8
"""
import argparse
import mmap
import os
import time
from concurrent.futures import ThreadPoolExecutor


def gbps(nbytes, sec):
    return nbytes / sec / 1e9


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="/data")
    p.add_argument("--total-mb", type=int, default=1024)
    p.add_argument("--page-kb", type=int, default=256, help="small-write chunk size")
    p.add_argument("--par", type=int, default=8, help="parallel files/threads")
    p.add_argument("--iters", type=int, default=5)
    args = p.parse_args()

    total = args.total_mb * 1024 * 1024
    page = args.page_kb * 1024
    buf = bytearray(os.urandom(min(total, 64 * 1024 * 1024)))  # source bytes (reused)

    def time_it(fn, warmup=1):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            fn()
        return (time.perf_counter() - t0) / args.iters

    def path(i=0):
        return os.path.join(args.dir, f"_l3w_{os.getpid()}_{i}.bin")

    def per_page(fsync_each):
        fp = path()
        with open(fp, "wb", buffering=0) as f:
            written = 0
            while written < total:
                n = min(page, total - written, len(buf))
                f.write(buf[:n]); written += n
                if fsync_each:
                    os.fsync(f.fileno())
            if not fsync_each:
                f.flush(); os.fsync(f.fileno())
        os.remove(fp)

    def bulk(fsync):
        fp = path()
        big = (buf * (total // len(buf) + 1))[:total]
        with open(fp, "wb", buffering=0) as f:
            f.write(big)
            if fsync:
                f.flush(); os.fsync(f.fileno())
        os.remove(fp)

    def bulk_odirect():
        fp = path()
        bs = 4096
        n = (total // bs) * bs
        m = mmap.mmap(-1, n)  # page-aligned
        m.write((buf * (n // len(buf) + 1))[:n])
        fd = os.open(fp, os.O_WRONLY | os.O_CREAT | os.O_DIRECT | os.O_TRUNC, 0o644)
        mv = memoryview(m)
        try:
            off = 0
            while off < n:
                off += os.pwrite(fd, mv[off:off + 64 * 1024 * 1024], off)
            os.fsync(fd)
        finally:
            mv.release()
            os.close(fd); m.close(); os.remove(fp)

    def parallel_files():
        per = total // args.par
        big = (buf * (per // len(buf) + 1))[:per]

        def one(i):
            fp = path(i)
            with open(fp, "wb", buffering=0) as f:
                f.write(big); f.flush(); os.fsync(f.fileno())
            os.remove(fp)
        with ThreadPoolExecutor(max_workers=args.par) as ex:
            list(ex.map(one, range(args.par)))

    print(f"=== L2->L3 write strategies @ {args.dir}  ({args.total_mb} MB, page={args.page_kb}KB, par={args.par}) ===")
    for name, fn in [
        ("per_page_fsync  (≈current)", lambda: per_page(True)),
        ("per_page_nofsync", lambda: per_page(False)),
        ("bulk_fsync", lambda: bulk(True)),
        ("bulk_nofsync", lambda: bulk(False)),
        ("bulk_odirect", bulk_odirect),
        (f"parallel_files x{args.par}", parallel_files),
    ]:
        try:
            t = time_it(fn)
            print(f"  {name:28} : {gbps(total, t):7.2f} GB/s  ({t*1e3:8.1f} ms)")
        except Exception as exc:
            print(f"  {name:28} : [error] {str(exc)[:70]}")


if __name__ == "__main__":
    main()
