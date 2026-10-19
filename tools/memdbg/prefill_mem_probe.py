"""Drive cached-prefix prefill batches to probe server peak memory.

Each case warms n random prefixes of --prefix tokens, then sends the n
prefixes extended by --extend new tokens concurrently, so the scheduler forms
one prefill batch of n seqs with n * prefix cached and n * extend new tokens.

Usage: python prefill_mem_probe.py --port 30000 --case 4x40000+4000 --case 1x150000+16000
"""

import argparse
import concurrent.futures as cf
import random
import time

import requests


def _gen(url, ids):
    r = requests.post(
        url,
        json={
            "input_ids": ids,
            "sampling_params": {"max_new_tokens": 1, "temperature": 0},
        },
        timeout=3600,
    )
    r.raise_for_status()
    return r.json()["meta_info"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--case", action="append", required=True, help="NxPREFIX+EXTEND")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    url = f"http://127.0.0.1:{args.port}/generate"
    rng = random.Random(args.seed)

    for case in args.case:
        n, rest = case.split("x")
        prefix, extend = (int(v) for v in rest.split("+"))
        n = int(n)
        prompts = [[rng.randrange(1000, 100000) for _ in range(prefix)] for _ in range(n)]
        t0 = time.time()
        for p in prompts:
            _gen(url, p)
        print(f"[probe] {case}: warmed {n}x{prefix} in {time.time() - t0:.1f}s", flush=True)
        ext = [p + [rng.randrange(1000, 100000) for _ in range(extend)] for p in prompts]
        t0 = time.time()
        with cf.ThreadPoolExecutor(n) as ex:
            metas = list(ex.map(lambda ids: _gen(url, ids), ext))
        cached = [m.get("cached_tokens") for m in metas]
        print(
            f"[probe] {case}: extend done in {time.time() - t0:.1f}s cached={cached}",
            flush=True,
        )


if __name__ == "__main__":
    main()
