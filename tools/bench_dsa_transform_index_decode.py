"""Is the tiled DSA page-table transform a drop-in replacement for the 2048 one?

#39340 keeps the original single-program kernel for width 2048 and routes only
other widths to the new tiled kernel. That split is only worth its branch if the
original is actually faster at 2048 -- and it is not obvious that it is, because
it launches one program per request, which at decode batch sizes leaves an
MI355X almost idle.

Checks equality first, then times both at the widths and batch sizes decode
actually emits. Rotates six input sets so the 256 MB Infinity Cache does not
answer instead of the kernel.

    PYTHONPATH=<tree>/python python3 bench_dsa_transform_index_decode.py
"""

import torch
import triton

from sglang.kernels.ops.attention.dsa.transform_index import (
    transform_index_page_table_decode_kernel,
    transform_index_page_table_decode_tiled_kernel,
)

CONTEXT = 65536
BLOCK_TOPK = 256
ROTATE = 6


def make_inputs(bs, topk, device):
    page_table = torch.randint(
        0, CONTEXT, (bs, CONTEXT), dtype=torch.int32, device=device
    )
    topk_indices = torch.randint(
        0, CONTEXT, (bs, topk), dtype=torch.int32, device=device
    )
    # ~10% of slots unselected, the -1 the kernels have to preserve
    topk_indices[torch.rand_like(topk_indices, dtype=torch.float32) < 0.1] = -1
    return page_table, topk_indices


def run_orig(page_table, topk_indices, result):
    transform_index_page_table_decode_kernel[(topk_indices.shape[0],)](
        page_table, topk_indices, result, 1, page_table_row_stride=page_table.stride(0)
    )


def run_tiled(page_table, topk_indices, result):
    topk = topk_indices.shape[1]
    transform_index_page_table_decode_tiled_kernel[
        (topk_indices.shape[0], triton.cdiv(topk, BLOCK_TOPK))
    ](
        page_table,
        topk_indices,
        result,
        page_table.stride(0),
        topk_indices.stride(0),
        topk_indices.stride(1),
        result.stride(0),
        result.stride(1),
        TOPK=topk,
        BLOCK_TOPK=BLOCK_TOPK,
        num_warps=4,
    )


def bench(fn, sets, results):
    def one():
        for (pt, ti), res in zip(sets, results):
            fn(pt, ti, res)

    return triton.testing.do_bench(one, warmup=50, rep=200) * 1000 / len(sets)


def main():
    dev = "cuda"
    pt, ti = make_inputs(8, 2048, dev)
    a, b = (torch.empty_like(ti) for _ in range(2))
    run_orig(pt, ti, a)
    run_tiled(pt, ti, b)
    torch.cuda.synchronize()
    assert torch.equal(a, b), "tiled kernel does not match the 2048 kernel"
    print("equality at width 2048: OK\n")

    print(f"{'batch':>6} {'width':>6} {'orig us':>10} {'tiled us':>10} {'tiled/orig':>11}")
    for topk in (2048, 2051):
        for bs in (1, 4, 16, 32, 64, 128):
            sets = [make_inputs(bs, topk, dev) for _ in range(ROTATE)]
            results = [torch.empty_like(ti) for _, ti in sets]
            t_tiled = bench(run_tiled, sets, results)
            if topk == 2048:
                t_orig = bench(run_orig, sets, results)
                ratio = f"{t_tiled / t_orig:.2f}x"
                print(f"{bs:>6} {topk:>6} {t_orig:>10.1f} {t_tiled:>10.1f} {ratio:>11}")
            else:
                print(f"{bs:>6} {topk:>6} {'n/a':>10} {t_tiled:>10.1f} {'—':>11}")


if __name__ == "__main__":
    main()
