import torch
import sgl_kernel  # noqa
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer_mla as jit_one,
    transfer_hicache_all_layer_mla as jit_all,
)

ops = torch.ops.sgl_kernel
item = 576           # FP8 MLA element bytes
L, dev_tok, host_tok, ntok = 4, 2048, 4096, 777
bq, nw = 2, 16

g = torch.Generator(); g.manual_seed(0)
didx = torch.randperm(dev_tok, generator=g)[:ntok].cuda().long()
hidx = torch.randperm(host_tok, generator=g)[:ntok].cuda().long()

def rand_dev(n):
    return torch.randint(0, 255, (n, item), dtype=torch.uint8, device="cuda")

# ---- one-layer load lf->lf: dst[didx] = src[didx] ----
src = rand_dev(dev_tok)
dst_aot = torch.zeros_like(src); dst_jit = torch.zeros_like(src)
ops.transfer_kv_per_layer_mla.default(src, dst_aot, didx, didx, item, bq, nw)
jit_one(cache_dst=dst_jit, indices_dst=didx, cache_src=src, indices_src=didx, element_dim=item)
ref = torch.zeros_like(src); ref[didx] = src[didx]
print("one-layer load lf->lf : JIT==ref", bool((dst_jit==ref).all()), " AOT==ref", bool((dst_aot==ref).all()))

# ---- one-layer load pf->lf (strided host src) ----
host = torch.randint(0, 255, (host_tok, L, item), dtype=torch.uint8, device="cuda")
dst_jit2 = rand_dev(dev_tok).zero_()
li = 2
jit_one(cache_dst=dst_jit2, indices_dst=didx, cache_src=host[:, li, :], indices_src=hidx, element_dim=item)
ref2 = torch.zeros_like(dst_jit2); ref2[didx] = host[hidx, li, :]
print("one-layer load pf->lf : JIT==ref", bool((dst_jit2==ref2).all()))

# ---- all-layer backup lf->lf via ptr tables ----
devs = [rand_dev(dev_tok) for _ in range(L)]
devs2 = [torch.zeros_like(d) for d in devs]
dptr = torch.tensor([d.data_ptr() for d in devs], dtype=torch.uint64, device="cuda")
dptr2 = torch.tensor([d.data_ptr() for d in devs2], dtype=torch.uint64, device="cuda")
jit_all(ptr_dst=dptr2, indices_dst=didx, ptr_src=dptr, indices_src=didx,
        cache_src_stride_bytes=item, cache_dst_stride_bytes=item, element_size=item)
ok_all = all(bool((devs2[l][didx]==devs[l][didx]).all()) for l in range(L))
print("all-layer backup lf->lf: JIT==ref", ok_all)

print("RESULT:", "ALL CORRECT" if all([
    (dst_jit==ref).all(), (dst_jit2==ref2).all(), ok_all]) else "MISMATCH")
