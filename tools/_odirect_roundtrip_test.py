import os, shutil, tempfile
import torch
from sglang.srt.mem_cache.hicache_storage import HiCacheFile, HiCacheStorageConfig

d = tempfile.mkdtemp(prefix="odtest_", dir="/home/jacchang")
try:
    cfg = HiCacheStorageConfig(tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
                               model_name="m", is_mla_model=True,
                               attn_cp_rank=0, attn_cp_size=1,
                               enable_storage_metrics=False,
                               is_page_first_layout=True)
    fb = HiCacheFile(cfg, file_path=d)
    allok = True
    for dt in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn):
        for n in (576, 1152, 4096, 70000, 36864):
            if dt == torch.uint8:
                src = torch.randint(0, 255, (n,), dtype=torch.uint8)
            else:
                src = torch.randint(0, 255, (n,), dtype=torch.uint8).view(dt)
            src = src.contiguous()
            k = "k_%s_%d" % (str(dt), n)
            assert fb.set(k, src), "set failed"
            dst = torch.empty_like(src)
            fb.get(k, dst)
            ok = bool((dst.view(torch.uint8) == src.view(torch.uint8)).all())
            allok = allok and ok
            status = "OK" if ok else "FAIL"
            print("%-22s n=%6d bytes=%7d roundtrip=%s" %
                  (str(dt), n, src.view(torch.uint8).numel(), status))
    print("ALL ROUNDTRIPS OK" if allok else "SOME FAILED")
finally:
    shutil.rmtree(d, ignore_errors=True)
