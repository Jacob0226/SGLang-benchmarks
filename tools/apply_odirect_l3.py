#!/usr/bin/env python3
"""Patch SGLang's HiCacheFile backend to use O_DIRECT (bypass page cache) for the
L3 file read/write path, so a RAID0 /raid can reach its real bandwidth.

Run INSIDE the container:
    python3 tools/apply_odirect_l3.py            # apply
    python3 tools/apply_odirect_l3.py --restore  # revert

Patches the v1 set()/get() (batch_set/batch_get call into them). After applying,
ALWAYS validate correctness with a GSM8K run — a wrong read returns garbage KV.
"""
import os
import sys
import sglang.srt.mem_cache.hicache_storage as _h

DST = _h.__file__
BAK = DST + ".odirect.bak"

OLD_SET = '''            value.contiguous().view(dtype=torch.uint8).numpy().tofile(tensor_path)
            return True'''

NEW_SET = '''            import numpy as _np
            _src = value.contiguous().view(dtype=torch.uint8).numpy().reshape(-1)
            _n = _src.nbytes
            _pad = (-_n) % 4096
            _raw = _np.empty(_n + _pad + 4096, dtype=_np.uint8)
            _o = (-_raw.ctypes.data) % 4096
            _buf = _raw[_o:_o + _n + _pad]
            _buf[:_n] = _src
            if _pad:
                _buf[_n:] = 0
            _fd = os.open(tensor_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_DIRECT, 0o644)
            try:
                _mv = memoryview(_buf)
                _w = 0
                while _w < len(_mv):
                    _w += os.write(_fd, _mv[_w:])
            finally:
                os.close(_fd)
            return True'''

OLD_GET = '''            expected = target_location.numel() * target_location.element_size()
            with open(tensor_path, "rb", buffering=0) as f:
                buf = memoryview(target_location.view(torch.uint8).contiguous().numpy())
                if f.readinto(buf) != expected:
                    raise IOError(f"Short read for {key}")
            return target_location'''

NEW_GET = '''            import numpy as _np
            _dst = target_location.view(torch.uint8).contiguous().numpy().reshape(-1)
            _n = _dst.nbytes
            _pad = (-_n) % 4096
            _raw = _np.empty(_n + _pad + 4096, dtype=_np.uint8)
            _o = (-_raw.ctypes.data) % 4096
            _buf = _raw[_o:_o + _n + _pad]
            _fd = os.open(tensor_path, os.O_RDONLY | os.O_DIRECT)
            try:
                _mv = memoryview(_buf)
                _r = 0
                while _r < len(_mv):
                    _k = os.preadv(_fd, [_mv[_r:]], _r)
                    if _k == 0:
                        break
                    _r += _k
            finally:
                os.close(_fd)
            _dst[:] = _buf[:_n]
            return target_location'''


def main():
    restore = "--restore" in sys.argv
    if restore:
        if os.path.exists(BAK):
            with open(BAK) as f:
                src = f.read()
            with open(DST, "w") as f:
                f.write(src)
            print(f"restored original {DST}")
        else:
            print(f"no backup at {BAK}")
        return

    with open(DST) as f:
        src = f.read()
    if "O_DIRECT" in src:
        print("already patched (O_DIRECT present). use --restore first to re-apply.")
        return
    # WRITE-ONLY: only the backup (set) path uses O_DIRECT. Reads (get) stay
    # buffered so prefetch keeps page-cache reuse across repeated L3 reads.
    if OLD_SET not in src:
        print("ERROR: expected set() body not found — sglang version changed?")
        sys.exit(1)
    if not os.path.exists(BAK):
        with open(BAK, "w") as f:
            f.write(src)
    src = src.replace(OLD_SET, NEW_SET)
    with open(DST, "w") as f:
        f.write(src)
    print(f"applied O_DIRECT (write-only) patch to {DST} (backup at {BAK})")
    print("NEXT: restart the server and run GSM8K to validate correctness.")


if __name__ == "__main__":
    main()
