"""Per-window GPU time split of scheduler batches (prefill vs decode).

Times each run_batch forward with CUDA events on the current stream and reads
them back once they complete, so it never synchronizes the overlap scheduler.

Env:
  SGLANG_STEPTIME=1           enable (tp rank 0 only)
  SGLANG_STEPTIME_WINDOW=60   seconds per summary line
  SGLANG_STEPTIME_FILE=...    jsonl output (default /tmp/steptime.jsonl)
"""

import json
import logging
import os
import time
from collections import deque

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_STEPTIME") == "1"
_WINDOW = float(os.environ.get("SGLANG_STEPTIME_WINDOW", "60"))
_FILE = os.environ.get("SGLANG_STEPTIME_FILE", "/tmp/steptime.jsonl")

_PREFIX_BUCKETS = (0, 32 << 10, 128 << 10, 512 << 10, 2 << 20, 8 << 20)

_pending = deque()
_path = {"value": None}


def _new_window():
    return {
        "t0": time.monotonic(),
        "decode": {"n": 0, "ms": 0.0, "bs": 0},
        "extend": {"n": 0, "ms": 0.0, "tokens": 0, "prefix": 0, "seqs": 0},
        "other": {"n": 0, "ms": 0.0},
        "paths": {},
        "prefix_buckets": {},
    }


_win = _new_window()


def note_prefill_path(use_mha: bool, owned: bool, split_indexer: bool):
    if not _ENABLED:
        return
    _path["value"] = "mha" if use_mha else "owned" if owned else (
        "gathered_split" if split_indexer else "gathered"
    )


_mem = {"t0": 0.0}


def _mem_tick(tp_rank: int):
    now = time.monotonic()
    if now - _mem["t0"] < _WINDOW:
        return
    _mem["t0"] = now
    free, total = torch.cuda.mem_get_info()
    reserved = torch.cuda.memory_reserved()
    logger.info(
        f"[steptime-mem] tp{tp_rank} nontorch={(total - free - reserved) / 2**30:.2f}G "
        f"reserved={reserved / 2**30:.2f}G free={free / 2**30:.2f}G"
    )


def _describe(batch):
    # Read before the forward: the MTP worker rewrites forward_mode and
    # extend_num_tokens on the batch for its draft extend.
    mode = batch.forward_mode
    if mode.is_decode() or mode.is_target_verify():
        return ("decode", batch.batch_size(), 0, 0)
    if mode.is_extend():
        prefix = batch.prefix_lens
        prefix = int(sum(prefix.tolist() if torch.is_tensor(prefix) else prefix or []))
        return ("extend", batch.batch_size(), int(batch.extend_num_tokens or 0), prefix)
    return ("other", batch.batch_size(), 0, 0)


def before_batch(batch, tp_rank: int):
    if not _ENABLED:
        return None
    _mem_tick(tp_rank)
    if tp_rank != 0:
        return None
    ev = torch.cuda.Event(enable_timing=True)
    ev.record()
    _path["value"] = None
    return ev, _describe(batch)


def after_batch(tp_rank: int, token):
    if token is None:
        return
    try:
        _after_batch(*token)
    except Exception:
        logger.exception("[steptime] failed; disabling")
        globals()["_ENABLED"] = False


def _bucket(prefix_tokens: int) -> str:
    label = "0"
    for b in _PREFIX_BUCKETS:
        if prefix_tokens >= b:
            label = f">={b >> 10}k"
    return label


def _after_batch(start_ev, desc):
    end_ev = torch.cuda.Event(enable_timing=True)
    end_ev.record()
    kind = desc[0]
    meta = (*desc, _path["value"] if kind == "extend" else None)
    _pending.append((start_ev, end_ev, meta))

    while _pending and _pending[0][1].query():
        s, e, (kind, bs, tokens, prefix, path) = _pending.popleft()
        ms = s.elapsed_time(e)
        w = _win[kind]
        w["n"] += 1
        w["ms"] += ms
        if kind == "decode":
            w["bs"] += bs
        elif kind == "extend":
            w["tokens"] += tokens
            w["prefix"] += prefix
            w["seqs"] += bs
            key = path or "none"
            p = _win["paths"].setdefault(key, {"n": 0, "ms": 0.0, "tokens": 0})
            p["n"] += 1
            p["ms"] += ms
            p["tokens"] += tokens
            b = _win["prefix_buckets"].setdefault(_bucket(prefix), {"n": 0, "ms": 0.0, "tokens": 0})
            b["n"] += 1
            b["ms"] += ms
            b["tokens"] += tokens

    now = time.monotonic()
    if now - _win["t0"] >= _WINDOW:
        _flush(now)


def _flush(now):
    global _win
    out = dict(_win)
    out["wall_s"] = now - out.pop("t0")
    out["ts"] = time.time()
    with open(_FILE, "a") as f:
        f.write(json.dumps(out) + "\n")
    d, e = out["decode"], out["extend"]
    logger.info(
        f"[steptime] {out['wall_s']:.0f}s decode {d['ms'] / 1e3:.1f}s/{d['n']} "
        f"(bs {d['bs'] / max(d['n'], 1):.0f}) extend {e['ms'] / 1e3:.1f}s/{e['n']} "
        f"({e['tokens']} tok) paths={ {k: round(v['ms'] / 1e3, 1) for k, v in out['paths'].items()} }"
    )
    _win = _new_window()
