"""Per-batch torch peak-memory logger for SGLang scheduler debugging.

Env:
  SGLANG_MEMDBG=1            log every new peak (>= SGLANG_MEMDBG_STEP_MB above the last)
  SGLANG_MEMDBG_HISTORY=1    record allocator history and dump a snapshot on each new peak
  SGLANG_MEMDBG_DIR=...      snapshot directory (default /tmp/memdbg)
"""

import logging
import os
import pickle

import torch

logger = logging.getLogger(__name__)

_ENABLED = os.environ.get("SGLANG_MEMDBG") == "1"
_HISTORY = os.environ.get("SGLANG_MEMDBG_HISTORY") == "1"
_DIR = os.environ.get("SGLANG_MEMDBG_DIR", "/tmp/memdbg")
_STEP = int(os.environ.get("SGLANG_MEMDBG_STEP_MB", "256")) << 20
_MAX_DUMPS = int(os.environ.get("SGLANG_MEMDBG_MAX_DUMPS", "8"))

_state = {"peak": 0, "dumps": 0, "recording": False, "n": 0}


def _lens(v):
    if v is None:
        return []
    if torch.is_tensor(v):
        return v.tolist()
    return list(v)


def _describe(batch):
    prefix = _lens(batch.prefix_lens)
    ext = _lens(batch.extend_lens)
    return (
        f"mode={batch.forward_mode.name} bs={batch.batch_size()} "
        f"ext={batch.extend_num_tokens} prefix_sum={sum(prefix)} "
        f"prefix_max={max(prefix, default=0)} ext_max={max(ext, default=0)}"
    )


def after_batch(batch, tp_rank: int):
    if not _ENABLED:
        return
    try:
        _after_batch(batch, tp_rank)
    except Exception:
        logger.exception("[memdbg] failed; disabling")
        globals()["_ENABLED"] = False


def _after_batch(batch, tp_rank: int):
    _state["n"] += 1
    if _HISTORY and not _state["recording"]:
        torch.cuda.memory._record_memory_history(max_entries=400000)
        _state["recording"] = True
        torch.cuda.reset_peak_memory_stats()
        return
    peak = torch.cuda.max_memory_allocated()
    if peak < _state["peak"] + _STEP:
        return
    _state["peak"] = peak
    free, _ = torch.cuda.mem_get_info()
    logger.info(
        f"[memdbg] tp{tp_rank} batch#{_state['n']} new peak "
        f"alloc={peak / 2**30:.2f}G cur={torch.cuda.memory_allocated() / 2**30:.2f}G "
        f"reserved={torch.cuda.memory_reserved() / 2**30:.2f}G "
        f"free={free / 2**30:.2f}G {_describe(batch)}"
    )
    if _HISTORY and tp_rank == 0 and _state["dumps"] < _MAX_DUMPS:
        os.makedirs(_DIR, exist_ok=True)
        path = os.path.join(_DIR, f"peak_{_state['dumps']:02d}_b{_state['n']}.pickle")
        with open(path, "wb") as f:
            pickle.dump(torch.cuda.memory._snapshot(), f)
        _state["dumps"] += 1
        logger.info(f"[memdbg] snapshot -> {path}")
