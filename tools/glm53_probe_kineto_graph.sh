#!/usr/bin/env bash
# Is there a knob that makes roctracer/kineto report HIP graph replays?
#
# ROCm graph-ON decode captures exactly one forward no matter how many steps are
# requested, while the same shape captures all of them with --disable-cuda-graph
# and B200 captures all of them with graphs on. If the missing records are a
# disabled activity domain rather than a hard limit, there may be a flag.
#
# Output: ~/kineto_graph_probe.txt
exec > /home/jacchang/kineto_graph_probe.txt 2>&1

echo "=== torch / kineto build ==="
python3 - <<'PY'
import torch
print("torch", torch.__version__, "| hip", torch.version.hip)
try:
    print("kineto available:", torch.profiler.kineto_available())
except Exception as e:
    print("kineto_available:", e)
try:
    from torch.profiler import _ExperimentalConfig
    import inspect
    print("_ExperimentalConfig signature:")
    print("   ", inspect.signature(_ExperimentalConfig.__init__))
except Exception as e:
    print("_ExperimentalConfig:", e)
try:
    from torch.autograd.profiler import _supported_activities
    print("supported activities:", _supported_activities())
except Exception as e:
    print("supported activities:", e)
PY

echo
echo "=== anything graph-related in torch's profiler/kineto python surface ==="
python3 - <<'PY'
import os, torch, re
root = os.path.dirname(torch.__file__)
pats = re.compile(r"graph", re.I)
hits = 0
for sub in ("profiler", "autograd/profiler.py", "_C/_profiler.pyi"):
    p = os.path.join(root, sub)
    files = []
    if os.path.isdir(p):
        for dp, _, fns in os.walk(p):
            files += [os.path.join(dp, f) for f in fns if f.endswith((".py", ".pyi"))]
    elif os.path.isfile(p):
        files = [p]
    for f in files:
        try:
            for i, line in enumerate(open(f, errors="ignore"), 1):
                if pats.search(line) and re.search(r"cuda_?graph|hip_?graph|graph_trace|GRAPH", line):
                    print(f"  {f.replace(root,'torch')}:{i}: {line.strip()[:110]}")
                    hits += 1
        except Exception:
            pass
print(f"  ({hits} hits)")
PY

echo
echo "=== kineto / roctracer env knobs visible in the shared objects ==="
for so in $(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib/libtorch_cpu.so \
          /opt/rocm/lib/libroctracer64.so /opt/rocm/lib/librocprofiler64.so; do
    [ -f "$so" ] || continue
    echo "--- $so"
    strings "$so" 2>/dev/null | grep -aoE '\b(KINETO|LIBKINETO|ROCTRACER|ROCP|HSA|AMD_LOG)[A-Z0-9_]*' \
        | sort -u | head -25
done

echo
echo "=== does kineto mention graph activity for ROCm? ==="
so=$(python3 -c "import torch,os;print(os.path.dirname(torch.__file__))")/lib/libtorch_cpu.so
strings "$so" 2>/dev/null | grep -aiE 'graph.*(launch|replay|exec|trace)|hipGraph' | sort -u | head -20
