#!/usr/bin/env bash
# Kill a stuck SGLang server inside a container, in the order that actually works.
#
#   NRUN_JOBID=<jobid> kill_sglang.sh <container> [port]
#
# Runs the in-container half from a FILE, never from `bash -c "<script text>"`:
# a pattern passed on the command line lands in the shell's own /proc cmdline,
# so `pkill -f` matches the shell and the exec dies before the cleanup finishes.
# For the same reason the GLM.sh pattern is bracketed (GLM[.]sh) and the engine
# is matched on comm (no -f), whose value for this shell is just "bash".
#
# Order matters: GLM.sh has a config loop and a server-restart retry, so killing
# the engine first just makes it launch a new one. Orchestrator, then engine.
set -uo pipefail

CONTAINER="${1:?usage: kill_sglang.sh <container> [port]}"
PORT="${2:-8234}"
HELPER="/home/jacchang/SGLang-benchmarks/tools/.kill_sglang_inner.sh"

cat > "$HELPER" <<'EOS'
#!/usr/bin/env bash
port="${1:-8234}"
pkill -9 -f "GLM[.]sh"      || true   # orchestrator first
pkill -9 -f "bench[_]serving" || true
pkill -9 -f "sgl[_]eval"    || true
pkill -9 python3            || true   # then the engine, by comm
pkill -9 sglang             || true
sleep 8
if ss -ltn 2>/dev/null | grep -q ":${port}\b"; then
    echo "WARNING: port ${port} still held (host networking -- check other containers)"
else
    echo "port ${port} free"
fi
pgrep -c sglang >/dev/null 2>&1 && echo "sglang procs left: $(pgrep -c sglang)" || echo "no sglang procs"
EOS
chmod +x "$HELPER"

NRUN_JOBID="${NRUN_JOBID:-}" ~/SGLang-benchmarks/tools/nrun.sh \
    "docker exec ${CONTAINER} bash ${HELPER} ${PORT}"
