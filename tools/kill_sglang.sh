#!/usr/bin/env bash
# Kill every SGLang process in a benchmark container and wait for the port to
# be released. Use this instead of ad-hoc pkill -- see the "Killing a stuck
# SGLang server" section of the sglang-benchmark skill for why the obvious
# commands leave processes behind.
#
#   ./kill_sglang.sh                        # container from $SGLANG_CONTAINER, port 8234
#   ./kill_sglang.sh jacchang_GLM5_0728_    # explicit container
#   ./kill_sglang.sh jacchang_GLM5_0728_ 8552
#   ./kill_sglang.sh --here                 # already inside the container
set -uo pipefail

CONTAINER="${1:-${SGLANG_CONTAINER:-}}"
PORT="${2:-${SGLANG_PORT:-8234}}"

if [ -z "$CONTAINER" ]; then
    echo "usage: $0 <container|--here> [port]" >&2
    echo "   or: SGLANG_CONTAINER=<name> $0" >&2
    exit 2
fi

# The kill sequence is fed to the remote shell over STDIN (`bash -s`), never as
# an argument to `bash -c`. With `bash -c "$script"` the entire script text --
# including comments -- lands in the shell's own /proc/<pid>/cmdline, so every
# `pkill -f <literal>` below would match the shell itself and SIGKILL the
# cleanup halfway through. Over stdin the cmdline is just "bash -s -- <port>".
emit_kill_script() {
    cat <<'EOS'
set -u
port="$1"

# 1. The orchestrator FIRST. GLM.sh has a config loop and a server-restart
#    retry, so killing the engine while GLM.sh lives just makes it launch a new
#    server a few seconds later. Its comm is "bash", so -f is unavoidable here.
pkill -9 -f "GLM[.]sh"
pkill -9 -f "bench[_]serving"

# 2. The engine, matched on comm (no -f) so pkill can never match itself.
#    "python3 -m sglang.launch_server" has comm=python3; the workers are
#    renamed by setproctitle to sglang::scheduler_TPn / sglang::tokenizer_worker
#    / sglang::detokenizer, which comm truncates to 15 chars. Neither group
#    matches the other, so both lines are required.
pkill -9 python3
pkill -9 sglang

# 3. Wait on the PORT, not on the process list. The listening socket is held by
#    a tokenizer_worker, so an empty `pgrep -f launch_server` proves nothing.
for i in $(seq 1 30); do
    if ! ss -ltn 2>/dev/null | grep -q ":${port}\b"; then
        echo "port ${port} free after ${i}s"
        exit 0
    fi
    sleep 1
done

echo "ERROR: port ${port} still held after 30s" >&2
ss -ltnp 2>/dev/null | grep ":${port}\b" >&2
ps -eo pid,comm,args --no-headers | grep -iE "sglang|GLM" | grep -v grep >&2
exit 1
EOS
}

if [ "$CONTAINER" = "--here" ]; then
    emit_kill_script | bash -s -- "$PORT"
else
    emit_kill_script | docker exec -i "$CONTAINER" bash -s -- "$PORT"
fi
