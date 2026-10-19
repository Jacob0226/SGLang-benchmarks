#!/usr/bin/env bash
# Kill one SGLang server (launcher + all descendants) by its --port, leaving others alone.
# Run inside the container: docker exec <container> bash kill_sglang_port.sh <port>
set -uo pipefail
PORT=${1:?usage: kill_sglang_port.sh <port>}

root=$(pgrep -f "launch_server.*--port ${PORT}( |$)" | head -1)
[ -z "$root" ] && { echo "no server on port $PORT"; exit 0; }

tree() { echo "$1"; for c in $(pgrep -P "$1"); do tree "$c"; done; }
pids=$(tree "$root")
kill -9 $pids 2>/dev/null
sleep 8
if ss -ltn 2>/dev/null | grep -q ":${PORT}\b"; then
    echo "WARNING: port ${PORT} still held"
else
    echo "killed $(echo $pids | wc -w) procs; port ${PORT} free"
fi
