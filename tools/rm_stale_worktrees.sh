#!/bin/bash
# Delete deregistered worktree directories under ~/PR (root-owned __pycache__ removed via the container).
# usage: rm_stale_worktrees.sh <container> <dir>...
container=$1; shift
cd ~/PR || exit 1
reg=$( (git -C ~/PR/sglang worktree list --porcelain; git -C ~/PR/aiter worktree list --porcelain) | sed -n "s|^worktree ||p")
for d in "$@"; do
  if echo "$reg" | grep -qx "$PWD/$d"; then echo "registered, skip $d"; continue; fi
  docker exec "$container" rm -rf "$PWD/$d" >/dev/null 2>&1
  if [ -e "$d" ]; then echo "still exists: $d"; else echo "rm $d ok"; fi
done
