#!/usr/bin/env bash
# Run a short command on the compute node of a spur/slurm job.
#
# The spur `srun` client on this cluster intermittently hangs before the step
# ever starts (no output, `timeout` inside the step never fires because the step
# does not exist yet). Roughly every other invocation. So bound each attempt and
# retry rather than waiting forever.
#
# Only for SHORT commands -- long work should be launched with `docker exec -d`
# and polled through a log file on the shared home.
#
#   ~/SGLang-benchmarks/tools/nrun.sh 'docker ps --format "{{.Names}}"'
#   NRUN_JOBID=147020 NRUN_TIMEOUT=45 ~/SGLang-benchmarks/tools/nrun.sh '...'
set -uo pipefail

for i in $(seq 1 "${NRUN_TRIES:-5}"); do
    # Retry only on a hung srun client (timeout -> 124). A command that ran and
    # returned nonzero must be reported, not re-run: `pgrep -c` returning 1 for
    # "no matches" is a normal answer, and re-running it looped this script.
    timeout "${NRUN_TIMEOUT:-40}" srun --jobid="${NRUN_JOBID:-147020}" --overlap \
        bash -c "$*" </dev/null 2>&1 | grep -v '^spur: warning: raw mode'
    rc=${PIPESTATUS[0]}
    [ "$rc" -eq 124 ] || exit "$rc"
    echo "[nrun] attempt ${i}: srun hung (no step started); retrying" >&2
done
echo "[nrun] giving up after ${NRUN_TRIES:-5} attempts" >&2
exit 1
