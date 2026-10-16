#!/bin/bash
# Drain the run queue on this machine, one run at a time, under caffeinate, respecting power and thermals.
#   scripts/run-queue.sh            run until the queue is empty
#   MAX_RUNS=3 scripts/run-queue.sh run at most 3 jobs
#   touch STOP                      finish the current run, then stop (file is removed on exit)
#   Ctrl-C                          stop now; the run checkpoints and goes back to pending, resume by rerunning
#   ALLOW_BATTERY=1                 override the AC-power check
set -u
cd "$(dirname "$0")/.."
LOCK=queue/.lock
if [ -e "$LOCK" ] && kill -0 "$(cut -d' ' -f1 "$LOCK")" 2>/dev/null; then
  echo "another run-queue loop is active on this machine: $(cat "$LOCK")"; exit 1
fi
mkdir -p queue
echo "$$ $(hostname) $(date -Iseconds)" > "$LOCK"
trap 'rm -f "$LOCK" STOP' EXIT
trap 'echo; echo "interrupted: current run checkpointed and returned to pending; rerun scripts/run-queue.sh to resume"; exit 130' INT
MAX_RUNS=${MAX_RUNS:-1000000}
n=0
while [ "$n" -lt "$MAX_RUNS" ]; do
  [ -e STOP ] && { echo "STOP file present, exiting"; break; }
  if [ -z "${ALLOW_BATTERY:-}" ] && ! pmset -g batt | grep -q "AC Power"; then
    echo "on battery; waiting for AC power (set ALLOW_BATTERY=1 to override)"; sleep 120; continue
  fi
  lim=$(pmset -g therm 2>/dev/null | awk '/CPU_Speed_Limit/{print $NF}')
  if [ -n "${lim:-}" ] && [ "$lim" -lt 80 ]; then
    echo "thermal throttling (CPU_Speed_Limit=$lim); cooling off for 5 min"; sleep 300; continue
  fi
  caffeinate -dimsu .venv/bin/python -m lmcp queue next
  rc=$?
  if [ "$rc" -eq 3 ]; then echo "queue empty"; break; fi
  n=$((n+1))
done
echo "run-queue finished after $n run(s)"
