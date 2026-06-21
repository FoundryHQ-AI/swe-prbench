#!/usr/bin/env bash
# Stop every Claude Code CLI eval process started by this fork. Safe to run
# multiple times. Used by the rate-limit watchdog and for manual pause.
set -u
killed=0
for pat in 'eval_harness/run_eval.py' 'run_single_agent.sh' 'claude -p --model claude-opus'; do
  pids=$(pgrep -f "${pat}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "stop_all: killing pids ($pat): ${pids}"
    kill ${pids} 2>/dev/null || true
    killed=1
  fi
done
sleep 2
# Force-kill any survivors
for pat in 'eval_harness/run_eval.py' 'run_single_agent.sh' 'claude -p --model claude-opus'; do
  pids=$(pgrep -f "${pat}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "stop_all: force-killing pids ($pat): ${pids}"
    kill -9 ${pids} 2>/dev/null || true
  fi
done
if [[ ${killed} -eq 0 ]]; then
  echo "stop_all: nothing to kill"
fi
