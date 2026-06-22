#!/usr/bin/env bash
# Stop every Claude Code CLI eval process started by this fork. Safe to run
# multiple times. Used by the rate-limit watchdog and for manual pause.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# Tell the sequential balancer to exit cleanly on its next 30s tick.
mkdir -p "${REPO_ROOT}/claude_code_eval/results"
touch "${REPO_ROOT}/claude_code_eval/results/STOP_BALANCER"

killed=0
PATTERNS=(
  'sequential_balancer.sh'
  'eval_harness/run_eval.py'
  'run_single_agent.sh'
  'claude -p --model claude-opus'
  'claude -p --model claude-sonnet'
)
for pat in "${PATTERNS[@]}"; do
  pids=$(pgrep -f "${pat}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "stop_all: killing pids ($pat): ${pids}"
    kill ${pids} 2>/dev/null || true
    killed=1
  fi
done
sleep 2
# Force-kill any survivors
for pat in "${PATTERNS[@]}"; do
  pids=$(pgrep -f "${pat}" 2>/dev/null || true)
  if [[ -n "${pids}" ]]; then
    echo "stop_all: force-killing pids ($pat): ${pids}"
    kill -9 ${pids} 2>/dev/null || true
  fi
done
if [[ ${killed} -eq 0 ]]; then
  echo "stop_all: nothing to kill"
fi
