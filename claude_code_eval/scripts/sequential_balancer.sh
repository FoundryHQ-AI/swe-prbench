#!/usr/bin/env bash
# Sequential per-record balancer.
#
# Inner loop:
#   1. Pick whichever agent has the fewest eval_results files (the laggard).
#   2. Launch run_single_agent.sh for it in the background.
#   3. Wait until its eval_results count grows by at least 1.
#   4. Kill the agent.
#   5. Repeat (re-picking the laggard).
#
# Exits cleanly when its parent (e.g. resume_when_reset.sh) sends SIGTERM,
# or when ./claude_code_eval/results/STOP_BALANCER exists, or when all 4
# agents are at 300/300.
#
# Why: under a per-window token budget, fanning out to 4 parallel agents
# splits the budget 4 ways. Running one agent at a time lets that agent
# use 100% of the per-claude-cli stream. Re-picking after every record
# keeps the per-agent counts within 1 of each other so the cross-agent
# comparison stays fair as the experiment progresses.
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source .venv/bin/activate

ALL_AGENTS=(opus_4_8_low opus_4_8_medium opus_4_8_max opus_4_7_max)
MAX_PER_AGENT=300
STOP_FLAG="${REPO_ROOT}/claude_code_eval/results/STOP_BALANCER"
mkdir -p "$(dirname "${STOP_FLAG}")"
rm -f "${STOP_FLAG}"

count_for() {
  find "claude_code_eval/results/runs/${1}__judge_claude_sonnet_46_judge/eval_results" \
    -maxdepth 1 -type f 2>/dev/null | wc -l | tr -d ' '
}

cleanup() {
  echo "[$(date '+%H:%M:%S')] balancer: cleanup — killing child run_single_agent / claude -p / run_eval.py"
  pkill -f "run_single_agent.sh" 2>/dev/null || true
  pkill -f "eval_harness/run_eval.py" 2>/dev/null || true
  pkill -f "claude -p --model claude-opus" 2>/dev/null || true
  pkill -f "claude -p --model claude-sonnet" 2>/dev/null || true
  exit 0
}
trap cleanup SIGTERM SIGINT

echo "[$(date '+%H:%M:%S')] balancer: starting sequential per-record mode"

while true; do
  if [[ -f "${STOP_FLAG}" ]]; then
    echo "[$(date '+%H:%M:%S')] balancer: STOP_BALANCER flag present — exiting"
    cleanup
  fi

  # Find the laggard. Break ties by ALL_AGENTS order. If everyone is at
  # MAX_PER_AGENT we're done.
  LAGGARD=""
  LAGGARD_COUNT=999999999
  TOTAL=0
  for a in "${ALL_AGENTS[@]}"; do
    c=$(count_for "${a}")
    TOTAL=$((TOTAL + c))
    if [[ ${c} -lt ${LAGGARD_COUNT} && ${c} -lt ${MAX_PER_AGENT} ]]; then
      LAGGARD_COUNT=${c}
      LAGGARD=${a}
    fi
  done

  if [[ -z "${LAGGARD}" ]]; then
    echo "[$(date '+%H:%M:%S')] balancer: every agent at ${MAX_PER_AGENT}/${MAX_PER_AGENT} — done"
    exit 0
  fi

  START_COUNT=${LAGGARD_COUNT}
  echo "[$(date '+%H:%M:%S')] balancer: pick ${LAGGARD} (was ${START_COUNT}/${MAX_PER_AGENT}, total=${TOTAL})"

  CONCURRENCY=2 bash claude_code_eval/scripts/run_single_agent.sh "${LAGGARD}" \
    >> "claude_code_eval/results/logs/${LAGGARD}.log" 2>&1 &
  AGENT_PGID=$!
  disown ${AGENT_PGID} 2>/dev/null || true

  # Poll for a new record. Cap each cycle at 30 min so a stuck agent
  # doesn't hold the whole window — kill and retry the laggard pick.
  CYCLE_DEADLINE=$(( $(date +%s) + 1800 ))
  while true; do
    sleep 30
    if [[ -f "${STOP_FLAG}" ]]; then
      cleanup
    fi
    CUR=$(count_for "${LAGGARD}")
    if [[ ${CUR} -gt ${START_COUNT} ]]; then
      echo "[$(date '+%H:%M:%S')] balancer: ${LAGGARD} ${START_COUNT} -> ${CUR}; killing and re-picking"
      break
    fi
    if [[ $(date +%s) -ge ${CYCLE_DEADLINE} ]]; then
      echo "[$(date '+%H:%M:%S')] balancer: ${LAGGARD} stalled for 30min — killing and re-picking"
      break
    fi
  done

  # Kill the agent we spawned (and its descendants). pkill the patterns
  # used by run_single_agent.sh and its claude -p children.
  pkill -P ${AGENT_PGID} 2>/dev/null || true
  pkill -f "run_single_agent.sh" 2>/dev/null || true
  pkill -f "eval_harness/run_eval.py" 2>/dev/null || true
  pkill -f "claude -p --model" 2>/dev/null || true
  sleep 2
done
