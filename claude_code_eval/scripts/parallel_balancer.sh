#!/usr/bin/env bash
# Two-slot parallel balancer.
#
# Maintains 2 independent slots; each slot runs one agent at a time. When
# a slot's agent completes a record, that slot is re-picked (most-behind
# agent excluding whatever the other slot is currently running).
#
# Why: once token budget stops being the bottleneck (multi-window
# burn-down + mid-run window resets), wall-clock is the limit. Running 2
# agents at once roughly doubles throughput without re-introducing the
# lopsided progress problem that 4-way parallel had.
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

# Pick the agent with the smallest record count, excluding the agent
# already running in the other slot (if any). Tie-break by ALL_AGENTS order.
# Echoes the agent name, or empty if no agent left to schedule.
pick_laggard() {
  local exclude="${1:-}"
  local laggard=""
  local laggard_count=999999999
  for a in "${ALL_AGENTS[@]}"; do
    [[ "${a}" == "${exclude}" ]] && continue
    local c
    c=$(count_for "${a}")
    if [[ ${c} -lt ${laggard_count} && ${c} -lt ${MAX_PER_AGENT} ]]; then
      laggard_count=${c}
      laggard=${a}
    fi
  done
  echo "${laggard}"
}

# Kill an agent process tree by PID.
kill_slot() {
  local pid="${1}"
  local agent="${2}"
  [[ -z "${pid}" || "${pid}" == "0" ]] && return
  pkill -P "${pid}" 2>/dev/null || true
  kill "${pid}" 2>/dev/null || true
  # Also kill any claude -p process whose argv contains this agent's model
  # tier, since run_single_agent's python child may have been orphaned.
  case "${agent}" in
    opus_4_8_*) pkill -f "claude -p --model claude-opus-4-8" 2>/dev/null || true ;;
    opus_4_7_*) pkill -f "claude -p --model claude-opus-4-7" 2>/dev/null || true ;;
  esac
}

launch_slot() {
  local agent="${1}"
  CONCURRENCY=2 bash claude_code_eval/scripts/run_single_agent.sh "${agent}" \
    >> "claude_code_eval/results/logs/${agent}.log" 2>&1 &
  echo $!
}

cleanup() {
  echo "[$(date '+%H:%M:%S')] balancer: cleanup — killing both slots"
  pkill -f "run_single_agent.sh" 2>/dev/null || true
  pkill -f "eval_harness/run_eval.py" 2>/dev/null || true
  pkill -f "claude -p --model claude-opus" 2>/dev/null || true
  pkill -f "claude -p --model claude-sonnet" 2>/dev/null || true
  exit 0
}
trap cleanup SIGTERM SIGINT

echo "[$(date '+%H:%M:%S')] balancer: starting 2-slot parallel mode"

slot1_agent=""; slot1_start=0; slot1_pid=0
slot2_agent=""; slot2_start=0; slot2_pid=0

while true; do
  [[ -f "${STOP_FLAG}" ]] && { echo "[$(date '+%H:%M:%S')] balancer: STOP_BALANCER flag — exiting"; cleanup; }

  # Ensure slot1 is occupied.
  if [[ -z "${slot1_agent}" ]]; then
    pick=$(pick_laggard "${slot2_agent}")
    if [[ -n "${pick}" ]]; then
      slot1_agent="${pick}"
      slot1_start=$(count_for "${slot1_agent}")
      slot1_pid=$(launch_slot "${slot1_agent}")
      echo "[$(date '+%H:%M:%S')] balancer: slot1 pick ${slot1_agent} (was ${slot1_start}/${MAX_PER_AGENT}, pid=${slot1_pid})"
    fi
  fi

  # Ensure slot2 is occupied.
  if [[ -z "${slot2_agent}" ]]; then
    pick=$(pick_laggard "${slot1_agent}")
    if [[ -n "${pick}" ]]; then
      slot2_agent="${pick}"
      slot2_start=$(count_for "${slot2_agent}")
      slot2_pid=$(launch_slot "${slot2_agent}")
      echo "[$(date '+%H:%M:%S')] balancer: slot2 pick ${slot2_agent} (was ${slot2_start}/${MAX_PER_AGENT}, pid=${slot2_pid})"
    fi
  fi

  # If both slots empty (everyone done), exit.
  if [[ -z "${slot1_agent}" && -z "${slot2_agent}" ]]; then
    echo "[$(date '+%H:%M:%S')] balancer: every agent at ${MAX_PER_AGENT}/${MAX_PER_AGENT} — done"
    exit 0
  fi

  # Cycle deadline: kill stalled slots so a hung agent doesn't lock the loop.
  CYCLE_DEADLINE=$(( $(date +%s) + 1800 ))
  while true; do
    sleep 15
    [[ -f "${STOP_FLAG}" ]] && cleanup

    progressed_any=0

    if [[ -n "${slot1_agent}" ]]; then
      cur=$(count_for "${slot1_agent}")
      if [[ ${cur} -gt ${slot1_start} ]]; then
        echo "[$(date '+%H:%M:%S')] balancer: slot1 ${slot1_agent} ${slot1_start} -> ${cur}; re-picking"
        kill_slot "${slot1_pid}" "${slot1_agent}"
        slot1_agent=""; slot1_start=0; slot1_pid=0
        progressed_any=1
      fi
    fi

    if [[ -n "${slot2_agent}" ]]; then
      cur=$(count_for "${slot2_agent}")
      if [[ ${cur} -gt ${slot2_start} ]]; then
        echo "[$(date '+%H:%M:%S')] balancer: slot2 ${slot2_agent} ${slot2_start} -> ${cur}; re-picking"
        kill_slot "${slot2_pid}" "${slot2_agent}"
        slot2_agent=""; slot2_start=0; slot2_pid=0
        progressed_any=1
      fi
    fi

    [[ ${progressed_any} -eq 1 ]] && break

    if [[ $(date +%s) -ge ${CYCLE_DEADLINE} ]]; then
      echo "[$(date '+%H:%M:%S')] balancer: 30min stall — killing both slots and re-picking"
      kill_slot "${slot1_pid}" "${slot1_agent}"
      kill_slot "${slot2_pid}" "${slot2_agent}"
      slot1_agent=""; slot1_start=0; slot1_pid=0
      slot2_agent=""; slot2_start=0; slot2_pid=0
      break
    fi
  done

  sleep 2
done
