#!/usr/bin/env bash
# Usage orchestrator: probe Claude subscription usage, stop runs at the
# threshold, and resume after the window resets.
#
# Loop:
#   while true:
#     check_usage
#     if SESSION_PCT >= THRESHOLD_STOP and runs are alive:
#       run stop_all.sh
#       inner loop: every WAIT_INTERVAL, check_usage; if <= THRESHOLD_RESUME, resume
#     sleep CHECK_INTERVAL
#
# Defaults: stop at 70%, wait 60min between checks once paused, resume
# when session usage drops back to <=5% (i.e. window reset).
set -u

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

CHECK_INTERVAL="${CHECK_INTERVAL:-1800}"      # 30 min between checks while running
WAIT_INTERVAL="${WAIT_INTERVAL:-3600}"        # 1h between checks while paused
THRESHOLD_STOP="${THRESHOLD_STOP:-70}"        # stop runs at this session_pct
THRESHOLD_RESUME="${THRESHOLD_RESUME:-5}"     # resume when session_pct drops below this

state="running"   # "running" or "paused"

while true; do
  out=$(bash claude_code_eval/scripts/check_usage.sh 2>/dev/null || true)
  session_pct=$(echo "${out}" | grep -E "^SESSION_PCT=" | head -1 | cut -d= -f2)
  week_pct=$(echo "${out}" | grep -E "^WEEK_PCT=" | head -1 | cut -d= -f2)
  reset=$(echo "${out}" | grep -E "^SESSION_RESET=" | head -1 | cut -d= -f2-)
  echo "USAGE_PROBE state=${state} session_pct=${session_pct} week_pct=${week_pct} reset=${reset} at $(date '+%H:%M:%S')"

  procs=$(ps aux | grep "eval_harness/run_eval.py" | grep -v grep | wc -l | tr -d ' ')

  case "${state}" in
    running)
      # Numeric check; treat 'unknown' / non-numeric as "no signal, keep going"
      if [[ "${session_pct}" =~ ^[0-9]+$ ]] && [[ ${session_pct} -ge ${THRESHOLD_STOP} ]]; then
        echo "USAGE_STOP session_pct=${session_pct} >= THRESHOLD_STOP=${THRESHOLD_STOP} — stopping runs"
        bash claude_code_eval/scripts/stop_all.sh || true
        state="paused"
        sleep_seconds=${WAIT_INTERVAL}
      elif [[ ${procs} -eq 0 ]]; then
        echo "USAGE_NOTE no run_eval.py processes detected while state=running — exiting orchestrator"
        break
      else
        sleep_seconds=${CHECK_INTERVAL}
      fi
      ;;
    paused)
      if [[ "${session_pct}" =~ ^[0-9]+$ ]] && [[ ${session_pct} -le ${THRESHOLD_RESUME} ]]; then
        echo "USAGE_RESUME session_pct=${session_pct} <= THRESHOLD_RESUME=${THRESHOLD_RESUME} — relaunching agents"
        bash claude_code_eval/scripts/resume_when_reset.sh || true
        state="running"
        sleep_seconds=${CHECK_INTERVAL}
      else
        sleep_seconds=${WAIT_INTERVAL}
      fi
      ;;
  esac

  sleep "${sleep_seconds}"
done
