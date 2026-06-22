#!/usr/bin/env bash
# Re-launch the four background agents after a usage-cap stop. Logs go
# into the same per-agent log files (append mode), and concurrency stays
# at 2 by default.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source .venv/bin/activate

# Default: launch the sequential balancer, which runs one record at a
# time and re-picks the laggard after every record. This keeps per-agent
# counts within 1 of each other (fair comparison) and concentrates the
# per-window token budget on one stream at a time.
#
# AGENTS env var override (space-separated list) reverts to legacy
# parallel fan-out — useful for testing.
if [[ -n "${AGENTS:-}" ]]; then
  echo "resume_when_reset: legacy parallel fan-out — AGENTS=${AGENTS}"
  for agent in ${AGENTS}; do
    CONCURRENCY=2 bash claude_code_eval/scripts/run_single_agent.sh "${agent}" \
      >> "claude_code_eval/results/logs/${agent}.log" 2>&1 &
  done
  disown
else
  echo "resume_when_reset: launching sequential balancer (1 record, then re-pick laggard)"
  bash claude_code_eval/scripts/sequential_balancer.sh \
    >> "claude_code_eval/results/logs/balancer.log" 2>&1 &
  disown
fi
