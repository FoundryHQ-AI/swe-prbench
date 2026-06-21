#!/usr/bin/env bash
# Re-launch the four background agents after a usage-cap stop. Logs go
# into the same per-agent log files (append mode), and concurrency stays
# at 2 by default.
set -u
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"
# shellcheck source=/dev/null
source .venv/bin/activate
for agent in opus_4_8_low opus_4_8_medium opus_4_8_max opus_4_7_max; do
  CONCURRENCY=2 bash claude_code_eval/scripts/run_single_agent.sh "${agent}" \
    >> "claude_code_eval/results/logs/${agent}.log" 2>&1 &
done
disown
echo "resume_when_reset: relaunched 4 agents in the background"
