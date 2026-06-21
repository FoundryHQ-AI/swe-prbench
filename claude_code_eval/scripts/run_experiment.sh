#!/usr/bin/env bash
# Run the Claude Code CLI experiment: 4 agent configs × 100 PRs × 3 contexts.
#
# Layout assumed:
#   ./swe-prbench-data/   (downloaded from HuggingFace, see top-level README)
#   ./eval_harness/model_endpoints.claude_code.yaml
#   ./claude_code_eval/results/runs/   (created by this script)
#
# Usage:
#   ./claude_code_eval/scripts/run_experiment.sh            # all 4 agents
#   ./claude_code_eval/scripts/run_experiment.sh smoke      # smoke test on 3 PRs, opus_4_8_low only
#   ./claude_code_eval/scripts/run_experiment.sh opus_4_8_low opus_4_7_max  # specific subset
#
# Environment overrides:
#   CONCURRENCY=N           agent calls in flight (default 2)
#   AGENT_MAX_TOKENS=N      max output tokens per agent call (default 4000)
#   CLAUDE_CLI_TIMEOUT_SEC=N per-call subprocess timeout (default 900)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/swe-prbench-data/dataset}"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/eval_harness/model_endpoints.claude_code.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/claude_code_eval/results/runs}"
CONCURRENCY="${CONCURRENCY:-2}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-4000}"
TASK_IDS_FILE="${TASK_IDS_FILE:-${REPO_ROOT}/claude_code_eval/results/eval_100_task_ids.txt}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "error: dataset not found at ${DATASET_ROOT}" >&2
  echo "       run: hf download foundry-ai/swe-prbench --repo-type dataset --local-dir ./swe-prbench-data" >&2
  exit 2
fi

if [[ ! -f "${TASK_IDS_FILE}" ]]; then
  echo "extracting task_ids for the eval_100 split..."
  python3 "${REPO_ROOT}/claude_code_eval/scripts/extract_eval_100_task_ids.py" \
    --split "${DATASET_ROOT}/evals/eval_100.json" \
    --out "${TASK_IDS_FILE}"
fi

# Pick which agent models to run.
MODE="${1:-all}"
if [[ "${MODE}" == "smoke" ]]; then
  AGENTS=(opus_4_8_low)
  TASK_LIMIT_FLAG=(--max-prs 3)
  echo "==> smoke test: opus_4_8_low on 3 PRs"
elif [[ "${MODE}" == "all" ]]; then
  AGENTS=(opus_4_8_low opus_4_8_medium opus_4_8_max opus_4_7_max)
  TASK_LIMIT_FLAG=()
else
  AGENTS=("$@")
  TASK_LIMIT_FLAG=()
fi

mkdir -p "${OUTPUT_ROOT}"

# Read task IDs into an argv list (one per line, skip blanks). macOS ships
# with bash 3.2 which lacks `mapfile`, so use a portable read loop instead.
TASK_IDS=()
while IFS= read -r line; do
  [[ -z "${line//[[:space:]]/}" ]] && continue
  TASK_IDS+=("$line")
done < "${TASK_IDS_FILE}"
echo "task_ids: ${#TASK_IDS[@]} entries"

for agent in "${AGENTS[@]}"; do
  echo
  echo "============================================================"
  echo "  AGENT: ${agent}    (judge: claude_sonnet_46_judge)"
  echo "============================================================"
  python3 eval_harness/run_eval.py \
    --contexts "${DATASET_ROOT}/contexts" \
    --annotations "${DATASET_ROOT}/annotations" \
    --prs "${DATASET_ROOT}/prs.jsonl" \
    --output "${OUTPUT_ROOT}" \
    --model-config "${MODEL_CONFIG}" \
    --model "${agent}" \
    --agent-max-tokens "${AGENT_MAX_TOKENS}" \
    --concurrency "${CONCURRENCY}" \
    --task-ids "${TASK_IDS[@]}" \
    "${TASK_LIMIT_FLAG[@]}"
done

echo
echo "==> done. building results table..."
python3 "${REPO_ROOT}/claude_code_eval/scripts/build_results_table.py" \
  --runs-root "${OUTPUT_ROOT}" \
  --out-csv "${REPO_ROOT}/claude_code_eval/results/leaderboard.csv" \
  --out-md "${REPO_ROOT}/claude_code_eval/results/leaderboard.md" \
  --readme "${REPO_ROOT}/README.md"
