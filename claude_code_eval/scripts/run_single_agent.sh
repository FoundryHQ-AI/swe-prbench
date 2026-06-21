#!/usr/bin/env bash
# Run one agent against the eval_100 split. Intended to be launched in
# parallel for each of the four agent configurations.
#
# Usage: ./run_single_agent.sh <agent_model_id>
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <agent_model_id>" >&2
  exit 2
fi

AGENT="$1"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT:-${REPO_ROOT}/swe-prbench-data/dataset}"
MODEL_CONFIG="${MODEL_CONFIG:-${REPO_ROOT}/eval_harness/model_endpoints.claude_code.yaml}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/claude_code_eval/results/runs}"
CONCURRENCY="${CONCURRENCY:-2}"
AGENT_MAX_TOKENS="${AGENT_MAX_TOKENS:-4000}"
TASK_IDS_FILE="${TASK_IDS_FILE:-${REPO_ROOT}/claude_code_eval/results/eval_100_task_ids.txt}"

if [[ ! -f "${TASK_IDS_FILE}" ]]; then
  python3 "${REPO_ROOT}/claude_code_eval/scripts/extract_eval_100_task_ids.py" \
    --split "${DATASET_ROOT}/evals/eval_100.json" \
    --out "${TASK_IDS_FILE}"
fi

TASK_IDS=()
while IFS= read -r line; do
  [[ -z "${line//[[:space:]]/}" ]] && continue
  TASK_IDS+=("$line")
done < "${TASK_IDS_FILE}"

echo "[${AGENT}] starting at $(date '+%F %T'), ${#TASK_IDS[@]} task_ids, concurrency=${CONCURRENCY}"

python3 eval_harness/run_eval.py \
  --contexts "${DATASET_ROOT}/contexts" \
  --annotations "${DATASET_ROOT}/annotations" \
  --prs "${DATASET_ROOT}/prs.jsonl" \
  --output "${OUTPUT_ROOT}" \
  --model-config "${MODEL_CONFIG}" \
  --model "${AGENT}" \
  --agent-max-tokens "${AGENT_MAX_TOKENS}" \
  --concurrency "${CONCURRENCY}" \
  --task-ids "${TASK_IDS[@]}"

echo "[${AGENT}] done at $(date '+%F %T')"
