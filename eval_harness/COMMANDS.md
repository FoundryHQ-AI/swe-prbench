# SWE-PRBench Eval Harness — Commands

This repository is the **evaluation harness only**. Download the dataset (contexts, annotations, `prs.jsonl`) from the project’s Hugging Face dataset page and lay it out as:

```
<DATASET_ROOT>/
├── prs.jsonl
├── annotations/{task_id}_human.json
└── contexts/config_{A,B,C}/{task_id}.json
```

See **`RUBRIC.md`** for judge taxonomy and **`pipeline_version.txt`** for the frozen protocol version.

## Prerequisites

```bash
cd /path/to/swe-prbench-harness
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp eval_harness/model_endpoints.example.yaml eval_harness/model_endpoints.yaml
# Fill in API keys via env vars referenced in model_endpoints.yaml (never commit secrets).
```

```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
```

## Run evaluation

Single agent model (judge from `defaults.judge_model` in config):

```bash
python3 eval_harness/run_eval.py \
  --contexts /path/to/dataset/contexts \
  --annotations /path/to/dataset/annotations \
  --prs /path/to/dataset/prs.jsonl \
  --output results/runs \
  --model-config eval_harness/model_endpoints.yaml \
  --model YOUR_AGENT_MODEL_ID
```

Sweep all models defined in `model_endpoints.yaml`:

```bash
python3 eval_harness/run_eval.py \
  --contexts /path/to/dataset/contexts \
  --annotations /path/to/dataset/annotations \
  --prs /path/to/dataset/prs.jsonl \
  --output results/runs \
  --model-config eval_harness/model_endpoints.yaml \
  --agent-models all
  --concurrency 4
```

Limit PR count (smoke test):

```bash
python3 eval_harness/run_eval.py ... --agent-models all --max-prs 2
```

## Outputs

Under each run directory (e.g. `agent_model__judge_model/`):

- `agent_outputs/*_agent.json`
- `judge_outputs/*_judge.json`
- `eval_results/*_eval.json`
- `eval_report.json`
- `validation_failures.json` (if any)
