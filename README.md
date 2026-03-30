# SWE-PRBench — Evaluation Harness

> Paper: [View Paper](https://arxiv.org/abs/2603.26130)

> Dataset: [View Dataset](https://huggingface.co/datasets/foundry-ai/swe-prbench)

> Blog: [View Blog](https://foundryhq.ai/blog/swe-prbench-benchmarking-ai-code-review-quality)

Public repository for **running evaluations** on the SWE-PRBench dataset.

| Path | Purpose |
|------|---------|
| `eval_harness/` | Agent + judge pipeline (`run_eval.py`), scoring — see `eval_harness/README.md` |
| `RUBRIC.md` | Frozen classification rubric (CONFIRMED / PLAUSIBLE / FABRICATED) |
| `pipeline_version.txt` | Protocol version — must match the dataset build (`v0.4.1`) |

Dataset (contexts, annotations, `prs.jsonl`) is hosted separately on HuggingFace — not in this repo.

---

## Leaderboard (Paper Baseline)

![SWE-PRBench Leaderboard](https://huggingface.co/datasets/foundry-ai/swe-prbench/resolve/main/leaderboard.png)

| Rank | Model | Overall (s̄) | DR_A | FPR |
|------|-------|-------------|------|-----|
| 1 | Claude Haiku 4.5 | 0.153 | 0.306 | 0.346 |
| 2 | Claude Sonnet 4.6 | 0.152 | 0.297 | 0.227 |
| 3 | DeepSeek V3 | 0.150 | 0.312 | 0.315 |
| 4 | Mistral Large 3 | 0.147 | 0.305 | 0.353 |
| 5 | GPT-4o | 0.113 | 0.220 | 0.193 |
| 6 | GPT-4o-mini | 0.108 | 0.210 | 0.353 |
| 7 | Mistral Small | 0.106 | 0.257 | 0.251 |
| 8 | Llama 3.3 70B | 0.079 | 0.223 | 0.417 |

Evaluated on `evals/eval_100.json`. Judge: GPT-5.2. Pipeline: v0.4.1.

---

## Quick Start

**Step 1 — Download the dataset:**
```bash
huggingface-cli download foundry-ai/swe-prbench \
  --local-dir ./swe-prbench-data
```

The dataset must be laid out as:
```
<DATASET_ROOT>/
├── prs.jsonl
├── annotations/{task_id}_human.json
└── contexts/config_{A,B,C}/{task_id}.json
```

**Step 2 — Install the harness:**
```bash
git clone https://github.com/<org>/swe-prbench-harness.git
cd swe-prbench-harness
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp eval_harness/model_endpoints.example.yaml eval_harness/model_endpoints.yaml
# Fill in API keys via env vars
```

**Step 3 — Set API keys:**
```bash
export ANTHROPIC_API_KEY=...
export OPENAI_API_KEY=...
export GEMINI_API_KEY=...
export GROQ_API_KEY=...
export MISTRAL_API_KEY=...
```

---

## Running Evaluation

**Single model** (judge from `defaults.judge_model` in config):
```bash
python3 eval_harness/run_eval.py \
  --contexts ./swe-prbench-data/dataset/contexts \
  --annotations ./swe-prbench-data/dataset/annotations \
  --prs ./swe-prbench-data/dataset/prs.jsonl \
  --split ./swe-prbench-data/dataset/evals/eval_100.json \
  --output results/runs \
  --model-config eval_harness/model_endpoints.yaml \
  --model YOUR_AGENT_MODEL_ID
```

**Sweep all models** defined in `model_endpoints.yaml`:
```bash
python3 eval_harness/run_eval.py \
  --contexts ./swe-prbench-data/dataset/contexts \
  --annotations ./swe-prbench-data/dataset/annotations \
  --prs ./swe-prbench-data/dataset/prs.jsonl \
  --split ./swe-prbench-data/dataset/evals/eval_100.json \
  --output results/runs \
  --model-config eval_harness/model_endpoints.yaml \
  --agent-models all \
  --concurrency 4
```

**Smoke test** (limit PR count):
```bash
python3 eval_harness/run_eval.py \
  --contexts ./swe-prbench-data/dataset/contexts \
  --annotations ./swe-prbench-data/dataset/annotations \
  --prs ./swe-prbench-data/dataset/prs.jsonl \
  --output results/runs \
  --model-config eval_harness/model_endpoints.yaml \
  --agent-models all \
  --max-prs 2
```

### Outputs

Each run produces a directory under `results/runs/<agent_model>__judge_<judge_model>/`:

| File | Contents |
|------|----------|
| `agent_outputs/*_agent.json` | Raw agent outputs per PR |
| `judge_outputs/*_judge.json` | Judge classifications per PR |
| `eval_results/*_eval.json` | Scored results per PR |
| `eval_report.json` | Aggregate report for leaderboard |
| `validation_failures.json` | Parse failures and fallbacks |

---

## Reproducibility Note

Scores reported in the paper reflect pipeline version `v0.4.1` with GPT-5.2 as judge at temperature=0. Frontier model APIs do not guarantee full determinism at temperature=0, so minor score variation across independent runs is expected. The two-tier ranking structure and A>B>C ordering are stable across runs and confirmed by cross-judge validation in the paper.

---

## Docs

- **Command reference:** `eval_harness/COMMANDS.md`
- **CLI layout:** `eval_harness/README.md`
- **Classification rubric:** `RUBRIC.md`

---

## Citation

If you use SWE-PRBench in your research, please cite the dataset:
```bibtex
@misc{kumar2026sweprbench,
  title = {SWE-PRBench: Benchmarking AI Code Review Quality
         Against Real Pull Request Feedback},
  author={Kumar, Deepak},
  archivePrefix = {arXiv},
  primaryClass = {cs.SE},
  url = {https://arxiv.org/abs/2603.26130}
}
```

## License

Evaluation harness: MIT License  
Dataset: CC BY 4.0 (see HuggingFace)
