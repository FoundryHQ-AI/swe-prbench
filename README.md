# SWE-PRBench — Evaluation Harness (Claude Code CLI fork)

> Fork of [FoundryHQ-AI/swe-prbench](https://github.com/FoundryHQ-AI/swe-prbench)
> adding a `cli_claude` provider that drives the local `claude` CLI
> (Claude Code) in headless mode under the user's logged-in OAuth
> subscription — no API key billing.

> Paper: [View Paper](https://arxiv.org/abs/2603.26130)

> Dataset: [View Dataset](https://huggingface.co/datasets/foundry-ai/swe-prbench)

> Blog: [View Blog](https://foundryhq.ai/blog/swe-prbench-benchmarking-ai-code-review-quality)

Public repository for **running evaluations** on the SWE-PRBench dataset.

| Path | Purpose |
|------|---------|
| `eval_harness/` | Agent + judge pipeline (`run_eval.py`), scoring — see `eval_harness/README.md` |
| `eval_harness/model_endpoints.claude_code.yaml` | Fork-only config: 4 Claude Opus 4.7/4.8 agent entries + Sonnet 4.6 judge, all over `cli_claude` |
| `claude_code_eval/` | Driver + aggregator + results for the Claude Code CLI experiment — see `claude_code_eval/README.md` |
| `RUBRIC.md` | Frozen classification rubric (CONFIRMED / PLAUSIBLE / FABRICATED) |
| `pipeline_version.txt` | Protocol version — must match the dataset build (`v0.4.1`) |

Dataset (contexts, annotations, `prs.jsonl`) is hosted separately on HuggingFace — not in this repo.

---

## Claude Code CLI experiment (this fork)

Four agent configurations across the Claude Opus 4.7/4.8 reasoning-effort
axis, evaluated on the `eval_100` split with Claude Sonnet 4.6 as judge.
All calls route through `claude -p` so the cost is covered by the user's
Anthropic subscription rather than API-key billing.

<!-- BEGIN: CLAUDE_CODE_LEADERBOARD -->

**Headline (mean of config_A/B/C, 100 PRs × 3 contexts = 300 records per agent):**

| Rank | Agent | Overall | DR_A  | FPR_A | Halluc_A | F1_A  | Coverage_A |
|------|-------|---------|-------|-------|----------|-------|------------|
| 1 | `opus_4_8_low` | 0.437 | 0.633 | 0.080 | 0.080 | 0.500 | 0.860 |
| 2 | `opus_4_8_max` | 0.436 | 0.642 | 0.072 | 0.072 | 0.496 | 0.880 |
| 3 | `opus_4_8_medium` | 0.434 | 0.665 | 0.082 | 0.082 | 0.517 | 0.870 |
| 4 | `opus_4_7_max` | 0.411 | 0.696 | 0.068 | 0.068 | 0.447 | 0.890 |

**Per-context overall score:**

| Agent | config_A (diff only) | config_B (+ file content) | config_C (full context) |
|-------|----------------------|---------------------------|-------------------------|
| `opus_4_8_low`    | 0.466 | 0.414 | 0.430 |
| `opus_4_8_max`    | 0.467 | 0.416 | 0.425 |
| `opus_4_8_medium` | 0.479 | 0.410 | 0.412 |
| `opus_4_7_max`    | 0.452 | 0.394 | 0.387 |

**Key findings:**

1. **Reasoning-effort axis is essentially flat on Opus 4.8.** low / medium / max
   land within 0.003 of each other on overall. Higher effort does not help
   on this benchmark.
2. **Opus 4.7 max wins raw detection** (DR_A=0.696) **and has the lowest
   hallucination rate** (Halluc_A=0.068), but finishes last on overall
   because it emits ~37% more comments per PR (~4.9 vs ~3.6). The extra
   comments are overwhelmingly PLAUSIBLE (real-looking observations
   humans didn't raise), which the harness's precision component penalises.
3. **Diff-only (config_A) beats full context (B and C) for every agent**, by
   3-7 pp on overall_score. This matches the paper's headline finding
   across all 8 of its non-Anthropic models.
4. **Hallucination drops with more context for every agent** (e.g. medium:
   8.2% A → 4.7% B → 3.5% C). More context = fewer factual errors, but
   detection drops faster than hallucination falls.

_Judge: Claude Sonnet 4.6 via `cli_claude`. Numbers comparable across fork
agents only — not drop-in comparable to the paper's GPT-5.2-judge
leaderboard. CSV at [`claude_code_eval/results/leaderboard.csv`](claude_code_eval/results/leaderboard.csv);
per-PR eval reports under [`claude_code_eval/results/runs/`](claude_code_eval/results/runs/)._
<!-- END: CLAUDE_CODE_LEADERBOARD -->

Methodology, judge choice, and reproduction instructions:
[`claude_code_eval/README.md`](claude_code_eval/README.md).

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
