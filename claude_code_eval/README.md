# Claude Code CLI experiment

This directory is the entry point for the experiment added in this fork:
**evaluate Claude Opus 4.7 and 4.8 on SWE-PRBench by driving the local
`claude` CLI in headless mode**, billed through the user's Anthropic
subscription instead of an API key.

It is a self-contained add-on. The upstream `eval_harness/` runs exactly as
before — this folder only adds an alternate model config, a driver script,
and an aggregator that writes the leaderboard table back into this repo.

## What it adds to the upstream harness

| Change | Where | Why |
|---|---|---|
| `cli_claude` provider type | `eval_harness/model_clients.py` | Routes `model_router.generate()` calls through `claude -p --model X --effort Y --tools "" --no-session-persistence`, capturing stdout as the completion. No API key needed because the CLI uses the local OAuth session. |
| `model_endpoints.claude_code.yaml` | `eval_harness/` | Fork-specific model config with four agent entries (`opus_4_8_{low,medium,max}`, `opus_4_7_max`) and a Claude Sonnet 4.6 judge — all over `cli_claude`. |
| `scripts/run_experiment.sh` | here | Driver: extracts the `eval_100` task_id list, invokes the harness for each agent against the Sonnet judge, then builds the result table. |
| `scripts/build_results_table.py` | here | Walks `results/runs/<agent>__judge_*/eval_report.json` and emits `leaderboard.csv` + `leaderboard.md`. |
| `scripts/extract_eval_100_task_ids.py` | here | Pulls the 100 `task_id`s from `dataset/evals/eval_100.json`. |

## Methodology notes

- **Judge.** The paper's headline numbers use GPT-5.2 as judge with Claude
  Sonnet 4.6 as a cross-validator (κ=0.75). This experiment uses Sonnet 4.6
  as the judge because it stays inside the subscription and inside the
  paper's validated judge family. Numbers here are therefore comparable
  across the four agent configurations in this fork, but **not** drop-in
  comparable to the upstream paper leaderboard.
- **One-shot.** The harness's pipeline is one prompt → one completion. The
  `claude` CLI is invoked with `--tools ""` so it cannot read files, run
  bash, or otherwise behave as an agent. The diff and PR context are
  injected via stdin; the rubric/system prompt is passed via
  `--system-prompt`.
- **Effort axis.** Each opus_4_8_* entry differs only in the `--effort`
  flag passed to `claude` (low / medium / max). Everything else — split,
  judge, contexts, system prompt — is held constant.

## Running

```bash
# 1. Download the dataset (one-time, ~7 minutes)
hf download foundry-ai/swe-prbench --repo-type dataset --local-dir ./swe-prbench-data

# 2. Set up Python (3.10+ required for the harness)
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Sanity check (3 PRs × 3 contexts on opus_4_8_low)
bash claude_code_eval/scripts/run_experiment.sh smoke

# 4. Full run (4 agents × 100 PRs × 3 contexts)
bash claude_code_eval/scripts/run_experiment.sh

# 5. Just one agent
bash claude_code_eval/scripts/run_experiment.sh opus_4_8_max
```

Per-run output lands under `claude_code_eval/results/runs/<agent>__judge_claude_sonnet_46_judge/`
with the same `eval_report.json` + `eval_results/` layout the upstream harness
emits. The aggregator writes the combined leaderboard to
`claude_code_eval/results/leaderboard.{csv,md}`.

## Concurrency

`run_experiment.sh` defaults to `CONCURRENCY=2`. Each unit of concurrency is
a live `claude -p` subprocess plus a judge call; pushing this much past 4
on a laptop can cause OOMs or the OAuth rate limiter to kick in. Override
with `CONCURRENCY=4 bash claude_code_eval/scripts/run_experiment.sh`.

## Pause / resume

The full 4×100×3 run takes many hours and may need to be paused (subscription
token budget, machine reboot, etc.). The harness in this fork supports
**clean resume**: stop the running processes whenever, then re-launch with
the same script invocation, and it will skip any `(task_id, config)` pair
whose `eval_result_*.json` is already on disk and continue with the rest.

Stop a run cleanly by sending SIGINT/SIGTERM to the bash + python processes
(or close the terminal). Restart with:

```bash
# Resume a single agent
bash claude_code_eval/scripts/run_single_agent.sh opus_4_8_max

# Or resume all four agents (sequentially); already-done records are
# auto-skipped, so the cost of running this even after a partial run is
# only the remaining records.
bash claude_code_eval/scripts/run_experiment.sh
```

Resume is implemented as a one-line check at the top of the per-task
coroutine in `eval_harness/run_eval.py` (look for `eval_task_skip_resume`).
