# Eval harness layout

## Core (required to run `run_eval.py`)

| Module | Role |
|--------|------|
| `run_eval.py` | CLI: load contexts/annotations, call agent + judge, write per-PR results and `eval_report.json` |
| `runner.py` | Agent prompt + API calls |
| `judge.py` | Judge prompt + rubric classification |
| `loader.py` | Resolve `prs` records and paths to context JSON |
| `model_clients.py` | Anthropic / OpenAI-compatible / Google Gemini HTTP |
| `schema.py` | Dataclasses for I/O |
| `matching.py` | Judge alignment + semantic similarity (optional `sentence-transformers`) |
| `scorer.py` | Coverage / FPR-style aggregates |
| `assembler.py` | Build `EvalResult` from agent + judge |
| `aggregate.py` | Roll up records into `eval_report.json` |
| `validate_output.py` | Structural checks on results |
| `io_utils.py` | JSON helpers |
| `logging_utils.py` | Structlog / fallback logging |

## Optional maintainer utilities

These are **not** needed for a standard eval; use when debugging or revisiting runs.

| Script | Purpose |
|--------|---------|
| `rebuild_report.py` | Recompute `eval_report.json` from `eval_results/*_eval.json` |
| `rerun_failed.py` | Build shell commands to rerun failed task ids |
| `cross_judge_validation.py` | Secondary judge labels on a stratified sample (research / κ) |

## CLI groups

`run_eval.py --help` groups flags into: **Dataset paths**, **Models**, **Evaluation scope**, **Performance**, and **Optional batch APIs**.
