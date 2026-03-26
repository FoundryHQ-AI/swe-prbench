#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any

# Ensure repo root is importable when script is run directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval_harness.aggregate import generate_eval_report
from eval_harness.io_utils import write_json
from eval_harness.schema import EvalResult


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_eval_result(obj: dict[str, Any]) -> EvalResult:
    allowed = {f.name for f in fields(EvalResult)}
    filtered = {k: v for k, v in obj.items() if k in allowed}
    # Ensure optional list fields exist if older files omitted them.
    filtered.setdefault("matched_pairs", [])
    filtered.setdefault("missed_comments", [])
    filtered.setdefault("fabricated_comments", [])
    return EvalResult(**filtered)


def _discover_run_dirs(inputs: list[str]) -> list[Path]:
    run_dirs: list[Path] = []
    for raw in inputs:
        p = Path(raw).resolve()
        if not p.exists():
            continue
        # Run dir convention: contains eval_results/*.json
        if p.is_dir() and (p / "eval_results").is_dir():
            run_dirs.append(p)
            continue
        if p.is_dir():
            for sub in sorted(p.rglob("*")):
                if sub.is_dir() and (sub / "eval_results").is_dir():
                    run_dirs.append(sub.resolve())
    # deterministic unique
    seen: set[str] = set()
    out: list[Path] = []
    for d in run_dirs:
        s = str(d)
        if s in seen:
            continue
        out.append(d)
        seen.add(s)
    return out


def rebuild_run_report(run_dir: Path) -> tuple[int, str]:
    eval_results_dir = run_dir / "eval_results"
    files = sorted(eval_results_dir.glob("*_eval.json"))
    results: list[EvalResult] = []
    for f in files:
        obj = _load_json(f)
        if isinstance(obj, dict):
            try:
                results.append(_as_eval_result(obj))
            except Exception:
                continue
    report_path = run_dir / "eval_report.json"
    report = generate_eval_report(results, str(report_path))

    # Keep/refresh lightweight multi-model summary entry-like file if present.
    summary = {
        "run_dir": str(run_dir),
        "eval_records": len(results),
        "prs": int(report.get("total_prs") or 0),
        "report_run_date": report.get("run_date"),
    }
    write_json(summary, run_dir / "report_rebuild_summary.json")
    return len(results), str(report_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild eval_report.json from existing eval_results files.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Run directories (or parent directories) containing eval_results/*.json",
    )
    args = parser.parse_args()

    run_dirs = _discover_run_dirs(args.inputs)
    if not run_dirs:
        raise FileNotFoundError("No run directories with eval_results found in --inputs.")

    rebuilt: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        n, report_path = rebuild_run_report(run_dir)
        rebuilt.append({"run_dir": str(run_dir), "eval_records": n, "report_path": report_path})

    print(json.dumps({"runs_rebuilt": len(rebuilt), "details": rebuilt}, indent=2))


if __name__ == "__main__":
    main()

