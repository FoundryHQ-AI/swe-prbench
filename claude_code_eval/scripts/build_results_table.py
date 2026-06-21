#!/usr/bin/env python3
"""Walk ``results/runs/<agent>__judge_<judge>/eval_report.json`` and emit a
combined CSV + markdown leaderboard.

The columns mirror what the upstream paper's leaderboard exposes (overall
score, detection rate on config_A, false-positive rate) plus the per-config
breakdown so the reasoning-effort axis is visible.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunRow:
    agent_model: str
    judge_model: str
    n_prs: int
    n_records: int
    overall_a: float
    overall_b: float
    overall_c: float
    overall_mean: float
    detection_a: float
    fpr_a: float
    hallucination_a: float
    precision_a: float
    recall_a: float
    f1_a: float
    actionability_a: float
    coverage_a: float
    agent_parse_failures: int
    judge_parse_fallback_count: int


def _safe(d: dict, *path, default=0.0):
    cur = d
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur if cur is not None else default


def _collect_run(run_dir: Path) -> RunRow | None:
    report_path = run_dir / "eval_report.json"
    if not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"warn: failed to parse {report_path}: {e}", file=sys.stderr)
        return None

    # The run directory is `<agent>__judge_<judge>`; recover the IDs from it.
    name = run_dir.name
    if "__judge_" in name:
        agent, judge = name.split("__judge_", 1)
    else:
        agent, judge = name, "unknown"

    by_config = report.get("by_config") or {}
    cfg_a = by_config.get("config_A") or {}
    cfg_b = by_config.get("config_B") or {}
    cfg_c = by_config.get("config_C") or {}

    overall_vals = [
        float(_safe(cfg_a, "overall_score")),
        float(_safe(cfg_b, "overall_score")),
        float(_safe(cfg_c, "overall_score")),
    ]
    overall_vals = [v for v in overall_vals if v]
    overall_mean = round(sum(overall_vals) / len(overall_vals), 3) if overall_vals else 0.0

    return RunRow(
        agent_model=agent,
        judge_model=judge,
        n_prs=int(report.get("total_prs", 0)),
        n_records=int(report.get("total_eval_records", 0)),
        overall_a=float(_safe(cfg_a, "overall_score")),
        overall_b=float(_safe(cfg_b, "overall_score")),
        overall_c=float(_safe(cfg_c, "overall_score")),
        overall_mean=overall_mean,
        detection_a=float(_safe(cfg_a, "detection_rate")),
        fpr_a=float(_safe(cfg_a, "false_positive_rate")),
        hallucination_a=float(_safe(cfg_a, "hallucination_rate")),
        precision_a=float(_safe(cfg_a, "precision")),
        recall_a=float(_safe(cfg_a, "recall")),
        f1_a=float(_safe(cfg_a, "f1_score")),
        actionability_a=float(_safe(cfg_a, "actionability_score")),
        coverage_a=float(_safe(cfg_a, "coverage")),
        agent_parse_failures=int(report.get("agent_parse_failures", 0)),
        judge_parse_fallback_count=int(report.get("judge_parse_fallback_count", 0)),
    )


def _write_csv(rows: list[RunRow], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [f for f in RunRow.__dataclass_fields__]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fieldnames})


def _write_md(rows: list[RunRow], out_md: Path) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    # Sort by overall_mean desc, then by overall_a desc.
    rows = sorted(rows, key=lambda r: (-r.overall_mean, -r.overall_a))

    lines: list[str] = []
    lines.append("# Claude Code CLI on SWE-PRBench — Results")
    lines.append("")
    lines.append(
        f"Evaluated on the `eval_100` split, judged by Claude Sonnet 4.6 "
        f"via the local `claude` CLI. {len(rows)} agent configurations."
    )
    lines.append("")
    lines.append("## Headline leaderboard")
    lines.append("")
    lines.append(
        "| Rank | Agent | Overall (mean A/B/C) | DR_A | FPR_A | Halluc_A | F1_A | Coverage_A |"
    )
    lines.append("|------|-------|---------------------|------|-------|----------|------|------------|")
    for i, r in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{r.agent_model}` | {r.overall_mean:.3f} | "
            f"{r.detection_a:.3f} | {r.fpr_a:.3f} | "
            f"{r.hallucination_a:.3f} | {r.f1_a:.3f} | {r.coverage_a:.3f} |"
        )
    lines.append("")
    lines.append("## Per-context overall score")
    lines.append("")
    lines.append("| Agent | config_A (diff only) | config_B (with file content) | config_C (full context) |")
    lines.append("|-------|----------------------|------------------------------|------------------------|")
    for r in rows:
        lines.append(
            f"| `{r.agent_model}` | {r.overall_a:.3f} | {r.overall_b:.3f} | {r.overall_c:.3f} |"
        )
    lines.append("")
    lines.append("## Run-level diagnostics")
    lines.append("")
    lines.append("| Agent | Judge | PRs | Records | Parse failures | Judge fallbacks |")
    lines.append("|-------|-------|-----|---------|----------------|-----------------|")
    for r in rows:
        lines.append(
            f"| `{r.agent_model}` | `{r.judge_model}` | {r.n_prs} | {r.n_records} | "
            f"{r.agent_parse_failures} | {r.judge_parse_fallback_count} |"
        )
    lines.append("")
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _inject_readme_table(rows: list[RunRow], readme_path: Path) -> bool:
    if not readme_path.exists():
        return False
    text = readme_path.read_text(encoding="utf-8")
    begin = "<!-- BEGIN: CLAUDE_CODE_LEADERBOARD -->"
    end = "<!-- END: CLAUDE_CODE_LEADERBOARD -->"
    if begin not in text or end not in text:
        return False
    rows = sorted(rows, key=lambda r: (-r.overall_mean, -r.overall_a))
    block: list[str] = []
    block.append("")
    block.append("| Rank | Agent | Overall (mean A/B/C) | DR_A | FPR_A | Halluc_A | F1_A |")
    block.append("|------|-------|---------------------|------|-------|----------|------|")
    if not rows:
        block.append("| _no runs yet_ | | | | | | |")
    for i, r in enumerate(rows, 1):
        block.append(
            f"| {i} | `{r.agent_model}` | {r.overall_mean:.3f} | "
            f"{r.detection_a:.3f} | {r.fpr_a:.3f} | "
            f"{r.hallucination_a:.3f} | {r.f1_a:.3f} |"
        )
    block.append("")
    block.append(
        "_Judge: Claude Sonnet 4.6 (`cli_claude`). Numbers comparable across "
        "fork agents only — not drop-in comparable to the paper's GPT-5.2-judge "
        "leaderboard._"
    )
    block.append("")
    new_section = f"{begin}\n" + "\n".join(block) + f"{end}"
    pre, _, rest = text.partition(begin)
    _, _, post = rest.partition(end)
    readme_path.write_text(pre + new_section + post, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True, help="Directory holding <agent>__judge_<judge>/ subdirs.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--out-md", required=True, help="Output Markdown path.")
    parser.add_argument(
        "--readme",
        default=None,
        help="Optional path to top-level README to inject the table into "
        "(between BEGIN/END: CLAUDE_CODE_LEADERBOARD markers).",
    )
    args = parser.parse_args()

    root = Path(args.runs_root)
    if not root.exists():
        print(f"error: runs root not found: {root}", file=sys.stderr)
        return 2

    rows: list[RunRow] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        row = _collect_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        print(f"warn: no eval_report.json files found under {root}", file=sys.stderr)

    _write_csv(rows, Path(args.out_csv))
    _write_md(rows, Path(args.out_md))
    print(f"wrote {len(rows)} rows -> {args.out_csv}")
    print(f"wrote markdown leaderboard -> {args.out_md}")
    if args.readme:
        if _inject_readme_table(rows, Path(args.readme)):
            print(f"injected leaderboard into {args.readme}")
        else:
            print(f"warn: could not inject into {args.readme} (markers missing)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
