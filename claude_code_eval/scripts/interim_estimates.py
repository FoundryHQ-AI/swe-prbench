#!/usr/bin/env python3
"""Interim score estimates for the in-flight Claude Code CLI experiment.

The full experiment is 4 agents x 100 PRs x 3 contexts = 1200 records. This
script walks the partially-populated ``eval_results/`` snapshots, computes
per-config and aggregate means per agent, prints a human-readable interim
table to stdout, and writes a CSV at
``claude_code_eval/results/interim_estimates.csv``.

It is safe to re-run while the eval workers are still writing files: each
JSON file is read once and parse errors are skipped with a warning.

The reported numbers are NOT final. Sample sizes vary across agents; treat
these as directional signal only.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Match eval_harness/aggregate.py::_normalize_config_name
_CONFIG_NAME_MAP = {
    "config_A_diff_only": "config_A",
    "config_B_with_file_content": "config_B",
    "config_C_full_context": "config_C",
    "config_A": "config_A",
    "config_B": "config_B",
    "config_C": "config_C",
}
_CONFIGS = ["config_A", "config_B", "config_C"]
_METRICS = [
    "overall_score",
    "detection_rate",
    "false_positive_rate",
    "hallucination_rate",
    "f1_score",
    "coverage",
]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = REPO_ROOT / "claude_code_eval" / "results" / "runs"
CSV_OUT = REPO_ROOT / "claude_code_eval" / "results" / "interim_estimates.csv"

JUDGE_SUFFIX = "__judge_claude_sonnet_46_judge"


def _normalize_config(name: str) -> str:
    return _CONFIG_NAME_MAP.get(str(name), str(name))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _load_eval_record(path: Path) -> dict | None:
    """Read one eval result JSON. Returns None on read/parse failure."""
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        print(f"warn: skipping {path.name}: {e}", file=sys.stderr)
        return None
    if not isinstance(data, dict):
        print(f"warn: skipping {path.name}: not a JSON object", file=sys.stderr)
        return None
    return data


def _collect_agent_records(agent_dir: Path) -> list[dict]:
    eval_dir = agent_dir / "eval_results"
    if not eval_dir.is_dir():
        return []
    records: list[dict] = []
    for f in sorted(eval_dir.glob("*.json")):
        rec = _load_eval_record(f)
        if rec is None:
            continue
        records.append(rec)
    return records


def _summarize_records(records: list[dict]) -> dict[str, float]:
    """Mean of each metric over the given records."""
    summary: dict[str, float] = {"n": len(records)}
    for m in _METRICS:
        vals = []
        for r in records:
            v = r.get(m)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        summary[m] = _mean(vals)
    return summary


def _bucket_by_config(records: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        cfg = _normalize_config(r.get("config_name", ""))
        buckets[cfg].append(r)
    return buckets


def _agent_name_from_dir(dir_name: str) -> str:
    if dir_name.endswith(JUDGE_SUFFIX):
        return dir_name[: -len(JUDGE_SUFFIX)]
    if "__judge_" in dir_name:
        return dir_name.split("__judge_", 1)[0]
    return dir_name


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _print_table(per_agent: dict[str, dict]) -> None:
    print()
    print("=" * 88)
    print("CAVEAT: INTERIM ESTIMATES FROM A PARTIAL SAMPLE.")
    print("Sample sizes vary across agents; final numbers WILL differ.")
    print("Do not rank-order agents from this output.")
    print("=" * 88)
    print()

    # Per-agent record counts header
    print("## Record counts (per agent)")
    print()
    print(f"{'agent':<22} {'total':>6} {'A':>6} {'B':>6} {'C':>6}")
    print(f"{'-'*22} {'-'*6} {'-'*6} {'-'*6} {'-'*6}")
    # Sort by total records desc so the imbalance is obvious.
    agents_by_count = sorted(
        per_agent.items(), key=lambda kv: -kv[1]["total_records"]
    )
    for agent, info in agents_by_count:
        cc = info["per_config_counts"]
        print(
            f"{agent:<22} {info['total_records']:>6} "
            f"{cc.get('config_A', 0):>6} {cc.get('config_B', 0):>6} "
            f"{cc.get('config_C', 0):>6}"
        )
    print()

    # Per-agent metric table
    for agent, info in agents_by_count:
        print(f"## {agent}  (n_records={info['total_records']})")
        print()
        header = (
            f"{'config':<14} {'n':>4}  "
            f"{'overall':>8} {'detect':>8} {'fpr':>8} "
            f"{'halluc':>8} {'f1':>8} {'cover':>8}"
        )
        print(header)
        print("-" * len(header))
        for cfg in _CONFIGS:
            s = info["per_config"].get(cfg)
            if not s:
                print(f"{cfg:<14} {'-':>4}  {'(no records yet)':>56}")
                continue
            print(
                f"{cfg:<14} {int(s['n']):>4}  "
                f"{_fmt(s['overall_score']):>8} {_fmt(s['detection_rate']):>8} "
                f"{_fmt(s['false_positive_rate']):>8} "
                f"{_fmt(s['hallucination_rate']):>8} {_fmt(s['f1_score']):>8} "
                f"{_fmt(s['coverage']):>8}"
            )
        agg = info["aggregate"]
        print(
            f"{'ALL (A+B+C)':<14} {int(agg['n']):>4}  "
            f"{_fmt(agg['overall_score']):>8} {_fmt(agg['detection_rate']):>8} "
            f"{_fmt(agg['false_positive_rate']):>8} "
            f"{_fmt(agg['hallucination_rate']):>8} {_fmt(agg['f1_score']):>8} "
            f"{_fmt(agg['coverage']):>8}"
        )
        print()

    # Finished-task sets — important for fair comparison
    print("## Finished task_id sets (sorted, for fair-comparison auditing)")
    print()
    for agent, info in agents_by_count:
        tids = sorted(info["task_ids"])
        print(f"### {agent}  ({len(tids)} unique task_ids)")
        # Wrap at ~6 per line for readability
        line: list[str] = []
        for tid in tids:
            line.append(tid)
            if len(line) >= 6:
                print("    " + ", ".join(line))
                line = []
        if line:
            print("    " + ", ".join(line))
        print()

    # Pairwise overlap of completed task_ids — useful sanity check.
    print("## Pairwise task_id overlap (intersection sizes)")
    print()
    names = [a for a, _ in agents_by_count]
    header = "                       " + " ".join(f"{n[:14]:>14}" for n in names)
    print(header)
    for a in names:
        row = f"{a[:22]:<22} "
        for b in names:
            inter = len(per_agent[a]["task_ids"] & per_agent[b]["task_ids"])
            row += f"{inter:>14} "
        print(row)
    print()

    print("=" * 88)
    print("REMINDER: interim estimates only. Final report supersedes this output.")
    print("=" * 88)
    print()


def _write_csv(per_agent: dict[str, dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "agent",
        "config",
        "n_records",
        "overall_score",
        "detection_rate",
        "false_positive_rate",
        "hallucination_rate",
        "f1_score",
        "coverage",
    ]
    rows: list[dict] = []
    # Stable sort: by agent name.
    for agent in sorted(per_agent.keys()):
        info = per_agent[agent]
        for cfg in _CONFIGS:
            s = info["per_config"].get(cfg)
            if not s:
                continue
            rows.append({
                "agent": agent,
                "config": cfg,
                "n_records": int(s["n"]),
                "overall_score": round(s["overall_score"], 6),
                "detection_rate": round(s["detection_rate"], 6),
                "false_positive_rate": round(s["false_positive_rate"], 6),
                "hallucination_rate": round(s["hallucination_rate"], 6),
                "f1_score": round(s["f1_score"], 6),
                "coverage": round(s["coverage"], 6),
            })
        agg = info["aggregate"]
        rows.append({
            "agent": agent,
            "config": "ALL",
            "n_records": int(agg["n"]),
            "overall_score": round(agg["overall_score"], 6),
            "detection_rate": round(agg["detection_rate"], 6),
            "false_positive_rate": round(agg["false_positive_rate"], 6),
            "hallucination_rate": round(agg["hallucination_rate"], 6),
            "f1_score": round(agg["f1_score"], 6),
            "coverage": round(agg["coverage"], 6),
        })
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _verify(per_agent: dict[str, dict]) -> None:
    """Built-in sanity checks. Emits warnings, never raises."""
    for agent, info in per_agent.items():
        per_cfg_sum = sum(info["per_config_counts"].values())
        if per_cfg_sum != info["total_records"]:
            print(
                f"warn: {agent}: per-config record counts sum to "
                f"{per_cfg_sum} but total_records={info['total_records']}",
                file=sys.stderr,
            )
        for cfg, s in info["per_config"].items():
            v = s.get("overall_score", 0.0)
            if not (0.0 <= v <= 1.0):
                print(
                    f"warn: {agent}/{cfg}: overall_score={v:.4f} outside [0,1]",
                    file=sys.stderr,
                )
        v = info["aggregate"].get("overall_score", 0.0)
        if not (0.0 <= v <= 1.0):
            print(
                f"warn: {agent}: aggregate overall_score={v:.4f} outside [0,1]",
                file=sys.stderr,
            )


def main() -> int:
    if not RUNS_ROOT.is_dir():
        print(f"error: runs root not found: {RUNS_ROOT}", file=sys.stderr)
        return 2

    per_agent: dict[str, dict] = {}
    for agent_dir in sorted(RUNS_ROOT.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = _agent_name_from_dir(agent_dir.name)
        records = _collect_agent_records(agent_dir)
        if not records:
            print(f"warn: no records for {agent}", file=sys.stderr)
            continue

        buckets = _bucket_by_config(records)
        per_config: dict[str, dict] = {}
        per_config_counts: dict[str, int] = {}
        for cfg in _CONFIGS:
            rows = buckets.get(cfg, [])
            per_config_counts[cfg] = len(rows)
            if rows:
                per_config[cfg] = _summarize_records(rows)
        aggregate = _summarize_records(records)
        task_ids = {str(r.get("task_id")) for r in records if r.get("task_id")}

        per_agent[agent] = {
            "total_records": len(records),
            "per_config_counts": per_config_counts,
            "per_config": per_config,
            "aggregate": aggregate,
            "task_ids": task_ids,
        }

    if not per_agent:
        print("error: no agents had any eval_results", file=sys.stderr)
        return 2

    _verify(per_agent)
    _print_table(per_agent)
    _write_csv(per_agent, CSV_OUT)
    print(f"wrote interim CSV -> {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
