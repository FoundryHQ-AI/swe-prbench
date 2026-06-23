#!/usr/bin/env python3
"""Interim score estimates restricted to the COMMON TASK SUBSET.

Sister script to ``interim_estimates.py``. Whereas that script summarizes
each agent's metrics over its full partial sample (so sample sizes differ
across agents and the comparison is unfair), this one computes the
intersection of ``(task_id, normalized_config)`` pairs across ALL agents
and reports per-agent means only over that common set. Same PRs, same
contexts, evaluated by every agent — a fair per-agent comparison.

It is safe to re-run while the eval workers are paused or still writing:
each JSON file is read once and parse errors are skipped with a warning.

The reported numbers are NOT final. They are computed from a partial
sample, restricted to the subset of pairs that ALL 4 agents have already
completed. Treat as directional signal only.
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
CSV_OUT = REPO_ROOT / "claude_code_eval" / "results" / "interim_estimates_common.csv"

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


def _agent_name_from_dir(dir_name: str) -> str:
    if dir_name.endswith(JUDGE_SUFFIX):
        return dir_name[: -len(JUDGE_SUFFIX)]
    if "__judge_" in dir_name:
        return dir_name.split("__judge_", 1)[0]
    return dir_name


def _record_pair_key(rec: dict) -> tuple[str, str] | None:
    """The (task_id, normalized_config) identity for a record, or None."""
    tid = rec.get("task_id")
    cfg = _normalize_config(rec.get("config_name", ""))
    if not tid or cfg not in _CONFIGS:
        return None
    return (str(tid), cfg)


def _fmt(v: float) -> str:
    return f"{v:.3f}"


def _print_table(
    per_agent: dict[str, dict],
    common_pairs: set[tuple[str, str]],
    common_task_ids: set[str],
) -> None:
    print()
    print("=" * 88)
    print("CAVEAT: INTERIM ESTIMATES, COMMON TASK SUBSET (intersection across agents).")
    print("Same (task_id, config) pairs evaluated by every agent — fair comparison.")
    print("Numbers will shift as more PRs finish; do not treat as final.")
    print("=" * 88)
    print()

    print("## Common subset size")
    print()
    n_pairs = len(common_pairs)
    n_tids = len(common_task_ids)
    print(f"  unique task_ids in intersection : {n_tids}")
    print(f"  total (task_id, config) pairs   : {n_pairs}  (expected {n_tids} x 3 = {n_tids * 3})")
    if n_pairs != n_tids * 3:
        print("  note: pair count != task_ids x 3 — some configs missing for some tasks")
    print()

    # Record-count comparison: full vs common subset.
    print("## Record counts: full sample vs common subset")
    print()
    header = f"{'agent':<22} {'full':>8} {'common':>8}"
    print(header)
    print("-" * len(header))
    agents_sorted = sorted(per_agent.keys())
    for agent in agents_sorted:
        info = per_agent[agent]
        print(
            f"{agent:<22} {info['total_records']:>8} "
            f"{info['common_total']:>8}"
        )
    print()

    # Per-agent metric tables over the common subset.
    for agent in agents_sorted:
        info = per_agent[agent]
        print(f"## {agent}  (common n={info['common_total']})")
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
                print(f"{cfg:<14} {'-':>4}  {'(no records in common subset)':>56}")
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


def _verify(per_agent: dict[str, dict], expected_common_size: int) -> None:
    """Built-in sanity checks. Emits warnings, never raises."""
    for agent, info in per_agent.items():
        if info["common_total"] != expected_common_size:
            print(
                f"warn: {agent}: common_total={info['common_total']} "
                f"!= intersection size {expected_common_size}",
                file=sys.stderr,
            )
        per_cfg_sum = sum(
            int(s["n"]) for s in info["per_config"].values()
        )
        if per_cfg_sum != info["common_total"]:
            print(
                f"warn: {agent}: per-config record counts sum to "
                f"{per_cfg_sum} but common_total={info['common_total']}",
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

    # Pass 1: load every agent's records, keep them in memory, and build
    # each agent's set of (task_id, normalized_config) pairs.
    raw_records: dict[str, list[dict]] = {}
    pair_sets: dict[str, set[tuple[str, str]]] = {}
    for agent_dir in sorted(RUNS_ROOT.iterdir()):
        if not agent_dir.is_dir():
            continue
        agent = _agent_name_from_dir(agent_dir.name)
        records = _collect_agent_records(agent_dir)
        if not records:
            print(f"warn: no records for {agent}", file=sys.stderr)
            continue
        raw_records[agent] = records
        pair_sets[agent] = set()
        for r in records:
            key = _record_pair_key(r)
            if key is not None:
                pair_sets[agent].add(key)

    if not raw_records:
        print("error: no agents had any eval_results", file=sys.stderr)
        return 2

    # Intersection across ALL agents.
    common_pairs: set[tuple[str, str]] = set.intersection(*pair_sets.values())
    common_task_ids: set[str] = {tid for tid, _ in common_pairs}
    if not common_pairs:
        print("error: empty intersection — no (task_id, config) pair "
              "is completed by every agent", file=sys.stderr)
        return 3

    # Pass 2: per agent, filter records down to the intersection, then
    # summarize per-config and aggregate.
    per_agent: dict[str, dict] = {}
    for agent, records in raw_records.items():
        kept: list[dict] = []
        per_config_records: dict[str, list[dict]] = defaultdict(list)
        for r in records:
            key = _record_pair_key(r)
            if key is None or key not in common_pairs:
                continue
            kept.append(r)
            per_config_records[key[1]].append(r)

        per_config: dict[str, dict] = {}
        for cfg in _CONFIGS:
            rows = per_config_records.get(cfg, [])
            if rows:
                per_config[cfg] = _summarize_records(rows)
        aggregate = _summarize_records(kept)

        per_agent[agent] = {
            "total_records": len(records),  # full-sample size, for context
            "common_total": len(kept),
            "per_config": per_config,
            "aggregate": aggregate,
        }

    _verify(per_agent, expected_common_size=len(common_pairs))
    _print_table(per_agent, common_pairs, common_task_ids)
    _write_csv(per_agent, CSV_OUT)
    print(f"wrote common-subset CSV -> {CSV_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
