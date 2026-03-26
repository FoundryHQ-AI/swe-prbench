from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime

from eval_harness.io_utils import write_json
from eval_harness.schema import EvalResult


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _weighted_mean(
    records: list[EvalResult], score_attr: str = "overall_score", weight_attr: str = "difficulty_weight"
) -> float:
    """Difficulty-weighted aggregate: harder PRs count more."""
    total_weight = sum(getattr(r, weight_attr, 0.0) for r in records)
    if total_weight <= 0:
        return _mean([getattr(r, score_attr, 0.0) for r in records])
    weighted_sum = sum(
        getattr(r, score_attr, 0.0) * getattr(r, weight_attr, 0.0) for r in records
    )
    return float(weighted_sum / total_weight)


def _most_common(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0] if values else "easy"


def _normalize_config_name(config_name: str) -> str:
    mapping = {
        "config_A_diff_only": "config_A",
        "config_B_with_file_content": "config_B",
        "config_C_full_context": "config_C",
        "config_A": "config_A",
        "config_B": "config_B",
        "config_C": "config_C",
    }
    return mapping.get(str(config_name), str(config_name))


def _group_by_config(results: list[EvalResult]) -> dict:
    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        grouped[_normalize_config_name(r.config_name)].append(r)
    out: dict[str, dict] = {}
    for cfg, rows in grouped.items():
        out[cfg] = {
            "n": len(rows),
            "detection_rate": round(_mean([x.detection_rate for x in rows]), 3),
            "false_positive_rate": round(_mean([x.false_positive_rate for x in rows]), 3),
            "precision": round(_mean([getattr(x, "precision", 0.0) for x in rows]), 3),
            "recall": round(_mean([getattr(x, "recall", 0.0) for x in rows]), 3),
            "f1_score": round(_mean([getattr(x, "f1_score", 0.0) for x in rows]), 3),
            "hallucination_rate": round(_mean([getattr(x, "hallucination_rate", x.false_positive_rate) for x in rows]), 3),
            "actionability_score": round(_mean([x.actionability_score for x in rows]), 3),
            "semantic_alignment": round(_mean([x.semantic_alignment for x in rows]), 3),
            "redundancy_penalty": round(_mean([getattr(x, "redundancy_penalty", 0.0) for x in rows]), 3),
            "coverage": round(_mean([getattr(x, "coverage", 0.0) for x in rows]), 3),
            "attempt_rate": round(_mean([getattr(x, "attempt_rate", 1.0) for x in rows]), 3),
            "difficulty_weight": round(_mean([getattr(x, "difficulty_weight", 0.0) for x in rows]), 3),
            "pr_difficulty_tag": _most_common([getattr(x, "pr_difficulty_tag", "easy") for x in rows]),
            "no_attempt_count": sum(1 for x in rows if getattr(x, "failure_type", None) == "no_attempt"),
            "plausible_penalty": round(_mean([getattr(x, "plausible_penalty", 0.0) for x in rows]), 3),
            "fallback_penalty": round(_mean([getattr(x, "fallback_penalty", 0.0) for x in rows]), 3),
            "adjacent_detection_rate": round(_mean([x.adjacent_detection_rate for x in rows]), 3),
            "overall_score": round(_mean([x.overall_score for x in rows]), 3),
            "overall_score_weighted": round(_weighted_mean(rows), 3),
        }
    return out


def _group_by_difficulty(results: list[EvalResult]) -> dict:
    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        grouped[r.difficulty].append(r)
    out: dict[str, dict] = {}
    for diff, rows in grouped.items():
        out[diff] = {
            "n": len(rows),
            "detection_rate": round(_mean([x.detection_rate for x in rows]), 3),
            "coverage": round(_mean([getattr(x, "coverage", 0.0) for x in rows]), 3),
            "f1_score": round(_mean([getattr(x, "f1_score", 0.0) for x in rows]), 3),
            "adjacent_detection_rate": round(_mean([x.adjacent_detection_rate for x in rows]), 3),
            "overall_score": round(_mean([x.overall_score for x in rows]), 3),
            "overall_score_weighted": round(_weighted_mean(rows), 3),
        }
    return out


def _compute_ablation_table(results: list[EvalResult]) -> dict:
    table: dict[str, dict] = {}
    configs = ["config_A", "config_B", "config_C"]
    difficulties = ["Type1_Direct", "Type2_Contextual", "Type3_Latent_Candidate"]
    for difficulty in difficulties:
        table[difficulty] = {}
        for config in configs:
            subset = [
                r
                for r in results
                if r.difficulty == difficulty and _normalize_config_name(r.config_name) == config
            ]
            if not subset:
                continue
            table[difficulty][config] = {
                "n": len(subset),
                "detection_rate": round(_mean([r.detection_rate for r in subset]), 3),
                "actionability_score": round(_mean([r.actionability_score for r in subset]), 3),
                "adjacent_detection_rate": round(_mean([r.adjacent_detection_rate for r in subset]), 3),
                "overall_score": round(_mean([r.overall_score for r in subset]), 3),
            }
    return table


def generate_eval_report(results: list[EvalResult], output_path: str) -> dict:
    report = {
        "run_date": datetime.utcnow().isoformat(),
        "total_prs": len(set(r.pr_number for r in results)),
        "total_eval_records": len(results),
        "by_config": _group_by_config(results),
        "by_difficulty": _group_by_difficulty(results),
        "ablation_table": _compute_ablation_table(results),
        # Parse failures should reflect parser/request failures, not legitimate
        # empty review outputs ("[]").
        "agent_parse_failures": sum(1 for r in results if not bool(getattr(r, "agent_parse_success", True))),
        "judge_parse_fallback_count": sum(
            1 for r in results if bool(getattr(r, "judge_parse_fallback_used", False))
        ),
        "records": [asdict(r) for r in results],
    }
    write_json(report, output_path)
    return report

