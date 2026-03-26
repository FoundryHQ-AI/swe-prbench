from __future__ import annotations

from eval_harness.schema import EvalResult


def validate_eval_result(result: EvalResult) -> list[str]:
    failures: list[str] = []
    for dim in (
        "detection_rate",
        "false_positive_rate",
        "actionability_score",
        "semantic_alignment",
        "adjacent_detection_rate",
        "precision",
        "recall",
        "f1_score",
        "hallucination_rate",
        "redundancy_penalty",
        "plausible_penalty",
        "fallback_penalty",
        "judge_alignment_agreement",
    ):
        val = float(getattr(result, dim))
        if not (0.0 <= val <= 1.0):
            failures.append(f"out_of_range_{dim}: {val}")

    if result.config_name in ("config_A_diff_only", "config_B_with_file_content"):
        if float(result.adjacent_detection_rate) > 0.0:
            failures.append(f"D6_nonzero_in_{result.config_name}: {result.adjacent_detection_rate}")

    if not (0.0 <= float(result.overall_score) <= 1.0):
        failures.append(f"overall_score_out_of_range: {result.overall_score}")

    if (result.caught_human_comments + result.missed_human_comments) != result.total_human_comments:
        failures.append("human_comment_count_mismatch")

    if (
        result.confirmed_count + result.plausible_count + result.fabricated_count
    ) != result.total_agent_comments:
        failures.append("agent_comment_classification_count_mismatch")

    return failures

