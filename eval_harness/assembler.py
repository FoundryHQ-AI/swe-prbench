from __future__ import annotations

import math

from eval_harness.matching import normalize_judge_alignment
from eval_harness.scorer import per_pr_coverage_indicator
from eval_harness.schema import AgentOutput, EvalInput, EvalResult, JudgeOutput


def compute_overall_score(scores: dict[str, float]) -> float:
    confirmed = int(scores.get("confirmed_count", 0))
    actionability = float(scores.get("actionability_score", 0.0))
    # When confirmed=0, actionability contributes at 20% weight only (plausible-only discount)
    if confirmed == 0:
        actionability *= 0.2
    val = (
        (0.4 * float(scores.get("recall", 0.0)))
        + (0.25 * float(scores.get("precision", 0.0)))
        + (0.15 * float(scores.get("semantic_alignment", 0.0)))
        + (0.10 * actionability)
        + (0.05 * float(scores.get("efficiency", 0.0)))
        - (0.25 * float(scores.get("hallucination_rate", 0.0)))
        - (0.15 * float(scores.get("redundancy_penalty", 0.0)))
        - (0.10 * float(scores.get("plausible_penalty", 0.0)))
    )
    if float(scores.get("fallback_penalty", 0.0)) > 0.0:
        val *= 0.5
    return max(0.0, min(1.0, float(val)))


def assemble_eval_result(
    eval_input: EvalInput,
    agent_output: AgentOutput,
    judge_output: JudgeOutput,
    scores: dict[str, float],
    model: str,
) -> EvalResult:
    normalized = normalize_judge_alignment(eval_input, agent_output, judge_output)
    total_agent_comments = len(agent_output.comments)
    if not bool(agent_output.parse_success):
        overall = 0.0
    elif total_agent_comments == 0:
        overall = 0.0  # hard zero for no_attempt — no partial credit for not trying
    else:
        overall = compute_overall_score(scores)
        # No floor: plausible-only records get small positive from discounted actionability only
    matched_pairs = [
        {"agent_comment_id": a_id, "human_comment_id": h_id, "actionability": actionability}
        for a_id, h_id, actionability, _sim in normalized.confirmed_pairs
    ]
    missed = [
        {"comment_id": h_id}
        for h_id, status in normalized.human_status_by_id.items()
        if status == "MISSED"
    ]
    fabricated = [
        {"comment_id": a_id}
        for a_id, cls in normalized.agent_classification_by_id.items()
        if cls == "FABRICATED"
    ]
    confirmed_count = sum(1 for _aid, cls in normalized.agent_classification_by_id.items() if cls == "CONFIRMED")
    plausible_count = sum(1 for _aid, cls in normalized.agent_classification_by_id.items() if cls == "PLAUSIBLE")
    fabricated_count = sum(1 for _aid, cls in normalized.agent_classification_by_id.items() if cls == "FABRICATED")
    caught_human_comments = sum(1 for _hid, status in normalized.human_status_by_id.items() if status == "CAUGHT")
    total_human_comments = len(eval_input.human_comments)
    attempt_rate = 1.0 if total_agent_comments > 0 else 0.0
    coverage = per_pr_coverage_indicator(caught_human_comments)
    difficulty_weight = math.log(total_human_comments + 1)
    failure_type = "no_attempt" if total_agent_comments == 0 else None
    if total_human_comments > 15:
        pr_difficulty_tag = "hard"
    elif total_human_comments > 8:
        pr_difficulty_tag = "medium"
    else:
        pr_difficulty_tag = "easy"

    return EvalResult(
        task_id=eval_input.task_id,
        pr_number=eval_input.pr_number,
        config_name=eval_input.config_name,
        model=model,
        pipeline_version=eval_input.pipeline_version,
        difficulty=eval_input.difficulty,
        language=eval_input.language,
        detection_rate=float(scores.get("detection_rate", 0.0)),
        false_positive_rate=float(scores.get("false_positive_rate", 0.0)),
        severity_accuracy=float(scores.get("severity_accuracy", -1.0)),
        actionability_score=float(scores.get("actionability_score", 0.0)),
        semantic_alignment=float(scores.get("semantic_alignment", 0.0)),
        adjacent_detection_rate=float(scores.get("adjacent_detection_rate", 0.0)),
        overall_score=overall,
        precision=float(scores.get("precision", 0.0)),
        recall=float(scores.get("recall", 0.0)),
        f1_score=float(scores.get("f1_score", 0.0)),
        hallucination_rate=float(scores.get("hallucination_rate", scores.get("false_positive_rate", 0.0))),
        redundancy_penalty=float(scores.get("redundancy_penalty", 0.0)),
        plausible_penalty=float(scores.get("plausible_penalty", 0.0)),
        fallback_penalty=float(scores.get("fallback_penalty", 0.0)),
        judge_alignment_agreement=float(scores.get("judge_alignment_agreement", 0.0)),
        attempt_rate=attempt_rate,
        coverage=coverage,
        difficulty_weight=difficulty_weight,
        failure_type=failure_type,
        pr_difficulty_tag=pr_difficulty_tag,
        total_agent_comments=total_agent_comments,
        confirmed_count=confirmed_count,
        plausible_count=plausible_count,
        fabricated_count=fabricated_count,
        caught_human_comments=caught_human_comments,
        missed_human_comments=max(0, total_human_comments - caught_human_comments),
        total_human_comments=total_human_comments,
        matched_pairs=matched_pairs,
        missed_comments=missed,
        fabricated_comments=fabricated,
        agent_parse_success=bool(agent_output.parse_success),
        agent_parse_error=agent_output.parse_error,
        judge_parse_fallback_used=bool(str(judge_output.judge_prompt_version).endswith("+fallback")),
    )

