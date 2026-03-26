from __future__ import annotations

from typing import Any

import numpy as np

from eval_harness.logging_utils import get_logger
from eval_harness.matching import normalize_judge_alignment
from eval_harness.schema import AgentOutput, EvalInput, JudgeOutput


def per_pr_coverage_indicator(caught_human_comments: int) -> float:
    """Per-PR utility: 1.0 if at least one human issue was caught, else 0.0."""
    return 1.0 if int(caught_human_comments or 0) >= 1 else 0.0


def compute_pr_coverage(records: list[dict[str, Any]]) -> float:
    """Aggregate: fraction of PRs with at least one caught human comment (mean of per-PR indicators)."""
    if not records:
        return 0.0
    covered = sum(
        1 for r in records
        if int(r.get("caught_human_comments", 0) or 0) >= 1
    )
    return covered / len(records)


def compute_dimension_scores(
    eval_input: EvalInput,
    agent_output: AgentOutput,
    judge_output: JudgeOutput,
) -> dict[str, float]:
    if not bool(agent_output.parse_success):
        return {
            "detection_rate": 0.0,
            "false_positive_rate": 1.0 if len(agent_output.comments) > 0 else 0.0,
            "hallucination_rate": 1.0 if len(agent_output.comments) > 0 else 0.0,
            "severity_accuracy": -1.0 if eval_input.has_severity_annotations else -1.0,
            "actionability_score": 0.0,
            "semantic_alignment": 0.0,
            "adjacent_detection_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1_score": 0.0,
            "redundancy_penalty": 1.0,
            "efficiency": 0.0,
            "plausible_penalty": 1.0,
            "fallback_penalty": 0.0,
            "judge_alignment_agreement": 0.0,
        }

    scores: dict[str, float] = {}
    normalized = normalize_judge_alignment(eval_input, agent_output, judge_output)
    total_human = len(eval_input.human_comments)
    total_agent = len(agent_output.comments)
    confirmed = len(normalized.confirmed_pairs)
    fabricated = sum(
        1 for _id, cls in normalized.agent_classification_by_id.items() if cls == "FABRICATED"
    )
    caught = sum(1 for _id, st in normalized.human_status_by_id.items() if st == "CAUGHT")
    plausible = sum(1 for _id, cls in normalized.agent_classification_by_id.items() if cls == "PLAUSIBLE")

    recall = (caught / total_human) if total_human > 0 else 0.0
    precision = (confirmed / total_agent) if total_agent > 0 else 0.0
    pr_denom = precision + recall
    f1_score = (2.0 * precision * recall / pr_denom) if pr_denom > 0 else 0.0
    duplicates = max(0, total_agent - int(normalized.unique_agent_cluster_count))
    redundancy_penalty = (duplicates / total_agent) if total_agent > 0 else 0.0
    if duplicates > 0:
        log = get_logger()
        log.info(
            "redundancy_detected",
            task_id=eval_input.task_id,
            config=eval_input.config_name,
            duplicates=duplicates,
            total_agent=total_agent,
            unique_clusters=normalized.unique_agent_cluster_count,
        )
    hallucination_rate = (fabricated / total_agent) if total_agent > 0 else 0.0
    if total_agent < 3:
        plausible_penalty = 0.0
    else:
        ratio = (plausible / total_agent) if total_agent > 0 else 0.0
        plausible_penalty = max(0.0, ratio - 0.7)

    if eval_input.has_severity_annotations:
        scores["severity_accuracy"] = _compute_severity_accuracy(eval_input, agent_output, normalized.confirmed_pairs)
    else:
        scores["severity_accuracy"] = -1.0

    if total_agent > 0:
        actionability_by_agent = {c.comment_id: c.actionability_score for c in judge_output.agent_classifications}
        raw_scores = [
            int(actionability_by_agent.get(aid, 3))
            for aid in normalized.agent_classification_by_id.keys()
        ]
        actionability = float((np.mean(raw_scores) - 1.0) / 4.0)
    else:
        actionability = 0.0

    semantic_alignment = _compute_semantic_alignment(
        normalized.confirmed_pairs, agent_output, eval_input
    )
    if (
        eval_input.difficulty in ("Type2_Contextual", "Type3_Latent_Candidate")
        and eval_input.config_name == "config_C_full_context"
        and len(agent_output.comments) > 0
    ):
        outside_diff = sum(1 for c in agent_output.comments if c.is_outside_diff)
        adjacent_detection_rate = outside_diff / len(agent_output.comments)
    else:
        adjacent_detection_rate = 0.0

    if str(judge_output.judge_prompt_version or "").endswith("+fallback"):
        fallback_penalty = 0.5
    else:
        fallback_penalty = 0.0

    scores["detection_rate"] = float(recall)
    scores["recall"] = float(recall)
    scores["precision"] = float(precision)
    scores["f1_score"] = float(f1_score)
    scores["semantic_alignment"] = float(semantic_alignment)
    scores["actionability_score"] = float(actionability)
    scores["adjacent_detection_rate"] = float(adjacent_detection_rate)
    scores["hallucination_rate"] = float(hallucination_rate)
    scores["false_positive_rate"] = float(hallucination_rate)
    scores["redundancy_penalty"] = float(redundancy_penalty)
    efficiency = (confirmed / total_agent) if total_agent > 0 else 0.0
    scores["efficiency"] = float(efficiency)
    scores["plausible_penalty"] = float(plausible_penalty)
    scores["fallback_penalty"] = float(fallback_penalty)
    scores["judge_alignment_agreement"] = float(
        _compute_judge_alignment_agreement(
            normalized.confirmed_pairs, agent_output, eval_input
        )
    )
    scores["confirmed_count"] = float(confirmed)
    scores["plausible_count"] = float(plausible)
    scores["fabricated_count"] = float(fabricated)
    scores["caught_human_comments"] = float(caught)
    scores["missed_human_comments"] = float(max(0, total_human - caught))
    if confirmed == 0 and total_agent > 0:
        log = get_logger()
        log.warning(
            "no_matches_detected",
            task_id=eval_input.task_id,
            config=eval_input.config_name,
            total_agent=total_agent,
            total_human=total_human,
        )
    return scores


def _compute_severity_accuracy(
    eval_input: EvalInput,
    agent_output: AgentOutput,
    confirmed_pairs: list[tuple[str, str, int, float]],
) -> float:
    human_severity_map = {c.get("comment_id"): c.get("severity") for c in eval_input.human_comments}
    agent_map = {c.comment_id: c for c in agent_output.comments}
    correct = 0
    total = 0
    for a_id, h_id, _actionability, _sim in confirmed_pairs:
        human_sev = human_severity_map.get(h_id)
        agent_comment = agent_map.get(a_id)
        if not human_sev or not agent_comment or not agent_comment.severity_claim:
            continue
        total += 1
        if str(agent_comment.severity_claim).upper() == str(human_sev).upper():
            correct += 1
    return (correct / total) if total > 0 else -1.0


def _compute_semantic_alignment(
    confirmed_pairs: list[tuple[str, str, int, float]],
    agent_output: AgentOutput,
    eval_input: EvalInput,
) -> float:
    """Compute real embedding-based similarity for matched pairs. No floor."""
    if not confirmed_pairs:
        return 0.0
    agent_map = {c.comment_id: str(c.body or "").strip() for c in agent_output.comments}
    human_map = {
        str(c.get("comment_id") or ""): str(c.get("body") or "").strip()
        for c in eval_input.human_comments
    }
    similarities: list[float] = []
    try:
        from sentence_transformers import SentenceTransformer, util

        from eval_harness.matching import SEMANTIC_EMBEDDING_MODEL

        model = SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)
        for a_id, h_id, _actionability, _ in confirmed_pairs:
            a_text = agent_map.get(a_id, "")
            h_text = human_map.get(h_id, "")
            if not a_text or not h_text:
                continue
            a_emb = model.encode([a_text], convert_to_tensor=True)
            h_emb = model.encode([h_text], convert_to_tensor=True)
            sim = float(util.cos_sim(a_emb, h_emb)[0][0].item())
            similarities.append(sim)
    except Exception:
        similarities = [float(sim) for _a, _h, _actionability, sim in confirmed_pairs]
    return float(np.mean(similarities)) if similarities else 0.0


def _compute_judge_alignment_agreement(
    confirmed_pairs: list[tuple[str, str, int, float]],
    agent_output: AgentOutput,
    eval_input: EvalInput,
    threshold: float = 0.12,
) -> float:
    """Fraction of judge-CONFIRMED pairs with embedding similarity >= threshold."""
    if not confirmed_pairs:
        return 0.0
    agent_map = {c.comment_id: str(c.body or "").strip() for c in agent_output.comments}
    human_map = {
        str(c.get("comment_id") or ""): str(c.get("body") or "").strip()
        for c in eval_input.human_comments
    }
    agreed = 0
    try:
        from sentence_transformers import SentenceTransformer, util

        from eval_harness.matching import SEMANTIC_EMBEDDING_MODEL

        model = SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)
        for a_id, h_id, _actionability, _ in confirmed_pairs:
            a_text = agent_map.get(a_id, "")
            h_text = human_map.get(h_id, "")
            if not a_text or not h_text:
                continue
            a_emb = model.encode([a_text], convert_to_tensor=True)
            h_emb = model.encode([h_text], convert_to_tensor=True)
            sim = float(util.cos_sim(a_emb, h_emb)[0][0].item())
            if sim >= threshold:
                agreed += 1
    except Exception:
        for _a, _h, _actionability, sim in confirmed_pairs:
            if float(sim) >= threshold:
                agreed += 1
    return agreed / len(confirmed_pairs) if confirmed_pairs else 0.0

