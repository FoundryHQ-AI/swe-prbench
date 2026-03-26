from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from eval_harness.logging_utils import get_logger
from eval_harness.schema import AgentOutput, EvalInput, JudgeOutput


@dataclass
class NormalizedMatchView:
    confirmed_pairs: list[tuple[str, str, int, float]]
    agent_classification_by_id: dict[str, str]
    human_status_by_id: dict[str, str]
    human_matched_agent_by_id: dict[str, str | None]
    pair_similarity_by_agent_id: dict[str, float]
    duplicate_confirmed_count: int
    low_quality_plausible_count: int
    unique_agent_cluster_count: int


def normalize_judge_alignment(
    eval_input: EvalInput,
    agent_output: AgentOutput,
    judge_output: JudgeOutput,
) -> NormalizedMatchView:
    agent_ids = [c.comment_id for c in agent_output.comments]
    human_ids = [str(c.get("comment_id") or "") for c in eval_input.human_comments]
    agent_text = {c.comment_id: str(c.body or "") for c in agent_output.comments}
    human_text = {str(c.get("comment_id") or ""): str(c.get("body") or "") for c in eval_input.human_comments}
    human_file_by_id = {str(c.get("comment_id") or ""): str(c.get("file") or "") for c in eval_input.human_comments}
    human_line_by_id = {str(c.get("comment_id") or ""): c.get("line") for c in eval_input.human_comments}
    agent_file_by_id = {c.comment_id: str(c.file_reference or "") for c in agent_output.comments}
    agent_line_by_id = {c.comment_id: c.line_reference for c in agent_output.comments}
    raw_cls_by_id = {str(c.comment_id or ""): str(c.classification or "").upper() for c in judge_output.agent_classifications}
    actionability_by_id = {str(c.comment_id or ""): int(c.actionability_score) for c in judge_output.agent_classifications}

    pair_sims = _build_pair_similarity(agent_text, human_text)
    sim_threshold = 0.3

    # Stage 1 (primary): preserve judge-confirmed signal with 1:1 normalization.
    judge_confirmed: list[tuple[float, int, str, str]] = []
    for cls in judge_output.agent_classifications:
        if str(cls.classification or "").upper() != "CONFIRMED" or not cls.matched_human_comment_id:
            continue
        a_id = str(cls.comment_id or "")
        h_id = str(cls.matched_human_comment_id or "")
        if a_id not in agent_text or h_id not in human_text:
            continue
        sim = float(pair_sims.get((a_id, h_id), 0.0))
        judge_confirmed.append((sim, int(cls.actionability_score), a_id, h_id))
    judge_confirmed.sort(key=lambda x: (x[1], x[0]), reverse=True)

    used_agents: set[str] = set()
    used_humans: set[str] = set()
    confirmed_pairs: list[tuple[str, str, int, float]] = []
    for sim, actionability, a_id, h_id in judge_confirmed:
        if a_id in used_agents or h_id in used_humans:
            continue
        used_agents.add(a_id)
        used_humans.add(h_id)
        confirmed_pairs.append((a_id, h_id, int(actionability), sim))

    # Stage 2 (backup): semantic matching for remaining unmatched pairs.
    semantic_candidates: list[tuple[float, str, str]] = []
    for a_id in agent_ids:
        if a_id in used_agents:
            continue
        for h_id in human_ids:
            if h_id in used_humans:
                continue
            raw_sim = float(pair_sims.get((a_id, h_id), 0.0))
            affinity = _pair_affinity(
                raw_sim=raw_sim,
                agent_file=agent_file_by_id.get(a_id),
                human_file=human_file_by_id.get(h_id),
                agent_line=agent_line_by_id.get(a_id),
                human_line=human_line_by_id.get(h_id),
            )
            if affinity >= sim_threshold:
                semantic_candidates.append((affinity, a_id, h_id))
    semantic_candidates.sort(key=lambda x: x[0], reverse=True)
    for _affinity, a_id, h_id in semantic_candidates:
        if a_id in used_agents or h_id in used_humans:
            continue
        used_agents.add(a_id)
        used_humans.add(h_id)
        confirmed_pairs.append((a_id, h_id, int(actionability_by_id.get(a_id, 3)), float(pair_sims.get((a_id, h_id), 0.0))))

    judge_confirmed_total = sum(1 for cls in judge_output.agent_classifications if str(cls.classification).upper() == "CONFIRMED")
    duplicate_confirmed_count = max(0, int(judge_confirmed_total) - len(confirmed_pairs))

    agent_classification_by_id: dict[str, str] = {}
    selected_agent_ids = {a for a, _h, _actionability, _sim in confirmed_pairs}
    pair_similarity_by_agent_id = {a: float(sim) for a, _h, _actionability, sim in confirmed_pairs}
    low_quality_plausible_count = 0
    for a_id in agent_ids:
        raw = str(raw_cls_by_id.get(a_id) or "PLAUSIBLE").upper()
        if a_id in selected_agent_ids:
            agent_classification_by_id[a_id] = "CONFIRMED"
            continue
        if raw == "FABRICATED":
            agent_classification_by_id[a_id] = "FABRICATED"
            continue
        # Judge non-fabricated unmatched comments are treated as plausible noise.
        agent_classification_by_id[a_id] = "PLAUSIBLE"
        best_sim = (
            max(float(pair_sims.get((a_id, h_id), 0.0)) for h_id in human_ids)
            if human_ids
            else 0.0
        )
        if best_sim < 0.2:
            low_quality_plausible_count += 1

    human_status_by_id = {h: "MISSED" for h in human_ids}
    human_matched_agent_by_id = {h: None for h in human_ids}
    for a_id, h_id, _actionability, _sim in confirmed_pairs:
        if h_id in human_status_by_id:
            human_status_by_id[h_id] = "CAUGHT"
            human_matched_agent_by_id[h_id] = a_id

    unique_agent_cluster_count = _count_unique_agent_clusters(agent_output)

    return NormalizedMatchView(
        confirmed_pairs=confirmed_pairs,
        agent_classification_by_id=agent_classification_by_id,
        human_status_by_id=human_status_by_id,
        human_matched_agent_by_id=human_matched_agent_by_id,
        pair_similarity_by_agent_id=pair_similarity_by_agent_id,
        duplicate_confirmed_count=duplicate_confirmed_count,
        low_quality_plausible_count=low_quality_plausible_count,
        unique_agent_cluster_count=unique_agent_cluster_count,
    )


def _build_pair_similarity(
    agent_text_by_id: dict[str, str],
    human_text_by_id: dict[str, str],
) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    if not agent_text_by_id or not human_text_by_id:
        return out
    agent_ids = list(agent_text_by_id.keys())
    human_ids = list(human_text_by_id.keys())
    agent_texts = [agent_text_by_id[a] for a in agent_ids]
    human_texts = [human_text_by_id[h] for h in human_ids]
    try:
        model = _get_embedding_model()
        a_emb = model.encode(agent_texts)
        h_emb = model.encode(human_texts)
        for i, a_id in enumerate(agent_ids):
            for j, h_id in enumerate(human_ids):
                sim = _cosine(a_emb[i], h_emb[j])
                out[(a_id, h_id)] = float(sim)
        return out
    except Exception:
        # Fallback to lexical similarity when embeddings are unavailable.
        for a_id in agent_ids:
            for h_id in human_ids:
                out[(a_id, h_id)] = float(_lexical_similarity(agent_text_by_id[a_id], human_text_by_id[h_id]))
        return out


def _pair_affinity(
    raw_sim: float,
    agent_file: str | None,
    human_file: str | None,
    agent_line,
    human_line,
) -> float:
    score = float(raw_sim)
    af = str(agent_file or "")
    hf = str(human_file or "")
    if af and hf and af == hf:
        score += 0.15
        try:
            if agent_line is not None and human_line is not None:
                dist = abs(int(agent_line) - int(human_line))
                if dist <= 8:
                    score += 0.10
                elif dist <= 20:
                    score += 0.05
        except Exception:
            pass
    return max(0.0, min(1.0, score))


def _count_unique_agent_clusters(agent_output: AgentOutput) -> int:
    if not agent_output.comments:
        return 0
    clusters: list[list[int]] = []
    comments = list(agent_output.comments)
    embeddings = _embed_agent_comments(comments)
    for idx, c in enumerate(comments):
        placed = False
        for cluster in clusters:
            rep_idx = cluster[0]
            rep = comments[rep_idx]
            # Prefer same-file near-duplicate clustering.
            same_file = (c.file_reference or "") == (rep.file_reference or "")
            sim = _comment_similarity(c.body, rep.body, embeddings[idx], embeddings[rep_idx])
            if sim >= 0.65:
                cluster.append(idx)
                placed = True
                break
        if not placed:
            clusters.append([idx])
    return len(clusters)


def _embed_agent_comments(comments) -> list[np.ndarray | None]:
    texts = [str(c.body or "") for c in comments]
    try:
        model = _get_embedding_model()
        emb = model.encode(texts)
        return [np.asarray(v) for v in emb]
    except Exception:
        return [None for _ in texts]


def _comment_similarity(text_a: str, text_b: str, emb_a, emb_b) -> float:
    if emb_a is not None and emb_b is not None:
        return float(_cosine(emb_a, emb_b))
    return float(_lexical_similarity(text_a, text_b))


def _cosine(a, b) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# Code-focused embedding model for semantic scoring (better for code review comments)
SEMANTIC_EMBEDDING_MODEL = "all-mpnet-base-v2"


@lru_cache(maxsize=1)
def _get_embedding_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(SEMANTIC_EMBEDDING_MODEL)


def _lexical_similarity(a: str, b: str) -> float:
    ta = set(re.findall(r"[A-Za-z0-9_]+", str(a).lower()))
    tb = set(re.findall(r"[A-Za-z0-9_]+", str(b).lower()))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter / union) if union else 0.0
