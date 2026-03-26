"""
Cross-judge validation: rerun only the judge (e.g. Claude Sonnet) per agent comment
on a stratified PR sample, and compare labels to the existing primary judge (e.g. GPT-5.2).

Does not rerun agents. Expects a completed model run directory with eval_report.json,
agent_outputs/, and judge_outputs/.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval_harness.loader import load_eval_input, normalize_config_name
from eval_harness.schema import AgentComment, AgentOutput

# Align with eval_harness/judge.py rubric (abbreviated for single-comment calls).
_SINGLE_COMMENT_SYSTEM = """You are an expert evaluator assessing one AI code review comment against human expert review.

Use these definitions (exactly one label):
- CONFIRMED: The agent comment addresses the same underlying issue as a human comment, even if phrased differently. Set this only if it matches a human concern.
- PLAUSIBLE: Grounded in the code shown, factually reasonable, but no human comment raised this specific concern.
- FABRICATED: Factual errors about the code, references code not in context, or describes a bug that does not exist. Not being in ground truth alone is NOT sufficient for FABRICATED.

Respond with exactly one word on the first line: CONFIRMED, PLAUSIBLE, or FABRICATED"""


def stem_for_record(task_id: str, config_name: str) -> str:
    """File stem used by run_eval: {task_id}_{full_config_name}."""
    full = normalize_config_name(config_name)
    return f"{task_id}_{full}"


def stratified_task_ids(
    records: list[dict[str, Any]],
    *,
    seed: int = 42,
    n_type1: int = 8,
    n_type2: int = 8,
    n_type3: int = 4,
) -> list[str]:
    type1 = list({r["task_id"] for r in records if r.get("difficulty") == "Type1_Direct"})
    type2 = list({r["task_id"] for r in records if r.get("difficulty") == "Type2_Contextual"})
    type3 = list({r["task_id"] for r in records if r.get("difficulty") == "Type3_Latent_Candidate"})
    rng = random.Random(seed)

    def sample(pool: list[str], k: int) -> list[str]:
        k = min(k, len(pool))
        if k == 0:
            return []
        return rng.sample(pool, k)

    out = sample(type1, n_type1) + sample(type2, n_type2) + sample(type3, n_type3)
    if not out:
        raise ValueError("Stratified sample is empty — check eval_report records and difficulty fields.")
    return out


def load_agent_output_dict(path: Path) -> AgentOutput:
    raw = json.loads(path.read_text(encoding="utf-8"))
    comments: list[AgentComment] = []
    for c in raw.get("comments") or []:
        comments.append(
            AgentComment(
                comment_id=str(c.get("comment_id") or ""),
                body=str(c.get("body") or ""),
                file_reference=c.get("file_reference"),
                line_reference=c.get("line_reference"),
                severity_claim=c.get("severity_claim"),
                is_outside_diff=bool(c.get("is_outside_diff", False)),
            )
        )
    return AgentOutput(
        task_id=str(raw.get("task_id") or ""),
        config_name=str(raw.get("config_name") or ""),
        model=str(raw.get("model") or ""),
        raw_response=str(raw.get("raw_response") or ""),
        comments=comments,
        parse_success=bool(raw.get("parse_success", True)),
        parse_error=raw.get("parse_error"),
    )


def load_baseline_judge_labels(judge_path: Path) -> dict[str, str]:
    """comment_id -> CONFIRMED | PLAUSIBLE | FABRICATED from existing judge_outputs JSON."""
    data = json.loads(judge_path.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for item in data.get("agent_classifications") or []:
        cid = str(item.get("comment_id") or "")
        cls = str(item.get("classification") or "").strip().upper()
        if cid and cls in {"CONFIRMED", "PLAUSIBLE", "FABRICATED"}:
            out[cid] = cls
    return out


def build_single_comment_user_message(
    *,
    diff_patch: str,
    human_comments: list[dict[str, Any]],
    agent_comment: AgentComment,
    max_diff_chars: int = 12000,
) -> str:
    diff = (diff_patch or "")[:max_diff_chars]
    human_section = "## Human Expert Review Comments (Ground Truth)\n"
    for c in human_comments:
        human_section += (
            f"- comment_id: {c.get('comment_id')}\n"
            f"  file: {c.get('file', 'general')}\n"
            f"  line: {c.get('line', 'N/A')}\n"
            f"  body: {c.get('body', '')}\n\n"
        )
    ac = agent_comment
    agent_section = (
        "## AI Agent Comment (evaluate only this one)\n"
        f"- comment_id: {ac.comment_id}\n"
        f"  file: {ac.file_reference or 'general'}\n"
        f"  line: {ac.line_reference if ac.line_reference is not None else 'N/A'}\n"
        f"  body: {ac.body}\n"
    )
    return f"## Diff\n```diff\n{diff}\n```\n\n{human_section}\n\n{agent_section}"


def parse_classification_reply(text: str) -> str:
    t = (text or "").strip()
    for line in t.splitlines():
        u = line.strip().upper()
        for lab in ("CONFIRMED", "PLAUSIBLE", "FABRICATED"):
            if re.search(rf"\b{lab}\b", u):
                return lab
    # Last resort: substring
    for lab in ("CONFIRMED", "PLAUSIBLE", "FABRICATED"):
        if lab in t.upper():
            return lab
    return "PLAUSIBLE"


def call_sonnet_classification(
    *,
    user_message: str,
    model: str = "claude-sonnet-4-6",
    max_tokens: int = 64,
) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=_SINGLE_COMMENT_SYSTEM,
        messages=[{"role": "user", "content": user_message}],
    )
    block = msg.content[0]
    text = getattr(block, "text", "") or ""
    return parse_classification_reply(text)


@dataclass
class CommentJudgment:
    task_id: str
    config_name: str
    stem: str
    comment_id: str
    baseline_label: str | None
    sonnet_label: str | None
    agree: bool | None


def simplified_proxy_score(labels: list[str]) -> float:
    """Illustrative partial score: 0.40 * precision_proxy - 0.25 * hallucination_proxy (per user's note)."""
    if not labels:
        return 0.0
    confirmed = sum(1 for x in labels if x == "CONFIRMED")
    fabricated = sum(1 for x in labels if x == "FABRICATED")
    total = len(labels)
    precision = confirmed / total
    hallucination = fabricated / total
    return 0.40 * precision - 0.25 * hallucination


def _short_config_name(config_name: str) -> str:
    full = normalize_config_name(config_name)
    if "config_A" in full or config_name in ("A", "config_A"):
        return "config_A"
    if "config_B" in full or config_name in ("B", "config_B"):
        return "config_B"
    if "config_C" in full or config_name in ("C", "config_C"):
        return "config_C"
    return config_name


def cohens_kappa(
    labels_a: list[str],
    labels_b: list[str],
) -> float | None:
    if len(labels_a) != len(labels_b) or not labels_a:
        return None
    cats = sorted(set(labels_a) | set(labels_b))
    n = len(labels_a)
    p_o = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    p_e = 0.0
    for c in cats:
        pa = sum(1 for x in labels_a if x == c) / n
        pb = sum(1 for x in labels_b if x == c) / n
        p_e += pa * pb
    if p_e >= 1.0:
        return 1.0 if p_o == 1.0 else 0.0
    return (p_o - p_e) / (1.0 - p_e)


def run_cross_judge(
    *,
    run_dir: Path,
    contexts_dir: Path,
    annotations_dir: Path,
    prs_path: Path,
    selected_task_ids: list[str] | None = None,
    seed: int = 42,
    sonnet_model: str = "claude-sonnet-4-6",
    dry_run: bool = False,
    resume_path: Path | None = None,
) -> dict[str, Any]:
    """
    Execute Sonnet per-comment judge for all (task_id, config) in sample × {A,B,C}.
    Returns a dict suitable for JSON export (summary + per-comment rows).
    """
    run_dir = run_dir.resolve()
    report_path = run_dir / "eval_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing eval_report.json: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    records = [r for r in (report.get("records") or []) if isinstance(r, dict)]

    if selected_task_ids is None:
        selected_task_ids = stratified_task_ids(records, seed=seed)

    configs_full = ["config_A_diff_only", "config_B_with_file_content", "config_C_full_context"]

    # Cache: resume (stem -> list of {comment_id, sonnet_label})
    resume_map: dict[str, dict[str, str]] = {}
    if resume_path and resume_path.is_file():
        prev = json.loads(resume_path.read_text(encoding="utf-8"))
        for row in prev.get("comment_rows") or []:
            if isinstance(row, dict):
                st = str(row.get("stem") or "")
                cid = str(row.get("comment_id") or "")
                sl = row.get("sonnet_label")
                if st and cid and isinstance(sl, str):
                    resume_map.setdefault(st, {})[cid] = sl

    comment_rows: list[dict[str, Any]] = []
    baseline_all: list[str] = []
    sonnet_all: list[str] = []

    for task_id in selected_task_ids:
        for cfg in configs_full:
            rec = next(
                (
                    r
                    for r in records
                    if r.get("task_id") == task_id
                    and normalize_config_name(str(r.get("config_name", ""))) == cfg
                ),
                None,
            )
            if rec is None:
                continue

            stem = stem_for_record(task_id, cfg)
            agent_path = run_dir / "agent_outputs" / f"{stem}_agent.json"
            judge_path = run_dir / "judge_outputs" / f"{stem}_judge.json"
            if not agent_path.is_file():
                continue
            baseline_by_id = load_baseline_judge_labels(judge_path) if judge_path.is_file() else {}

            eval_input = load_eval_input(
                task_id=task_id,
                config_name=cfg,
                contexts_dir=str(contexts_dir),
                annotations_dir=str(annotations_dir),
                prs_path=str(prs_path),
            )
            agent_out = load_agent_output_dict(agent_path)
            if not agent_out.parse_success or not agent_out.comments:
                continue

            cached = resume_map.get(stem, {})

            for ac in agent_out.comments:
                cid = ac.comment_id
                base = baseline_by_id.get(cid)
                if base is None:
                    base = "PLAUSIBLE"

                sonnet = cached.get(cid)
                if sonnet is None and not dry_run:
                    user_msg = build_single_comment_user_message(
                        diff_patch=eval_input.diff_patch,
                        human_comments=eval_input.human_comments,
                        agent_comment=ac,
                    )
                    sonnet = call_sonnet_classification(user_message=user_msg, model=sonnet_model)
                elif sonnet is None and dry_run:
                    sonnet = None

                row = {
                    "task_id": task_id,
                    "difficulty": rec.get("difficulty"),
                    "config_name": _short_config_name(cfg),
                    "stem": stem,
                    "comment_id": cid,
                    "baseline_label": base,
                    "sonnet_label": sonnet,
                    "agree": (base == sonnet) if sonnet is not None else None,
                }
                comment_rows.append(row)
                if sonnet is not None:
                    baseline_all.append(base)
                    sonnet_all.append(sonnet)

    # Per-config summary
    by_cfg: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in comment_rows:
        if row.get("sonnet_label") is None:
            continue
        cfg = str(row.get("config_name") or "")
        by_cfg[cfg].append((str(row["baseline_label"]), str(row["sonnet_label"])))

    kappa = cohens_kappa(baseline_all, sonnet_all) if baseline_all and sonnet_all else None
    agreement = (
        sum(1 for a, b in zip(baseline_all, sonnet_all) if a == b) / len(baseline_all)
        if baseline_all
        else None
    )

    config_summary: dict[str, Any] = {}
    for cfg, pairs in sorted(by_cfg.items()):
        ba = [a for a, _ in pairs]
        so = [b for _, b in pairs]
        config_summary[cfg] = {
            "n_comments": len(pairs),
            "mean_baseline_proxy": simplified_proxy_score(ba),
            "mean_sonnet_proxy": simplified_proxy_score(so),
            "agreement_rate": sum(1 for a, b in zip(ba, so) if a == b) / len(pairs) if pairs else None,
        }

    return {
        "run_dir": str(run_dir),
        "seed": seed,
        "selected_task_ids": selected_task_ids,
        "n_comments_judged": len(baseline_all),
        "overall_agreement": agreement,
        "cohens_kappa": kappa,
        "by_config": config_summary,
        "comment_rows": comment_rows,
        "sonnet_model": sonnet_model,
        "dry_run": dry_run,
    }
