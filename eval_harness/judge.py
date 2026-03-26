from __future__ import annotations

import json
import re
from typing import Any

from eval_harness.logging_utils import get_logger
from eval_harness.model_clients import ModelRouter
from eval_harness.schema import (
    AgentOutput,
    EvalInput,
    HumanCommentStatus,
    JudgeClassification,
    JudgeOutput,
)

JUDGE_PROMPT_VERSION = "v1.0"

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator assessing AI code review quality against human expert review.

You are given:
1) PR diff
2) Human review comments (ground truth)  
3) AI agent review comments

## Step 1: Classify each agent comment

Use these definitions:

- CONFIRMED: The agent comment addresses the same underlying issue as a human
  comment, even if phrased differently, at a different abstraction level, or
  without citing the exact line number. Partial matches count if the core
  concern is the same. Set matched_human_comment_id to the matching human comment.

- PLAUSIBLE: A reasonable observation about the code that has no factual errors,
  but does not match any human comment. The agent may be correct but the human
  reviewers did not raise this point.

- FABRICATED: The comment contains factual errors about what the code does,
  references code that does not exist in the diff, or misidentifies the behavior
  of the code. Not being in ground truth is NOT sufficient for FABRICATED.

Examples of CONFIRMED (different wording, same issue):
  Human: "This needs to handle info < 0 per LAPACK docs"
  Agent:  "Error handling only checks positive return values but LAPACK
           can return negative codes for invalid arguments"
  → CONFIRMED

  Human: "free(NULL) is a no-op, remove the null check"
  Agent:  "The null check before free() is unnecessary"
  → CONFIRMED

## Step 2: Classify each human comment

Use your Step 1 results — do not classify independently:
- If a human comment H was matched by a CONFIRMED agent comment X,
  then H is CAUGHT with matched_agent_comment_id = X
- Otherwise H is MISSED

## Output

Return ONLY valid JSON:
{
  "agent_classifications": [
    {
      "comment_id": "...",
      "classification": "CONFIRMED|PLAUSIBLE|FABRICATED",
      "matched_human_comment_id": "..." or null,
      "actionability_score": 1-5,
      "reasoning": "one sentence"
    }
  ],
  "human_comment_statuses": [
    {
      "comment_id": "...",
      "status": "CAUGHT|MISSED",
      "matched_agent_comment_id": "..." or null
    }
  ]
}
"""


def run_judge(
    eval_input: EvalInput,
    agent_output: AgentOutput,
    judge_model: str,
    model_router: ModelRouter,
) -> JudgeOutput:
    user_message = _build_judge_message(eval_input, agent_output)
    try:
        raw = model_router.generate(
            model_id=judge_model,
            system=JUDGE_SYSTEM_PROMPT,
            user=user_message,
            max_tokens=4000,
            cache_system_prompt=True,
        )
        data = _parse_json_response(raw)
        agent_cls = _parse_agent_classifications(data.get("agent_classifications"))
        human_status = _parse_human_statuses(data.get("human_comment_statuses"))
        if not human_status:
            # Keep accounting stable even when judge output omits this section.
            human_status = _fallback_human_statuses(eval_input, parse_error=None)
        if not agent_cls and agent_output.comments:
            # Preserve count alignment: classify agent comments as PLAUSIBLE by default.
            agent_cls = _fallback_agent_classifications(agent_output, parse_error=None)
        return JudgeOutput(
            task_id=eval_input.task_id,
            config_name=eval_input.config_name,
            agent_classifications=agent_cls,
            human_comment_statuses=human_status,
            judge_model=judge_model,
            judge_prompt_version=JUDGE_PROMPT_VERSION,
        )
    except Exception as e:
        log = get_logger()
        log.warning(
            "judge_parse_failed_fallback_applied",
            task_id=eval_input.task_id,
            config=eval_input.config_name,
            judge_model=judge_model,
            error=str(e),
        )
        return JudgeOutput(
            task_id=eval_input.task_id,
            config_name=eval_input.config_name,
            agent_classifications=_fallback_agent_classifications(agent_output, parse_error=str(e)),
            human_comment_statuses=_fallback_human_statuses(eval_input, parse_error=str(e)),
            judge_model=judge_model,
            judge_prompt_version=f"{JUDGE_PROMPT_VERSION}+fallback",
        )


def _parse_json_response(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("Empty judge response.")
    try:
        data = json.loads(raw)
    except Exception:
        clean = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(clean)
        except Exception:
            # Last resort: parse the largest object-looking slice.
            start = clean.find("{")
            end = clean.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            data = json.loads(clean[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("Judge response must be a JSON object.")
    return data


def _parse_agent_classifications(raw: Any) -> list[JudgeClassification]:
    out: list[JudgeClassification] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        cls = str(item.get("classification") or "").strip().upper()
        if cls not in {"CONFIRMED", "PLAUSIBLE", "FABRICATED"}:
            continue
        score = item.get("actionability_score")
        try:
            score_i = int(score)
        except Exception:
            score_i = 3
        score_i = max(1, min(5, score_i))
        out.append(
            JudgeClassification(
                comment_id=str(item.get("comment_id") or ""),
                classification=cls,
                matched_human_comment_id=(
                    str(item.get("matched_human_comment_id"))
                    if item.get("matched_human_comment_id") not in (None, "")
                    else None
                ),
                actionability_score=score_i,
                reasoning=str(item.get("reasoning") or "") or None,
            )
        )
    return out


def _parse_human_statuses(raw: Any) -> list[HumanCommentStatus]:
    out: list[HumanCommentStatus] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        status = str(item.get("status") or "").strip().upper()
        if status not in {"CAUGHT", "MISSED"}:
            continue
        out.append(
            HumanCommentStatus(
                comment_id=str(item.get("comment_id") or ""),
                status=status,
                matched_agent_comment_id=(
                    str(item.get("matched_agent_comment_id"))
                    if item.get("matched_agent_comment_id") not in (None, "")
                    else None
                ),
            )
        )
    return out


def _build_judge_message(eval_input: EvalInput, agent_output: AgentOutput) -> str:
    diff_section = f"## Diff\n```diff\n{_build_structured_diff_snippet(eval_input.diff_patch, max_chars=12000)}\n```"
    human_section = "## Human Expert Review Comments (Ground Truth)\n"
    for c in eval_input.human_comments:
        human_section += (
            f"- comment_id: {c.get('comment_id')}\n"
            f"  file: {c.get('file', 'general')}\n"
            f"  line: {c.get('line', 'N/A')}\n"
            f"  severity: {c.get('severity', 'unspecified')}\n"
            f"  body: {c.get('body', '')}\n\n"
        )

    agent_section = "## AI Agent Review Comments (To Evaluate)\n"
    if not agent_output.comments:
        agent_section += "(Agent produced no comments)\n"
    for c in agent_output.comments:
        agent_section += (
            f"- comment_id: {c.comment_id}\n"
            f"  file: {c.file_reference or 'general'}\n"
            f"  line: {c.line_reference if c.line_reference is not None else 'N/A'}\n"
            f"  severity_claimed: {c.severity_claim or 'unspecified'}\n"
            f"  body: {c.body}\n\n"
        )
    return f"{diff_section}\n\n{human_section}\n\n{agent_section}"


def _build_structured_diff_snippet(diff_patch: str, max_chars: int = 12000) -> str:
    chunks = _parse_diff_chunks(diff_patch)
    # Prefer implementation files and logic-rich hunks.
    chunks.sort(
        key=lambda c: (
            c.get("is_test", False),
            -(int(c.get("lines_changed") or 0) + int(c.get("logic_markers") or 0) * 3),
        ),
    )
    out_parts: list[str] = []
    used = 0
    for c in chunks:
        raw = str(c.get("raw") or "")
        if not raw:
            continue
        projected = used + len(raw) + (2 if out_parts else 0)
        if projected > max_chars:
            continue
        out_parts.append(raw)
        used = projected
    if not out_parts:
        return (diff_patch or "")[:max_chars]
    return "\n\n".join(out_parts)


def _parse_diff_chunks(diff_patch: str) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    current_file = ""
    current_lines: list[str] = []
    for line in (diff_patch or "").splitlines():
        if line.startswith("diff --git "):
            if current_lines:
                raw = "\n".join(current_lines)
                chunks.append(_build_chunk(current_file, raw))
            current_file = _extract_file_path(line)
            current_lines = [line]
            continue
        if current_lines:
            current_lines.append(line)
    if current_lines:
        raw = "\n".join(current_lines)
        chunks.append(_build_chunk(current_file, raw))
    return chunks


def _extract_file_path(diff_header: str) -> str:
    m = re.search(r" b/(.+)$", diff_header)
    return str(m.group(1)) if m else ""


def _build_chunk(file_path: str, raw: str) -> dict[str, Any]:
    lines_changed = sum(1 for ln in raw.splitlines() if (ln.startswith("+") or ln.startswith("-")) and not ln.startswith("+++") and not ln.startswith("---"))
    lowered = raw.lower()
    logic_markers = sum(1 for marker in ("if ", "switch ", "case ", "return ", "error", "panic", "parse", "timeout", "duration") if marker in lowered)
    is_test = _looks_like_test_file(file_path)
    return {
        "file_path": file_path,
        "raw": raw,
        "lines_changed": lines_changed,
        "logic_markers": logic_markers,
        "is_test": is_test,
    }


def _looks_like_test_file(path: str) -> bool:
    p = str(path or "").lower()
    return any(x in p for x in ("test/", "tests/", "_test.", ".test.", ".spec.", "__tests__/"))


def _fallback_agent_classifications(
    agent_output: AgentOutput, parse_error: str | None
) -> list[JudgeClassification]:
    out: list[JudgeClassification] = []
    for c in agent_output.comments:
        out.append(
            JudgeClassification(
                comment_id=c.comment_id,
                classification="PLAUSIBLE",
                matched_human_comment_id=None,
                actionability_score=3,
                reasoning=("fallback_due_to_parse_error: " + parse_error) if parse_error else "fallback_default",
            )
        )
    return out


def _fallback_human_statuses(
    eval_input: EvalInput, parse_error: str | None
) -> list[HumanCommentStatus]:
    _ = parse_error
    return [
        HumanCommentStatus(
            comment_id=str(c.get("comment_id") or ""),
            status="MISSED",
            matched_agent_comment_id=None,
        )
        for c in eval_input.human_comments
    ]

