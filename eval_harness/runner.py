from __future__ import annotations

import json
import re
import time
from typing import Any

from eval_harness.model_clients import ModelRouter
from eval_harness.schema import AgentComment, AgentOutput, EvalInput

AGENT_SYSTEM_PROMPT = """
You are an expert code reviewer. Review the pull request and identify real issues.

## Rules
1. Every issue must trace to a specific line in the diff or provided code.
2. Do not invent behavior not visible in the code.
3. Do not flag issues in files you have not been shown.
4. Do not repeat the same concern in different words.

## Severity
- P0: production bug, data loss, or security issue
- P1: clear issue that should be fixed before merge
- P2: credible risk worth raising

## What to look for
Correctness bugs, missing error handling, edge cases, integration breakage, unhandled failures, performance issues in hot paths, broken contracts with callers.

## Output
Return ONLY a valid JSON array, no markdown, no prose.
Aim for 4-6 distinct issues. Return [] if nothing real to flag.

[{"body": "...", "file": "path or null", "line": 42 or null, "severity": "P0|P1|P2"}]
"""

AGENT_RETRY_SUFFIX = """
Return ONLY a valid JSON array. No markdown, no prose.
[{"body": "...", "file": "path or null", "line": 42 or null, "severity": "P0|P1|P2"}]
Aim for 4-6 issues. Return [] if none.
"""




MAX_AGENT_COMMENTS = 30


def run_agent(
    eval_input: EvalInput,
    model: str,
    model_router: ModelRouter,
    max_tokens: int = 4000,
) -> AgentOutput:
    raw_response = ""
    try:
        raw_response = _generate_with_retries(
            model_router=model_router,
            model_id=model,
            system=AGENT_SYSTEM_PROMPT,
            user=eval_input.rendered_context,
            max_tokens=max(256, int(max_tokens)),
            attempts=3,
        )
        try:
            return build_agent_output_from_raw(eval_input, model, raw_response)
        except Exception as first_parse_error:
            retry_raw = _generate_with_retries(
                model_router=model_router,
                model_id=model,
                system=AGENT_SYSTEM_PROMPT,
                user=eval_input.rendered_context + AGENT_RETRY_SUFFIX,
                max_tokens=max(1024, min(int(max_tokens), 16384)),
                attempts=2,
            )
            try:
                out = build_agent_output_from_raw(eval_input, model, retry_raw)
                out.raw_response = retry_raw
                return out
            except Exception as second_parse_error:
                combined_error = (
                    f"initial_parse_error={first_parse_error}; "
                    f"retry_parse_error={second_parse_error}"
                )
                return AgentOutput(
                    task_id=eval_input.task_id,
                    config_name=eval_input.config_name,
                    model=model,
                    raw_response=retry_raw or raw_response,
                    comments=[],
                    parse_success=False,
                    parse_error=combined_error,
                )
    except Exception as e:
        return AgentOutput(
            task_id=eval_input.task_id,
            config_name=eval_input.config_name,
            model=model,
            raw_response=raw_response,
            comments=[],
            parse_success=False,
            parse_error=str(e),
        )


def build_agent_output_from_raw(eval_input: EvalInput, model: str, raw_response: str) -> AgentOutput:
    parsed = _parse_agent_json(raw_response)
    comments = _parse_agent_comments(parsed, eval_input.task_id, eval_input.config_name, eval_input.diff_patch)
   
    if not comments and _looks_like_structured_comment_payload(raw_response):
        raise ValueError("Agent response looked structured but no parseable comments were extracted.")
    return AgentOutput(
        task_id=eval_input.task_id,
        config_name=eval_input.config_name,
        model=model,
        raw_response=raw_response,
        comments=comments,
        parse_success=True,
        parse_error=None,
    )


def _parse_agent_json(raw: str) -> Any:
    text = (raw or "").strip()
    if not text:
        return []

    try:
        return json.loads(text)
    except Exception:
        pass

    # Common case: fenced json block.
    no_fence = re.sub(r"```json|```", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(no_fence)
    except Exception:
        pass
    decoder = json.JSONDecoder()
    candidates: list[Any] = []
    for i, ch in enumerate(no_fence):
        if ch not in "[{":
            continue
        try:
            parsed, _end = decoder.raw_decode(no_fence[i:])
        except Exception:
            continue
        candidates.append(parsed)

    for parsed in reversed(candidates):
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict) and isinstance(parsed.get("comments"), list):
            return parsed["comments"]

    fragment_comments = _extract_comment_object_fragments(no_fence)
    if fragment_comments:
        return fragment_comments

    raise json.JSONDecodeError("Unable to parse agent JSON response", no_fence, 0)


def _normalize_severity(raw: Any) -> str | None:
    if not isinstance(raw, str):
        return None
    s = raw.strip().upper()
    return s if s in {"P0", "P1", "P2"} else None


def _parse_line(raw: Any) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _parse_agent_comments(
    data: Any, task_id: str, config_name: str, diff_patch: str
) -> list[AgentComment]:
    if isinstance(data, dict) and isinstance(data.get("comments"), list):
        data = data.get("comments")
    if not isinstance(data, list):
        return []
    diff_lines = _extract_diff_line_numbers(diff_patch)
    out: list[AgentComment] = []
    for idx, item in enumerate(data):
        if len(out) >= MAX_AGENT_COMMENTS:
            break
        if not isinstance(item, dict):
            continue
        body = str(item.get("body") or "").strip()
        if not body:
            continue
        line_ref = _parse_line(item.get("line"))
        is_outside_diff = line_ref is not None and line_ref not in diff_lines
        out.append(
            AgentComment(
                comment_id=f"{task_id}_{config_name}_{idx}",
                body=body,
                file_reference=str(item.get("file")) if item.get("file") not in (None, "") else None,
                line_reference=line_ref,
                severity_claim=_normalize_severity(item.get("severity")),
                is_outside_diff=is_outside_diff,
            )
        )
    return out


def _extract_diff_line_numbers(diff_patch: str) -> set[int]:
    lines: set[int] = set()
    current_new_line: int | None = None
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for ln in (diff_patch or "").splitlines():
        m = hunk_re.match(ln)
        if m:
            current_new_line = int(m.group(1))
            continue
        if current_new_line is None:
            continue
        if ln.startswith("+") and not ln.startswith("+++"):
            lines.add(current_new_line)
            current_new_line += 1
            continue
        if ln.startswith("-") and not ln.startswith("---"):
            continue
        current_new_line += 1
    return lines


def _extract_comment_object_fragments(text: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    out: list[dict[str, Any]] = []
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("body"), str):
            out.append(parsed)
    return out


def _looks_like_structured_comment_payload(raw: str) -> bool:
    txt = (raw or "").strip()
    if not txt:
        return False
    if re.search(r'"body"\s*:', txt):
        return True
    if "```json" in txt.lower():
        return True
    return False


def _generate_with_retries(
    model_router: ModelRouter,
    model_id: str,
    system: str,
    user: str,
    max_tokens: int,
    attempts: int,
) -> str:
    last_error: Exception | None = None
    for i in range(max(1, int(attempts))):
        try:
            raw = model_router.generate(
                model_id=model_id,
                system=system,
                user=user,
                max_tokens=max_tokens,
            )
            if str(raw or "").strip() == "":
                raise ValueError("Empty model response")
            return raw
        except Exception as e:
            last_error = e
            if i >= attempts - 1:
                break
            time.sleep(min(2 ** i, 8))
    raise RuntimeError(f"Agent generation failed after {attempts} attempts: {last_error}")

