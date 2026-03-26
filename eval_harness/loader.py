from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from eval_harness.io_utils import load_json, load_jsonl
from eval_harness.schema import EvalInput

CONFIG_SUFFIX_BY_FULL = {
    "config_A_diff_only": "config_A",
    "config_B_with_file_content": "config_B",
    "config_C_full_context": "config_C",
}


def normalize_config_name(config_name: str) -> str:
    alias = {
        "A": "config_A_diff_only",
        "B": "config_B_with_file_content",
        "C": "config_C_full_context",
        "config_A": "config_A_diff_only",
        "config_B": "config_B_with_file_content",
        "config_C": "config_C_full_context",
    }
    return alias.get(config_name, config_name)


def _candidate_context_paths(
    contexts_dir: str, task_id: str, pr_number: int, config_name: str
) -> list[Path]:
    """
    Resolve context JSON paths. Preferred layout (per config subfolder):

        contexts/config_A/{task_id}.json
        contexts/config_B/{task_id}.json
        contexts/config_C/{task_id}.json

    Legacy flat filenames under contexts/ are still accepted.
    """
    full_name = normalize_config_name(config_name)
    suffix = CONFIG_SUFFIX_BY_FULL.get(full_name, full_name)
    base = Path(contexts_dir)
    sub = base / suffix
    return [
        sub / f"{task_id}.json",
        sub / f"{task_id}_{suffix}.json",
        base / f"{task_id}_{suffix}.json",
        base / f"{task_id}_{full_name}.json",
        sub / f"{pr_number}_{suffix}.json",
        sub / f"{pr_number}_{full_name}.json",
        base / f"{pr_number}_{suffix}.json",
        base / f"{pr_number}_{full_name}.json",
    ]


def _candidate_annotation_paths(annotations_dir: str, task_id: str, pr_number: int) -> list[Path]:
    base = Path(annotations_dir)
    return [
        base / f"{task_id}_human.json",
        base / f"{pr_number}_human.json",
    ]


def load_pr_records(prs_path: str) -> list[dict[str, Any]]:
    p = Path(prs_path)
    if p.suffix.lower() == ".jsonl":
        return load_jsonl(p)
    data = load_json(p)
    if isinstance(data, dict):
        data = data.get("prs", data.get("items", []))
    return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []


def _fallback_task_id(pr: dict[str, Any]) -> str:
    if pr.get("task_id"):
        return str(pr.get("task_id"))
    repo = str(pr.get("repo") or "")
    pr_number = int(pr.get("pr_number") or pr.get("number") or 0)
    return f"{repo.split('/')[-1]}__{pr_number}"


def load_pr_record(prs_path: str, task_id: str | None = None, pr_number: int | None = None) -> dict[str, Any]:
    for pr in load_pr_records(prs_path):
        pr_task_id = _fallback_task_id(pr)
        pr_num = int(pr.get("pr_number") or pr.get("number") or 0)
        if task_id and pr_task_id == task_id:
            return pr
        if pr_number is not None and pr_num == int(pr_number):
            return pr
    raise KeyError(f"PR record not found for task_id={task_id!r}, pr_number={pr_number!r}")


def load_eval_input(
    task_id: str,
    config_name: str,
    contexts_dir: str,
    annotations_dir: str,
    prs_path: str,
) -> EvalInput:
    pr = load_pr_record(prs_path, task_id=task_id)
    pr_number = int(pr.get("pr_number") or pr.get("number") or 0)
    full_config = normalize_config_name(config_name)

    context_path = next(
        (p for p in _candidate_context_paths(contexts_dir, task_id, pr_number, full_config) if p.exists()),
        None,
    )
    if context_path is None:
        attempted = [str(p) for p in _candidate_context_paths(contexts_dir, task_id, pr_number, full_config)]
        raise FileNotFoundError(
            "Context file missing. Harness does not regenerate contexts.\n"
            f"Attempted:\n- " + "\n- ".join(attempted)
        )

    annotation_path = next(
        (p for p in _candidate_annotation_paths(annotations_dir, task_id, pr_number) if p.exists()),
        None,
    )
    if annotation_path is None:
        attempted = [str(p) for p in _candidate_annotation_paths(annotations_dir, task_id, pr_number)]
        raise FileNotFoundError("Annotation file missing.\nAttempted:\n- " + "\n- ".join(attempted))

    context = load_json(context_path)
    annotation = load_json(annotation_path)
    comments = annotation.get("comments", [])

    substantive_ids = set(annotation.get("substantive_comment_ids", []))
    if substantive_ids:
        ground_truth = [c for c in comments if c.get("comment_id") in substantive_ids]
    else:
        ground_truth = [c for c in comments if c.get("is_initiating_comment")]

    return EvalInput(
        task_id=task_id,
        pr_number=int(context.get("pr_number") or pr_number),
        repo=str(context.get("repo") or pr.get("repo") or ""),
        config_name=str(context.get("config_name") or full_config),
        rendered_context=str(context.get("rendered") or ""),
        total_tokens=int(context.get("total_tokens") or 0),
        pipeline_version=str(context.get("pipeline_version") or ""),
        difficulty=str(pr.get("difficulty") or ""),
        pr_type=str(pr.get("pr_type") or ""),
        language=str(pr.get("language") or ""),
        diff_patch=str(pr.get("diff_patch") or ""),
        human_comments=ground_truth,
        has_severity_annotations=any(c.get("severity") not in (None, "") for c in ground_truth),
    )


def discover_task_ids(prs_path: str, contexts_dir: str, configs: list[str]) -> list[str]:
    records = load_pr_records(prs_path)
    out: list[str] = []
    for pr in records:
        task_id = _fallback_task_id(pr)
        pr_number = int(pr.get("pr_number") or pr.get("number") or 0)
        valid = True
        for cfg in configs:
            if not any(
                p.exists()
                for p in _candidate_context_paths(contexts_dir, task_id, pr_number, normalize_config_name(cfg))
            ):
                valid = False
                break
        if valid:
            out.append(task_id)
    return out

