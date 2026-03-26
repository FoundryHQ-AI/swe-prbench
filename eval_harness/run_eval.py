#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=REPO_ROOT / ".env")
except Exception:
    env_path = REPO_ROOT / ".env"
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value

from eval_harness.logging_utils import get_logger, setup_logging
from eval_harness.aggregate import generate_eval_report
from eval_harness.assembler import assemble_eval_result
from eval_harness.io_utils import load_json, write_json
from eval_harness.judge import run_judge
from eval_harness.loader import discover_task_ids, load_eval_input, normalize_config_name
from eval_harness.model_clients import ModelEndpoint, ModelRouter
from eval_harness.runner import AGENT_SYSTEM_PROMPT, build_agent_output_from_raw, run_agent
from eval_harness.schema import EvalResult, HumanCommentStatus, JudgeOutput
from eval_harness.validate_output import validate_eval_result


def _parse_configs(values: list[str]) -> list[str]:
    if not values:
        return [
            "config_A_diff_only",
            "config_B_with_file_content",
            "config_C_full_context",
        ]
    return [normalize_config_name(v) for v in values]


def _parse_configs_from_defaults(raw: object) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    for v in raw:
        if isinstance(v, str) and v.strip():
            out.append(v.strip())
    return out


def _slugify(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_")


def _normalize_contexts_root(contexts_dir: Path) -> Path:
    """
    Treat either <dataset>/contexts or <dataset>/contexts/config_{A,B,C} as the same root.
    Loaders expect the parent `contexts/` directory that contains config_A|B|C subfolders.
    """
    if contexts_dir.name in ("config_A", "config_B", "config_C"):
        return contexts_dir.parent
    return contexts_dir


def _infer_output_root(
    explicit_output: str | None, contexts_dir: Path, annotations_dir: Path, prs_path: str
) -> Path:
    if explicit_output:
        return Path(explicit_output)

    if contexts_dir.name == "contexts" and annotations_dir.name == "annotations":
        c_parent = contexts_dir.parent.resolve()
        a_parent = annotations_dir.parent.resolve()
        prs_parent = Path(prs_path).resolve().parent
        if c_parent == a_parent == prs_parent:
            return c_parent / "evals"

    # Fallback: colocate next to contexts.
    return contexts_dir.resolve().parent / "evals"


def _infer_model_config(explicit_path: str | None) -> str | None:
    if explicit_path:
        return explicit_path
    default_cfg = REPO_ROOT / "eval_harness" / "model_endpoints.example.yaml"
    if default_cfg.exists():
        return str(default_cfg)
    return None


def _expand_model_ids(raw_values: list[str] | None, fallback: str, model_router: ModelRouter) -> list[str]:
    vals = [v.strip() for v in (raw_values or []) if isinstance(v, str) and v.strip()]
    if not vals:
        vals = [fallback]
    out: list[str] = []
    for v in vals:
        if v.lower() == "all":
            out.extend(model_router.models.keys())
        else:
            out.append(v)
    # keep deterministic unique order
    dedup: list[str] = []
    seen = set()
    for v in out:
        if v in seen:
            continue
        dedup.append(v)
        seen.add(v)
    return dedup


def _build_model_pairs(
    agent_models: list[str], judge_models: list[str], pair_mode: str
) -> list[tuple[str, str]]:
    if pair_mode == "cross":
        return [(a, j) for a in agent_models for j in judge_models]

    # aligned mode
    if len(agent_models) == len(judge_models):
        return list(zip(agent_models, judge_models))
    if len(agent_models) == 1:
        return [(agent_models[0], j) for j in judge_models]
    if len(judge_models) == 1:
        return [(a, judge_models[0]) for a in agent_models]
    raise ValueError(
        "For --pair-mode aligned, --agent-models and --judge-models must have the same length "
        "or one side must have length 1 for broadcasting."
    )


def _ensure_model_ids_exist(model_ids: Iterable[str], model_router: ModelRouter, flag_name: str) -> None:
    missing = [m for m in model_ids if not model_router.has_model(m)]
    if missing:
        available = ", ".join(sorted(model_router.models.keys()))
        raise KeyError(
            f"{flag_name} contains unknown model IDs: {missing}. "
            f"Available IDs in model config: [{available}]"
        )


def _has_numpy(python_exe: str) -> bool:
    try:
        proc = subprocess.run(
            [python_exe, "-c", "import numpy"],
            check=False,
            capture_output=True,
            text=True,
        )
        return proc.returncode == 0
    except Exception:
        return False


def _find_alternate_python_with_numpy(current_exe: str) -> str | None:
    candidates = [
        str(REPO_ROOT / ".venv" / "bin" / "python"),
        str(REPO_ROOT / "venv" / "bin" / "python"),
        str(REPO_ROOT / ".venv" / "bin" / "python3"),
        str(REPO_ROOT / "venv" / "bin" / "python3"),
    ]
    for candidate in candidates:
        if candidate == current_exe:
            continue
        if not Path(candidate).exists():
            continue
        if _has_numpy(candidate):
            return candidate
    return None


def _extract_text_from_anthropic_content(content: object) -> str:
    if not isinstance(content, list):
        return ""
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            t = block.get("text")
            if isinstance(t, str):
                text_parts.append(t)
            continue
        t = getattr(block, "text", None)
        if isinstance(t, str):
            text_parts.append(t)
    return "\n".join(text_parts).strip()


def _extract_attr_or_key(obj: object, name: str) -> object:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _run_agent_batch_anthropic(
    model_router: ModelRouter,
    agent_model_id: str,
    record_items: list[tuple[str, str, object, str]],
    poll_seconds: int,
    log,
) -> dict[str, object]:
    endpoint = model_router.resolve_endpoint(agent_model_id)
    if endpoint.provider != "anthropic":
        raise RuntimeError("Anthropic Batch API can only be used for anthropic provider models.")

    try:
        from anthropic import Anthropic
    except Exception as e:
        raise RuntimeError(
            "anthropic SDK is required for --use-anthropic-batch-agent. "
            "Install in active venv: python -m pip install anthropic"
        ) from e

    kwargs: dict[str, object] = {"api_key": endpoint.api_key}
    if endpoint.base_url:
        kwargs["base_url"] = endpoint.base_url
    client = Anthropic(**kwargs)

    requests_payload = []
    for task_id, cfg, eval_input, stem in record_items:
        _ = task_id, cfg
        requests_payload.append(
            {
                "custom_id": stem,
                "params": {
                    "model": endpoint.model,
                    "max_tokens": 2000,
                    "system": AGENT_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": eval_input.rendered_context}],
                },
            }
        )

    batch = client.messages.batches.create(requests=requests_payload)
    batch_id = _extract_attr_or_key(batch, "id")
    if not isinstance(batch_id, str) or not batch_id:
        raise RuntimeError("Failed to create Anthropic batch: missing batch id.")
    log.info("agent_batch_submitted", batch_id=batch_id, requests=len(requests_payload))

    poll_interval = max(5, int(poll_seconds))
    poll_count = 0
    while True:
        batch_obj = client.messages.batches.retrieve(batch_id)
        status = str(
            _extract_attr_or_key(batch_obj, "processing_status")
            or _extract_attr_or_key(batch_obj, "status")
            or ""
        ).lower()
        if status in {"ended", "completed", "succeeded"}:
            break
        if status in {"canceled", "cancelled", "errored", "failed", "expired"}:
            raise RuntimeError(f"Anthropic batch failed with status={status}")
        poll_count += 1
        log.info(
            "anthropic_batch_polling",
            batch_id=batch_id,
            status=status or "pending",
            poll_count=poll_count,
            wait_sec=poll_count * poll_interval,
        )
        time.sleep(poll_interval)

    # Parse batch results and map custom_id -> AgentOutput.
    out: dict[str, object] = {}
    result_obj = client.messages.batches.results(batch_id)
    if isinstance(result_obj, list):
        result_iter = result_obj
    else:
        data_attr = _extract_attr_or_key(result_obj, "data")
        if isinstance(data_attr, list):
            result_iter = data_attr
        else:
            try:
                result_iter = list(result_obj)
            except Exception:
                result_iter = []

    by_stem = {stem: (task_id, cfg, eval_input) for task_id, cfg, eval_input, stem in record_items}
    for item in result_iter:
        custom_id = _extract_attr_or_key(item, "custom_id")
        if not isinstance(custom_id, str) or custom_id not in by_stem:
            continue
        _, _, eval_input = by_stem[custom_id]

        result = _extract_attr_or_key(item, "result")
        result_type = str(_extract_attr_or_key(result, "type") or "").lower()
        if result_type and result_type != "succeeded":
            err_msg = str(_extract_attr_or_key(result, "error") or f"batch_result_type={result_type}")
            out[custom_id] = build_agent_output_from_raw(eval_input, agent_model_id, "[]")
            out[custom_id].parse_success = False
            out[custom_id].parse_error = err_msg
            continue

        message = _extract_attr_or_key(result, "message")
        content = _extract_attr_or_key(message, "content")
        raw_text = _extract_text_from_anthropic_content(content)
        try:
            out[custom_id] = build_agent_output_from_raw(eval_input, agent_model_id, raw_text)
        except Exception as e:
            out[custom_id] = build_agent_output_from_raw(eval_input, agent_model_id, "[]")
            out[custom_id].parse_success = False
            out[custom_id].parse_error = str(e)
            out[custom_id].raw_response = raw_text
    log.info("agent_batch_completed", batch_id=batch_id, completed=len(out), total=len(record_items))
    return out


def _run_agent_batch_openai(
    model_router: ModelRouter,
    agent_model_id: str,
    record_items: list[tuple[str, str, object, str]],
    poll_seconds: int,
    log,
) -> dict[str, object]:
    """Use OpenAI Batch API for agent calls (50% cost discount, higher rate limits)."""
    endpoint = model_router.resolve_endpoint(agent_model_id)
    if endpoint.provider not in ("openai", "openai_compatible"):
        raise RuntimeError("OpenAI Batch API can only be used for openai provider models.")

    try:
        from openai import OpenAI
    except Exception as e:
        raise RuntimeError(
            "openai SDK is required for --use-openai-batch-agent. "
            "Install: python -m pip install openai"
        ) from e

    client = OpenAI(api_key=endpoint.api_key, base_url=endpoint.base_url or None)

    # Build JSONL input (OpenAI batch format)
    lines = []
    for task_id, cfg, eval_input, stem in record_items:
        _ = task_id, cfg
        lines.append(
            json.dumps(
                {
                    "custom_id": stem,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": endpoint.model,
                        "messages": [
                            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
                            {"role": "user", "content": eval_input.rendered_context},
                        ],
                        "max_completion_tokens": 2000,
                    },
                }
            )
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        f.write("\n".join(lines))
        input_path = f.name

    try:
        with open(input_path, "rb") as f:
            batch_file = client.files.create(file=f, purpose="batch")
        input_file_id = batch_file.id

        batch = client.batches.create(
            input_file_id=input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_id = batch.id
        log.info("openai_batch_submitted", batch_id=batch_id, requests=len(record_items))

        poll_interval = max(5, int(poll_seconds))
        poll_count = 0
        while True:
            batch_obj = client.batches.retrieve(batch_id)
            status = str(getattr(batch_obj, "status", "") or "").lower()
            if status == "completed":
                break
            if status in ("failed", "expired", "cancelled", "cancelling"):
                raise RuntimeError(f"OpenAI batch failed with status={status}")
            poll_count += 1
            log.info(
                "openai_batch_polling",
                batch_id=batch_id,
                status=status or "pending",
                poll_count=poll_count,
                wait_sec=poll_count * poll_interval,
            )
            time.sleep(poll_interval)

        output_file_id = getattr(batch_obj, "output_file_id", None)
        if not output_file_id:
            raise RuntimeError("OpenAI batch completed but no output_file_id")

        content = client.files.content(output_file_id)
        if hasattr(content, "read"):
            output_text = content.read().decode("utf-8")
        elif hasattr(content, "text"):
            output_text = content.text
        else:
            output_text = str(content)
    finally:
        Path(input_path).unlink(missing_ok=True)

    by_stem = {stem: (task_id, cfg, eval_input) for task_id, cfg, eval_input, stem in record_items}
    out: dict[str, object] = {}
    for line in output_text.strip().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        custom_id = row.get("custom_id")
        if not isinstance(custom_id, str) or custom_id not in by_stem:
            continue
        _, _, eval_input = by_stem[custom_id]

        err = row.get("error")
        if err:
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            out[custom_id] = build_agent_output_from_raw(eval_input, agent_model_id, "[]")
            out[custom_id].parse_success = False
            out[custom_id].parse_error = err_msg
            continue

        resp = row.get("response") or {}
        body = resp.get("body") if isinstance(resp, dict) else None
        if not isinstance(body, dict):
            out[custom_id] = build_agent_output_from_raw(eval_input, agent_model_id, "[]")
            out[custom_id].parse_success = False
            out[custom_id].parse_error = "batch_response_missing_body"
            continue

        choices = body.get("choices") or []
        msg = choices[0].get("message", {}) if choices else {}
        raw_text = msg.get("content") or "[]"
        try:
            out[custom_id] = build_agent_output_from_raw(eval_input, agent_model_id, raw_text)
        except Exception as e:
            out[custom_id] = build_agent_output_from_raw(eval_input, agent_model_id, "[]")
            out[custom_id].parse_success = False
            out[custom_id].parse_error = str(e)
            out[custom_id].raw_response = raw_text

    log.info("openai_batch_completed", batch_id=batch_id, completed=len(out), total=len(record_items))
    return out


def _load_all_eval_results(run_root: Path) -> list[EvalResult]:
    out: list[EvalResult] = []
    eval_dir = run_root / "eval_results"
    if not eval_dir.exists():
        return out
    for p in sorted(eval_dir.glob("*_eval.json")):
        try:
            row = load_json(p)
        except Exception:
            continue
        if not isinstance(row, dict):
            continue
        try:
            out.append(EvalResult(**row))
        except Exception:
            continue
    return out


def main() -> None:
    try:
        from eval_harness.scorer import compute_dimension_scores
    except ModuleNotFoundError as e:
        if str(e).endswith("No module named 'numpy'"):
            alt_python = _find_alternate_python_with_numpy(sys.executable)
            if alt_python and os.environ.get("SWE_PRBENCH_REEXEC") != "1":
                os.environ["SWE_PRBENCH_REEXEC"] = "1"
                os.execv(alt_python, [alt_python, *sys.argv])
            raise RuntimeError(
                "NumPy is missing in the current interpreter.\n"
                f"Python executable: {sys.executable}\n"
                "Install it in this interpreter:\n"
                f"  {sys.executable} -m pip install numpy\n"
                "Or run with an interpreter that already has NumPy, e.g.:\n"
                f"  {REPO_ROOT}/venv/bin/python eval_harness/run_eval.py ...\n"
                f"  {REPO_ROOT}/.venv/bin/python eval_harness/run_eval.py ..."
            ) from e
        raise

    parser = argparse.ArgumentParser(
        description="SWE-PRBench evaluation harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Outputs are written under <output>/<agent>__judge_<judge>/ (e.g. results/runs/) "
            "with eval_report.json — use that same directory as compile_leaderboard_report.py --evals-dir."
        ),
    )
    g_data = parser.add_argument_group("Dataset paths")
    g_data.add_argument(
        "--contexts",
        required=True,
        help="Contexts root (directory containing config_A/, config_B/, config_C/) or a single config_* subfolder.",
    )
    g_data.add_argument("--annotations", required=True, help="Annotations directory (*_human.json)")
    g_data.add_argument("--prs", required=True, help="prs.jsonl or JSON array of PR records")
    g_data.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output root for runs (default: <dataset>/evals). "
            "Recommended: results/runs — each model pair gets a subdirectory with eval_report.json."
        ),
    )

    g_models = parser.add_argument_group("Models (see model_endpoints YAML)")
    g_models.add_argument(
        "--model",
        default=None,
        help="Agent model id (optional if --agent-models is set).",
    )
    g_models.add_argument("--judge-model", default=None, help="Judge model id (optional if fixed in config)")
    g_models.add_argument(
        "--agent-models",
        nargs="*",
        default=None,
        help="Agent model ids; use 'all' for every id in model config.",
    )
    g_models.add_argument(
        "--judge-models",
        nargs="*",
        default=None,
        help="Judge model ids (only when judge is not fixed in config).",
    )
    g_models.add_argument(
        "--pair-mode",
        choices=["aligned", "cross"],
        default="aligned",
        help="How to combine agent and judge lists (default: aligned).",
    )
    g_models.add_argument(
        "--model-config",
        default=None,
        help="YAML with models.* and defaults. Required for multi-provider evals.",
    )

    g_scope = parser.add_argument_group("Evaluation scope")
    g_scope.add_argument("--configs", nargs="*", default=None, help="A B C (default from config or A/B/C)")
    g_scope.add_argument("--task-ids", nargs="*", default=None, help="Subset of task_ids")
    g_scope.add_argument("--max-prs", type=int, default=0, help="Cap number of PRs (smoke tests)")

    g_perf = parser.add_argument_group("Performance")
    g_perf.add_argument(
        "--agent-max-tokens",
        type=int,
        default=4000,
        help="Max output tokens for agent (default: 4000).",
    )
    g_perf.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent eval records per model pair (default: 1).",
    )

    g_batch = parser.add_argument_group("Optional batch APIs (lower cost, slower turnaround)")
    g_batch.add_argument(
        "--use-anthropic-batch-agent",
        action="store_true",
        help="Anthropic Batch API for agent calls (anthropic agent models only).",
    )
    g_batch.add_argument(
        "--use-openai-batch-agent",
        action="store_true",
        help="OpenAI Batch API for agent calls (openai-compatible agent models only).",
    )
    g_batch.add_argument(
        "--batch-poll-seconds",
        type=int,
        default=30,
        help="Poll interval for Anthropic batch jobs (default: 30).",
    )

    parser.add_argument("-q", "--quiet", action="store_true", help="Less verbose logs")
    args = parser.parse_args()

    if (not args.model or not str(args.model).strip()) and not args.agent_models:
        parser.error("one of --model or --agent-models is required")

    setup_logging(verbose=not args.quiet, level="INFO")
    log = get_logger()

    contexts_dir = _normalize_contexts_root(Path(args.contexts))
    annotations_dir = Path(args.annotations)
    output_root = _infer_output_root(args.output, contexts_dir, annotations_dir, args.prs)
    if not contexts_dir.exists():
        raise FileNotFoundError(f"contexts dir not found: {contexts_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"annotations dir not found: {annotations_dir}")
    if not Path(args.prs).exists():
        raise FileNotFoundError(f"prs file not found: {args.prs}")

    model_config_path = _infer_model_config(args.model_config)

    if model_config_path:
        model_router = ModelRouter.from_config_file(model_config_path)
        default_cfgs = _parse_configs_from_defaults(model_router.defaults.get("configs"))
        configs = _parse_configs(args.configs or default_cfgs or ["A", "B", "C"])

        fixed_judge = model_router.defaults.get("judge_model")
        if isinstance(fixed_judge, str) and fixed_judge.strip():
            fixed_judge = fixed_judge.strip()
            if args.judge_model and args.judge_model != fixed_judge:
                raise ValueError(
                    f"Judge model is fixed by config defaults.judge_model={fixed_judge!r}; "
                    f"received --judge-model={args.judge_model!r}."
                )
            if args.judge_models:
                provided = [m for m in args.judge_models if isinstance(m, str) and m.strip()]
                if provided and any(m != fixed_judge for m in provided):
                    raise ValueError(
                        f"Judge model is fixed by config defaults.judge_model={fixed_judge!r}; "
                        f"received --judge-models={provided!r}."
                    )
            judge_seed = fixed_judge
            judge_sources = [fixed_judge]
        else:
            judge_seed = args.judge_model or args.model
            judge_sources = args.judge_models

        task_ids = list(args.task_ids) if args.task_ids else discover_task_ids(
            args.prs, str(contexts_dir), configs
        )
        if args.max_prs and args.max_prs > 0:
            task_ids = task_ids[: args.max_prs]
        if not task_ids:
            raise RuntimeError("No tasks discovered for requested configs. Check contexts/prs paths.")

        model_seed = str(args.model).strip() if args.model else ""
        if not model_seed and args.agent_models:
            model_seed = str(args.agent_models[0]).strip()
        if not model_seed:
            raise ValueError("Unable to determine model seed; provide --model or --agent-models.")

        agent_model_ids = _expand_model_ids(args.agent_models, model_seed, model_router)
        judge_model_ids = _expand_model_ids(judge_sources, judge_seed, model_router)
        _ensure_model_ids_exist(agent_model_ids, model_router, "--agent-models/--model")
        _ensure_model_ids_exist(judge_model_ids, model_router, "--judge-models/--judge-model")
        model_pairs = _build_model_pairs(agent_model_ids, judge_model_ids, args.pair_mode)
    else:
        if not args.model:
            raise ValueError("--model is required when --model-config is not provided.")
        configs = _parse_configs(args.configs or ["A", "B", "C"])
        task_ids = list(args.task_ids) if args.task_ids else discover_task_ids(
            args.prs, str(contexts_dir), configs
        )
        if args.max_prs and args.max_prs > 0:
            task_ids = task_ids[: args.max_prs]
        if not task_ids:
            raise RuntimeError("No tasks discovered for requested configs. Check contexts/prs paths.")

        # Backward-compatible fallback: both model refs are treated as Anthropic model names.
        api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is required when --model-config is not provided."
            )
        model_router = ModelRouter(
            models={
                args.model: ModelEndpoint(provider="anthropic", model=args.model, api_key=api_key),
                (args.judge_model or args.model): ModelEndpoint(
                    provider="anthropic", model=(args.judge_model or args.model), api_key=api_key
                ),
            }
        )
        model_pairs = [(args.model, (args.judge_model or args.model))]

    sweep_summary: list[dict] = []
    for agent_model_id, judge_model_id in model_pairs:
        # Always write runs into model-tagged subdirectories for easier comparison.
        run_tag = f"{_slugify(agent_model_id)}__judge_{_slugify(judge_model_id)}"
        run_root = output_root / run_tag
        (run_root / "agent_outputs").mkdir(parents=True, exist_ok=True)
        (run_root / "judge_outputs").mkdir(parents=True, exist_ok=True)
        (run_root / "eval_results").mkdir(parents=True, exist_ok=True)

        results = []
        validation_failures = []
        total = len(task_ids) * len(configs)
        log.info(
            "eval_model_pair_start",
            agent_model=agent_model_id,
            judge_model=judge_model_id,
            prs=len(task_ids),
            configs=len(configs),
            out=str(run_root),
            concurrency=max(1, int(args.concurrency)),
            use_batch=bool(args.use_anthropic_batch_agent or args.use_openai_batch_agent),
        )

        # Build eval inputs up front.
        record_items: list[tuple[str, str, object, str]] = []
        for task_id in task_ids:
            for cfg in configs:
                eval_input = load_eval_input(
                    task_id=task_id,
                    config_name=cfg,
                    contexts_dir=str(contexts_dir),
                    annotations_dir=str(annotations_dir),
                    prs_path=args.prs,
                )
                stem = f"{task_id}_{cfg}"
                record_items.append((task_id, cfg, eval_input, stem))

        agent_outputs_by_stem: dict[str, object] = {}
        endpoint = model_router.resolve_endpoint(agent_model_id)
        if args.use_anthropic_batch_agent and endpoint.provider == "anthropic":
            try:
                agent_outputs_by_stem = _run_agent_batch_anthropic(
                    model_router=model_router,
                    agent_model_id=agent_model_id,
                    record_items=record_items,
                    poll_seconds=max(5, int(args.batch_poll_seconds)),
                    log=log,
                )
            except Exception as e:
                log.warning(
                    "agent_batch_failed_fallback_to_realtime",
                    agent_model=agent_model_id,
                    error=str(e),
                )
                agent_outputs_by_stem = {}
        elif args.use_openai_batch_agent and endpoint.provider in ("openai", "openai_compatible"):
            try:
                agent_outputs_by_stem = _run_agent_batch_openai(
                    model_router=model_router,
                    agent_model_id=agent_model_id,
                    record_items=record_items,
                    poll_seconds=max(5, int(args.batch_poll_seconds)),
                    log=log,
                )
            except Exception as e:
                log.warning(
                    "openai_batch_failed_fallback_to_realtime",
                    agent_model=agent_model_id,
                    error=str(e),
                )
                agent_outputs_by_stem = {}

        async def _run_pair_async() -> None:
            semaphore = asyncio.Semaphore(max(1, int(args.concurrency)))
            progress = {"done": 0}
            progress_lock = asyncio.Lock()

            async def _run_one(task_id: str, cfg: str, eval_input, stem: str):
                async with semaphore:
                    log.info(
                        "eval_task_start",
                        task_id=task_id,
                        config=cfg,
                        total=total,
                        agent_model=agent_model_id,
                        judge_model=judge_model_id,
                    )
                    try:
                        if stem in agent_outputs_by_stem:
                            agent_output = agent_outputs_by_stem[stem]
                        else:
                            agent_output = await asyncio.to_thread(
                                run_agent, eval_input, agent_model_id, model_router, int(args.agent_max_tokens)
                            )
                        if not bool(agent_output.parse_success):
                            # Strict mode: parse failure yields hard-zero scoring and skips judge call.
                            judge_output = JudgeOutput(
                                task_id=eval_input.task_id,
                                config_name=eval_input.config_name,
                                agent_classifications=[],
                                human_comment_statuses=[
                                    HumanCommentStatus(
                                        comment_id=str(c.get("comment_id") or ""),
                                        status="MISSED",
                                        matched_agent_comment_id=None,
                                    )
                                    for c in eval_input.human_comments
                                ],
                                judge_model=judge_model_id,
                                judge_prompt_version="v1.0+agent_parse_failed",
                            )
                        else:
                            judge_output = await asyncio.to_thread(
                                run_judge, eval_input, agent_output, judge_model_id, model_router
                            )
                        scores = compute_dimension_scores(eval_input, agent_output, judge_output)
                        model_label = f"{agent_model_id}::judge={judge_model_id}"
                        result = assemble_eval_result(eval_input, agent_output, judge_output, scores, model_label)
                        failures = validate_eval_result(result)
                        if not agent_output.parse_success:
                            failures = list(failures) + [f"agent_parse_failed: {agent_output.parse_error}"]
                        if str(judge_output.judge_prompt_version).endswith("+fallback"):
                            failures = list(failures) + ["judge_parse_fallback_used"]
                        if failures:
                            validation_failures.append(
                                {
                                    "task_id": task_id,
                                    "config_name": cfg,
                                    "agent_model": agent_model_id,
                                    "judge_model": judge_model_id,
                                    "failures": failures,
                                }
                            )
                        write_json(asdict(agent_output), run_root / "agent_outputs" / f"{stem}_agent.json")
                        write_json(asdict(judge_output), run_root / "judge_outputs" / f"{stem}_judge.json")
                        write_json(asdict(result), run_root / "eval_results" / f"{stem}_eval.json")
                        results.append(result)
                    except Exception as e:
                        # Keep the full run alive even if one task/config pair fails.
                        validation_failures.append(
                            {
                                "task_id": task_id,
                                "config_name": cfg,
                                "agent_model": agent_model_id,
                                "judge_model": judge_model_id,
                                "failures": [f"pipeline_error: {e}"],
                            }
                        )
                        write_json(
                            {
                                "task_id": task_id,
                                "config_name": cfg,
                                "model": agent_model_id,
                                "raw_response": "",
                                "comments": [],
                                "parse_success": False,
                                "parse_error": f"pipeline_error: {e}",
                            },
                            run_root / "agent_outputs" / f"{stem}_agent.json",
                        )
                    finally:
                        async with progress_lock:
                            progress["done"] += 1
                            done = progress["done"]
                        log.info(
                            "eval_progress",
                            task_id=task_id,
                            config=cfg,
                            done=done,
                            total=total,
                            agent_model=agent_model_id,
                            judge_model=judge_model_id,
                        )

            coros = [_run_one(task_id, cfg, eval_input, stem) for task_id, cfg, eval_input, stem in record_items]
            await asyncio.gather(*coros)

        asyncio.run(_run_pair_async())

        # Rebuild report from all persisted eval result files so partial reruns
        # preserve full-run A/B/C coverage in eval_report.
        all_results = _load_all_eval_results(run_root)
        report_rows = all_results or results
        report = generate_eval_report(report_rows, str(run_root / "eval_report.json"))
        write_json(validation_failures, run_root / "validation_failures.json")
        log.info(
            "eval_model_pair_done",
            agent_model=agent_model_id,
            judge_model=judge_model_id,
            eval_records=len(report_rows),
            prs=len(task_ids),
            output=str(run_root),
            validation_failures=len(validation_failures),
        )
        sweep_summary.append(
            {
                "agent_model": agent_model_id,
                "judge_model": judge_model_id,
                "run_dir": str(run_root),
                "eval_records": len(report_rows),
                "prs": len(task_ids),
                "validation_failures": len(validation_failures),
                "report_run_date": report.get("run_date"),
            }
        )

    if len(model_pairs) > 1:
        write_json({"runs": sweep_summary}, output_root / "multi_model_summary.json")
    print(json.dumps({"runs": sweep_summary}, indent=2))


if __name__ == "__main__":
    main()

