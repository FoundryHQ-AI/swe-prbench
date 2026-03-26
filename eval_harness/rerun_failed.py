#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import yaml


CONFIG_ALIAS = {
    "config_A": "A",
    "config_B": "B",
    "config_C": "C",
    "config_A_diff_only": "A",
    "config_B_with_file_content": "B",
    "config_C_full_context": "C",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm_config_short(config_name: str) -> str | None:
    c = (config_name or "").strip()
    return CONFIG_ALIAS.get(c)


def _looks_like_suspicious_empty_success(record: dict[str, Any]) -> bool:
    if not bool(record.get("parse_success")):
        return False
    comments = record.get("comments")
    if not isinstance(comments, list) or comments:
        return False
    raw = str(record.get("raw_response") or "").strip()
    if not raw:
        return False
    if raw in {"[]", "{}"}:
        return False
    # Heuristics: looks like structured/prose output that should have parsed.
    probes = ['"body"', "```json", "severity", "file", "line", "["]
    return any(p in raw for p in probes)


def _is_empty_agent_output(record: dict[str, Any]) -> bool:
    """True if parse succeeded but comments list is empty (model returned [])."""
    if not bool(record.get("parse_success")):
        return False
    comments = record.get("comments")
    return isinstance(comments, list) and len(comments) == 0


def _classify_issue(record: dict[str, Any], include_empty: bool = False) -> str | None:
    parse_success = bool(record.get("parse_success"))
    parse_error = record.get("parse_error")
    if (not parse_success) or (parse_error not in (None, "")):
        err = str(parse_error or "")
        if "429" in err or "rate limit" in err.lower():
            return "rate_limit_or_transient_api_failure"
        return "agent_parse_or_request_failure"
    if _looks_like_suspicious_empty_success(record):
        return "suspicious_empty_comments_with_raw_response"
    if include_empty and _is_empty_agent_output(record):
        return "empty_comments_retry"
    return None


def _discover_failures(run_dir: Path, include_empty: bool = False) -> tuple[dict[str, set[str]], list[dict[str, Any]]]:
    """
    Returns:
      - map: short_config(A/B/C) -> task_id set
      - detailed failure records
    """
    out: dict[str, set[str]] = {"A": set(), "B": set(), "C": set()}
    details: list[dict[str, Any]] = []
    agent_dir = run_dir / "agent_outputs"
    if not agent_dir.exists():
        raise FileNotFoundError(f"agent_outputs not found: {agent_dir}")

    for p in sorted(agent_dir.glob("*_agent.json")):
        try:
            rec = _load_json(p)
        except Exception:
            continue
        if not isinstance(rec, dict):
            continue
        task_id = str(rec.get("task_id") or "").strip()
        cfg = _norm_config_short(str(rec.get("config_name") or ""))
        if not task_id or not cfg:
            continue

        issue = _classify_issue(rec, include_empty=include_empty)
        if issue:
            out[cfg].add(task_id)
            details.append(
                {
                    "task_id": task_id,
                    "config": cfg,
                    "issue_type": issue,
                    "agent_output_file": str(p),
                    "parse_success": bool(rec.get("parse_success")),
                    "parse_error": rec.get("parse_error"),
                }
            )
    return out, details


def _parse_models_from_run_dir(run_dir: Path) -> tuple[str, str | None]:
    """
    Expects folder like:
      <agent_model>__judge_<judge_model>
    """
    name = run_dir.name
    marker = "__judge_"
    if marker not in name:
        return name, None
    agent, judge = name.split(marker, 1)
    return agent.strip(), judge.strip() or None


def _fixed_judge_from_config(model_config_path: Path) -> str | None:
    try:
        cfg = yaml.safe_load(model_config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    defaults = cfg.get("defaults") or {}
    if not isinstance(defaults, dict):
        return None
    j = defaults.get("judge_model")
    if isinstance(j, str) and j.strip():
        return j.strip()
    return None


def _build_command(
    python_exe: str,
    contexts: str,
    annotations: str,
    prs: str,
    output_root: str,
    model_config: str,
    agent_model: str,
    judge_model: str | None,
    config_short: str,
    task_ids: list[str],
    concurrency: int,
) -> str:
    base = [
        python_exe,
        "eval_harness/run_eval.py",
        "--contexts",
        contexts,
        "--annotations",
        annotations,
        "--prs",
        prs,
        "--output",
        output_root,
        "--model-config",
        model_config,
        "--agent-models",
        agent_model,
        "--configs",
        config_short,
        "--concurrency",
        str(max(1, int(concurrency))),
        "--task-ids",
        *task_ids,
    ]
    if judge_model:
        base.extend(["--judge-model", judge_model])
    return " ".join(base)


def _execute_commands(commands: list[str], stop_on_error: bool) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for idx, cmd in enumerate(commands, start=1):
        proc = subprocess.run(cmd, shell=True, text=True, capture_output=True)
        item = {
            "index": idx,
            "command": cmd,
            "returncode": int(proc.returncode),
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
        }
        results.append(item)
        if proc.returncode != 0 and stop_on_error:
            break
    return results


def _discover_run_dirs(run_dir: Path | None, evals_root: Path | None) -> list[Path]:
    if run_dir is not None:
        if not run_dir.exists():
            return []
        return [run_dir.resolve()]
    if evals_root is None:
        return []
    if not evals_root.exists() or not evals_root.is_dir():
        return []
    out: list[Path] = []
    for child in sorted(evals_root.resolve().iterdir()):
        if child.is_dir() and (child / "agent_outputs").exists():
            out.append(child)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Parse existing eval agent outputs, record failed/suspicious records, and generate rerun commands."
        )
    )
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--run-dir",
        help="Path to one model-pair run directory, e.g. evals_fast/gpt_4o__judge_claude_sonnet_46",
    )
    target_group.add_argument(
        "--evals-root",
        help="Path to eval root containing many model-pair run dirs (each with agent_outputs/).",
    )
    target_group.add_argument(
        "--from-manifest",
        help="Execute commands from an existing rerun_manifest.json (no scanning).",
    )
    parser.add_argument("--contexts", required=True, help="Contexts directory path")
    parser.add_argument("--annotations", required=True, help="Annotations directory path")
    parser.add_argument("--prs", required=True, help="Path to prs.jsonl")
    parser.add_argument("--output-root", required=True, help="Eval output root (parent of model-pair dirs)")
    parser.add_argument("--model-config", required=True, help="Model config YAML path")
    parser.add_argument(
        "--python-exe",
        default="/Users/deepak/swe-prbench/venv/bin/python",
        help="Python executable used to run eval_harness/run_eval.py",
    )
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrency for reruns")
    parser.add_argument(
        "--manifest-out",
        default=None,
        help=(
            "Where to save rerun manifest JSON. "
            "Defaults to <run-dir>/rerun_manifest.json or <evals-root>/rerun_manifest.json."
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute generated rerun commands immediately after writing manifest.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="With --execute, stop at first command failure.",
    )
    parser.add_argument(
        "--include-empty",
        action="store_true",
        help="Also retry records where model returned [] (empty comments). Use for conservative models like Gemini.",
    )
    args = parser.parse_args()

    if args.from_manifest:
        manifest_path = Path(args.from_manifest).resolve()
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest file not found: {manifest_path}")
        payload = _load_json(manifest_path)
        commands = payload.get("commands") if isinstance(payload, dict) else None
        if not isinstance(commands, list):
            raise ValueError("Invalid manifest: missing top-level 'commands' array.")
        commands = [str(c).strip() for c in commands if str(c).strip()]
        if not commands:
            print(
                json.dumps(
                    {"manifest": str(manifest_path), "status": "ok", "message": "No commands to execute."},
                    indent=2,
                )
            )
            return
        if not args.execute:
            print(
                json.dumps(
                    {
                        "manifest": str(manifest_path),
                        "commands_count": len(commands),
                        "message": "Use --execute to run manifest commands.",
                    },
                    indent=2,
                )
            )
            return
        exec_results = _execute_commands(commands, stop_on_error=bool(args.stop_on_error))
        payload["execution"] = {
            "attempted": len(exec_results),
            "succeeded": len([r for r in exec_results if r.get("returncode") == 0]),
            "failed": len([r for r in exec_results if r.get("returncode") != 0]),
            "stop_on_error": bool(args.stop_on_error),
            "results": exec_results,
        }
        manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path),
                    "execution": {
                        "attempted": payload["execution"]["attempted"],
                        "succeeded": payload["execution"]["succeeded"],
                        "failed": payload["execution"]["failed"],
                    },
                },
                indent=2,
            )
        )
        return

    run_dir = Path(args.run_dir).resolve() if args.run_dir else None
    evals_root = Path(args.evals_root).resolve() if args.evals_root else None
    run_dirs = _discover_run_dirs(run_dir, evals_root)
    if not run_dirs:
        target = str(run_dir or evals_root or "")
        raise FileNotFoundError(
            "No run directories found (expected model run dirs containing agent_outputs).\n"
            f"Checked: {target}\n"
            "If you deleted eval outputs, run full eval first, or execute a previously saved manifest via --from-manifest."
        )

    fixed_judge = _fixed_judge_from_config(Path(args.model_config).resolve())

    manifest_runs: list[dict[str, Any]] = []
    all_commands: list[str] = []
    grand_total = 0

    for rd in run_dirs:
        failures_by_cfg, failure_details = _discover_failures(rd, include_empty=args.include_empty)
        total = sum(len(v) for v in failures_by_cfg.values())
        if total == 0:
            manifest_runs.append(
                {
                    "run_dir": str(rd),
                    "failed_or_suspicious_records": 0,
                    "commands": [],
                    "failures": [],
                }
            )
            continue
        grand_total += total

        agent_model, judge_from_dir = _parse_models_from_run_dir(rd)
        judge_model = None if (fixed_judge and fixed_judge == judge_from_dir) else judge_from_dir

        commands: list[str] = []
        for cfg in ("A", "B", "C"):
            task_ids = sorted(failures_by_cfg[cfg])
            if not task_ids:
                continue
            cmd = _build_command(
                python_exe=args.python_exe,
                contexts=args.contexts,
                annotations=args.annotations,
                prs=args.prs,
                output_root=args.output_root,
                model_config=args.model_config,
                agent_model=agent_model,
                judge_model=judge_model,
                config_short=cfg,
                task_ids=task_ids,
                concurrency=int(args.concurrency),
            )
            commands.append(cmd)
            all_commands.append(cmd)

        manifest_runs.append(
            {
                "run_dir": str(rd),
                "agent_model": agent_model,
                "judge_model": judge_from_dir,
                "failed_or_suspicious_records": total,
                "commands": commands,
                "failures": failure_details,
            }
        )

    if args.manifest_out:
        manifest_out = Path(args.manifest_out).resolve()
    elif run_dir is not None:
        manifest_out = run_dir / "rerun_manifest.json"
    else:
        manifest_out = (evals_root or Path.cwd()) / "rerun_manifest.json"
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "summary": {
            "runs_scanned": len(run_dirs),
            "runs_with_failures": len([r for r in manifest_runs if r.get("failed_or_suspicious_records", 0) > 0]),
            "failed_or_suspicious_records": grand_total,
            "commands_count": len(all_commands),
        },
        "runs": manifest_runs,
        "commands": all_commands,
    }

    if args.execute and all_commands:
        exec_results = _execute_commands(all_commands, stop_on_error=bool(args.stop_on_error))
        payload["execution"] = {
            "attempted": len(exec_results),
            "succeeded": len([r for r in exec_results if r.get("returncode") == 0]),
            "failed": len([r for r in exec_results if r.get("returncode") != 0]),
            "stop_on_error": bool(args.stop_on_error),
            "results": exec_results,
        }

    manifest_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    out = {"manifest": str(manifest_out), **payload["summary"]}
    if "execution" in payload:
        out["execution"] = {
            "attempted": payload["execution"]["attempted"],
            "succeeded": payload["execution"]["succeeded"],
            "failed": payload["execution"]["failed"],
        }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

