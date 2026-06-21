#!/usr/bin/env python3
"""Extract the eval_100 task_id list as a newline-separated file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        default="swe-prbench-data/dataset/evals/eval_100.json",
        help="Path to eval split JSON (default: eval_100.json from the dataset).",
    )
    parser.add_argument(
        "--out",
        default="claude_code_eval/results/eval_100_task_ids.txt",
        help="Output file for newline-separated task_ids.",
    )
    args = parser.parse_args()

    split_path = Path(args.split)
    if not split_path.exists():
        print(f"error: split file not found: {split_path}", file=sys.stderr)
        return 2

    data = json.loads(split_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        print(f"error: expected list at top level, got {type(data).__name__}", file=sys.stderr)
        return 2

    task_ids = [str(row.get("task_id")) for row in data if isinstance(row, dict) and row.get("task_id")]
    if not task_ids:
        print("error: no task_ids found in split", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(task_ids) + "\n", encoding="utf-8")
    print(f"wrote {len(task_ids)} task_ids -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
