#!/usr/bin/env python3
"""Sustained-failure watchdog.

Reads CLAUDE_CLI_FAILURE_SUSTAINED lines on stdin (one per cli_claude
retry-loop exhaustion). Calls stop_all.sh only if at least
``THRESHOLD`` events arrive within ``WINDOW_SEC`` seconds — single
transient blips are absorbed silently.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

WINDOW_SEC = 600
THRESHOLD = 3

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
STOP_SCRIPT = os.path.join(REPO_ROOT, "claude_code_eval", "scripts", "stop_all.sh")


def main() -> int:
    hits: list[float] = []
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        now = time.time()
        hits = [t for t in hits if now - t < WINDOW_SEC]
        hits.append(now)
        sys.stdout.write(
            f"SUSTAINED_EVENT ({len(hits)}/{THRESHOLD} within {WINDOW_SEC}s): {line}\n"
        )
        sys.stdout.flush()
        if len(hits) >= THRESHOLD:
            sys.stdout.write(
                f"RATE_LIMIT_HIT: {THRESHOLD} sustained failures within "
                f"{WINDOW_SEC}s — stopping runs\n"
            )
            sys.stdout.flush()
            subprocess.run(["bash", STOP_SCRIPT], check=False)
            sys.stdout.write("RATE_LIMIT_HIT: all runs stopped, exiting watchdog\n")
            sys.stdout.flush()
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
