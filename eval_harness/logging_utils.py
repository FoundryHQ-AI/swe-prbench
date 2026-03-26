"""
Structured logging for the eval harness (stdlib / structlog).
Standalone copy — no dependency on data collection packages.
"""

import sys
from datetime import datetime
from pathlib import Path
from threading import Lock

try:
    import structlog

    _HAS_STRUCTLOG = True
except ImportError:
    _HAS_STRUCTLOG = False

_FALLBACK_LEVEL = 20  # INFO
_LOG_FILE_PATH: str | None = None
_LOG_FILE_LOCK = Lock()


def _append_file_log(line: str) -> None:
    if not _LOG_FILE_PATH:
        return
    try:
        p = Path(_LOG_FILE_PATH)
        p.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_FILE_LOCK:
            with open(p, "a", encoding="utf-8") as f:
                f.write(line)
                if not line.endswith("\n"):
                    f.write("\n")
    except Exception:
        return


def _fallback_log(level: str, event: str, **kwargs):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    extra = " ".join(f"{k}={v}" for k, v in kwargs.items()) if kwargs else ""
    line = f"{ts} [{level.upper():5}] {event} {extra}".strip()
    print(line, flush=True)
    _append_file_log(line)


def setup_logging(verbose: bool = True, level: str = "INFO", log_file_path: str | None = None) -> None:
    global _FALLBACK_LEVEL, _LOG_FILE_PATH
    _FALLBACK_LEVEL = {"debug": 10, "info": 20, "warning": 30, "error": 40}.get(level.lower(), 20)
    _LOG_FILE_PATH = log_file_path

    if not _HAS_STRUCTLOG:
        return

    levels = {"debug": 10, "info": 20, "warning": 30, "error": 40}
    min_level = levels.get(level.lower(), 20)

    def filter_by_level(logger, method_name, event_dict):
        if levels.get(method_name, 0) < min_level:
            raise structlog.DropEvent
        return event_dict

    def file_sink_processor(logger, method_name, event_dict):
        ts = event_dict.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        event = event_dict.get("event", "")
        parts = []
        for k, v in event_dict.items():
            if k in ("timestamp", "event", "level"):
                continue
            parts.append(f"{k}={v}")
        level_up = str(event_dict.get("level", method_name)).upper()
        line = f"{ts} [{level_up:5}] {event} {' '.join(parts)}".strip()
        _append_file_log(line)
        return event_dict

    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        filter_by_level,
        structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S"),
        structlog.processors.StackInfoRenderer(),
        file_sink_processor,
    ]
    if verbose:
        processors.append(structlog.dev.ConsoleRenderer(colors=True))
    else:
        processors.append(structlog.processors.JSONRenderer())

    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = "swe_prbench_eval"):
    if _HAS_STRUCTLOG:
        return structlog.get_logger(name)

    levels = {"debug": 10, "info": 20, "warning": 30, "error": 40}
    min_level = _FALLBACK_LEVEL

    class _FallbackLogger:
        def _log(self, level: str, event: str, **kwargs):
            if levels.get(level, 0) >= min_level:
                _fallback_log(level, event, **kwargs)

        def info(self, event: str, **kwargs):
            self._log("info", event, **kwargs)

        def warning(self, event: str, **kwargs):
            self._log("warning", event, **kwargs)

        def error(self, event: str, **kwargs):
            self._log("error", event, **kwargs)

        def debug(self, event: str, **kwargs):
            self._log("debug", event, **kwargs)

    return _FallbackLogger()
