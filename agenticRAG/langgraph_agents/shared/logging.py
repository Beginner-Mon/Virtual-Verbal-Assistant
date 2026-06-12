"""Structured JSON logging with request_id correlation.

Usage in nodes:
    from langgraph_agents.shared.logging import get_logger, with_request_id

    logger = get_logger("langgraph.planner")

    async def planner_node(state, config):
        request_id = config["configurable"].get("request_id", "-")
        with with_request_id(request_id):
            logger.info("planner_start", extra={"intent_hint": ...})
            ...
"""

import contextvars
import json
import logging
import logging.handlers
import os
from contextlib import contextmanager
from datetime import datetime, timezone

_request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")


class JsonFormatter(logging.Formatter):
    """Emits one JSON object per log line with ts, lvl, logger, msg, request_id."""

    def formatTime(self, record, datefmt=None):
        """Override to emit ISO 8601 UTC with microseconds (portable on Windows)."""
        ct = self.converter(record.created)
        return datetime.fromtimestamp(record.created, tz=timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record),
            "lvl": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": _request_id_var.get(),
        }
        # Merge extra fields (anything outside the standard LogRecord attrs)
        std_attrs = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {
            "message", "asctime",
        }
        for k, v in record.__dict__.items():
            if k not in std_attrs:
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


@contextmanager
def with_request_id(request_id: str):
    """Set request_id in context for the duration of this block.

    Usage:
        with with_request_id(request_id):
            logger.info("something")  # carries request_id
    """
    token = _request_id_var.set(request_id or "-")
    try:
        yield
    finally:
        _request_id_var.reset(token)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given name. Use as drop-in for logging.getLogger."""
    return logging.getLogger(name)


def configure_root_logger(level: str = "INFO") -> None:
    """Call once at FastAPI startup (lifespan).

    Replaces all root handlers. Always adds a StreamHandler (stderr).
    If LOG_FILE is set, also adds a RotatingFileHandler (10 MB × 5 backups).
    """
    root = logging.getLogger()
    root.setLevel(level)
    for h in root.handlers[:]:
        root.removeHandler(h)

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    log_file = os.getenv("LOG_FILE")
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
