"""Tests for structured JSON logging (Phase 6 P0.1)."""

import asyncio
import json
import logging
from io import StringIO

import pytest

from langgraph_agents.shared.logging import (
    JsonFormatter, configure_root_logger, get_logger, with_request_id,
)


@pytest.fixture
def log_stream():
    """Fixture: configure a fresh StreamHandler → StringIO, return the buffer."""
    stream = StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("test_logging")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    yield logger, stream
    logger.handlers.clear()


@pytest.mark.unit
def test_log_format_is_json(log_stream):
    logger, stream = log_stream
    with with_request_id("r1"):
        logger.info("test_event", extra={"foo": 42})
    line = stream.getvalue().strip()
    record = json.loads(line)
    assert record["ts"]
    assert record["lvl"] == "INFO"
    assert record["logger"] == "test_logging"
    assert record["msg"] == "test_event"
    assert record["request_id"] == "r1"
    assert record["foo"] == 42


@pytest.mark.unit
def test_request_id_propagates_in_context(log_stream):
    logger, stream = log_stream
    with with_request_id("abc-123"):
        logger.info("inside_context")
    line = stream.getvalue().strip()
    record = json.loads(line)
    assert record["request_id"] == "abc-123"


@pytest.mark.unit
def test_request_id_default_when_outside_context(log_stream):
    logger, stream = log_stream
    logger.info("no_context")
    line = stream.getvalue().strip()
    record = json.loads(line)
    assert record["request_id"] == "-"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_request_id_isolated_across_async_tasks():
    """Two concurrent tasks with different request_ids must not leak."""
    results = {}

    async def task(name: str, rid: str):
        stream = StringIO()
        handler = logging.StreamHandler(stream)
        handler.setFormatter(JsonFormatter())
        t_logger = logging.getLogger(f"test_isolate_{name}")
        t_logger.setLevel(logging.DEBUG)
        t_logger.handlers.clear()
        t_logger.addHandler(handler)
        t_logger.propagate = False
        with with_request_id(rid):
            t_logger.info("event_a")
            await asyncio.sleep(0.02)
            t_logger.info("event_b")
        results[name] = stream.getvalue().strip()

    await asyncio.gather(task("A", "rid-A"), task("B", "rid-B"))

    for name, raw in results.items():
        lines = raw.splitlines()
        assert len(lines) == 2, f"Task {name}: expected 2 log lines"
        for line in lines:
            record = json.loads(line)
            assert record["request_id"] == f"rid-{name}", (
                f"Task {name}: expected rid-{name}, got {record['request_id']}"
            )


@pytest.mark.unit
def test_extra_fields_merged(log_stream):
    logger, stream = log_stream
    with with_request_id("x"):
        logger.info("extra_test", extra={"node": "planner", "elapsed_ms": 42})
    line = stream.getvalue().strip()
    record = json.loads(line)
    assert record["node"] == "planner"
    assert record["elapsed_ms"] == 42


@pytest.mark.unit
def test_exc_info_included(log_stream):
    logger, stream = log_stream
    with with_request_id("err-1"):
        try:
            raise ValueError("boom")
        except ValueError:
            logger.error("error_event", exc_info=True)
    line = stream.getvalue().strip()
    record = json.loads(line)
    assert record["lvl"] == "ERROR"
    assert "exc" in record
    assert "ValueError" in record["exc"]


@pytest.mark.unit
def test_configure_root_logger_sets_json_handler():
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    try:
        configure_root_logger(level="DEBUG")
        assert len(root.handlers) == 1
        handler = root.handlers[0]
        assert isinstance(handler.formatter, JsonFormatter)
        assert root.level == logging.DEBUG
    finally:
        root.handlers.clear()
        for h in original_handlers:
            root.addHandler(h)
