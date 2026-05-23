"""Tests for memory node — recall detection (Task 2.5.4)."""

import pytest

from langgraph_agents.nodes.memory import _needs_recall


@pytest.mark.unit
@pytest.mark.parametrize("query,expected", [
    ("Xin chao", False),
    ("Hello", False),
    ("ban con nho bai tap tuan truoc khong", True),
    ("nho lai buoi tap hom qua", True),
    ("lan truoc minh da hoi ve dau lung", True),
    ("remember my last session", True),
    ("last time we discussed", True),
    ("truoc do toi bi dau vai", True),
    ("hom qua toi tap bai gi", True),
    ("da noi ve bai tap lung", True),
    ("tuan truoc co bai tap nao tot", True),
])
def test_recall_detection(query, expected):
    assert _needs_recall(query) == expected, f"Query: {query!r}"
