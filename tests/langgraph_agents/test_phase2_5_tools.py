"""Tests for pgvector @tool (Task 2.5.5)."""

import pytest

from langgraph_agents.tools.pgvector_tool import kb_search


@pytest.mark.unit
def test_pgvector_tool_metadata():
    assert kb_search.name == "kb_search"
    assert kb_search.description
    assert "knowledge base" in kb_search.description.lower()
