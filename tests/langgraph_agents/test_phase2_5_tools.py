"""Tests for pgvector @tool (Task 2.5.5)."""

import pytest

from langgraph_agents.tools.pgvector_tool import pgvector_search


@pytest.mark.unit
def test_pgvector_tool_metadata():
    assert pgvector_search.name == "pgvector_search"
    assert pgvector_search.description
    assert "knowledge base" in pgvector_search.description.lower()
