"""Tests for api/crud_app.py — the app the CRUD Lambda runs.

Two things are worth testing here and they are both about what is ABSENT:
the heavy imports the split exists to avoid, and any way to be served without a
token.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_AGENTIC_ROOT = Path(__file__).resolve().parents[2] / "agenticRAG"


@pytest.mark.unit
def test_crud_app_does_not_import_the_heavy_stack():
    """Importing crud_app must not drag in torch, LangChain, or the graph.

    This is the whole reason the split exists — serving GET /sessions should not
    require a process that loads sentence_transformers at boot. The check runs in
    a FRESH interpreter on purpose: inside the test session those modules have
    almost certainly been imported by some other test, so asserting against this
    process's sys.modules would pass for the wrong reason and keep passing after
    the property was lost.
    """
    probe = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(_AGENTIC_ROOT)!r})

        import langgraph_agents.api.crud_app  # noqa: F401

        forbidden = [
            m for m in (
                "torch",
                "sentence_transformers",
                "langchain_openai",
                "langgraph_agents.graph",
                "langgraph_agents.nodes.summarizer",
                "langgraph_agents.shared.embedding",
            )
            if m in sys.modules
        ]
        print("LEAKED:" + ",".join(forbidden))
    """)

    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, f"probe failed:\n{result.stderr}"

    leaked = [
        line[len("LEAKED:"):]
        for line in result.stdout.splitlines()
        if line.startswith("LEAKED:")
    ]
    assert leaked, f"probe produced no verdict:\n{result.stdout}\n{result.stderr}"
    assert leaked[0] == "", (
        f"crud_app pulled in {leaked[0]}. Something imported from this module "
        f"reaches the graph or the embedding stack — find it before the Lambda "
        f"package grows by 2 GB."
    )


@pytest.mark.unit
@pytest.mark.parametrize("method,path", [
    ("GET", "/sessions"),
    ("GET", "/sessions/00000000-0000-0000-0000-000000000009"),
    ("DELETE", "/sessions/00000000-0000-0000-0000-000000000009"),
    ("GET", "/me/memory"),
    ("POST", "/me/memory"),
    ("DELETE", "/me/memory/00000000-0000-0000-0000-000000000009"),
])
def test_every_crud_route_requires_a_token(method, path):
    """No route serves an unauthenticated caller, in any environment.

    Parametrised over all of them rather than spot-checking one: the failure
    this guards against is a single route added later without the dependency.
    """
    from langgraph_agents.api.crud_app import create_crud_app

    # Deliberately NOT `with TestClient(...)`. The context manager runs the
    # lifespan, which opens a connection to Neon in us-east-1 — about 7 seconds
    # per test, for a check that never reaches the database because the request
    # is rejected before the handler runs.
    client = TestClient(create_crud_app())
    response = client.request(method, path, json={"fact_text": "x"})

    assert response.status_code == 401, (
        f"{method} {path} answered {response.status_code} without a token"
    )


@pytest.mark.unit
def test_health_needs_no_token():
    """The warmer and the load balancer probe must not need a credential."""
    from langgraph_agents.api.crud_app import create_crud_app

    client = TestClient(create_crud_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.unit
def test_crud_routes_run_no_vector_queries():
    """Vector search belongs to the agent, not here. Owner's decision, 18-08.

    This records that decision rather than guessing at one, and it happens to be
    load-bearing for a second reason: db/postgres.py opens each connection with
    `SET hnsw.iterative_scan`, which keeps recall from collapsing when a filter
    prunes HNSW candidates. That is a session setting, and this app runs against
    Neon's POOLED endpoint, where PgBouncer hands out a different backend per
    transaction — so the SET lands on whichever backend was lent at init time and
    a later query may run somewhere it never reached.

    So a vector query added here would not fail. It would return worse results on
    some requests, with nothing in the logs — which is why this is a test and not
    a comment.
    """
    import ast
    from pathlib import Path

    source = (
        Path(__file__).resolve().parents[2]
        / "agenticRAG" / "langgraph_agents" / "api" / "routes_crud.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    # Docstrings, not code. The module docstring says "no embeddings", which a
    # text search reads as an embedding query — the same trap as scanning for
    # app.user_id by line in test_pg_user_scope.py. Comments are absent from the
    # AST entirely, so only executed string literals remain.
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr):
                first = node.body[0].value
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    docstrings.add(id(first))

    markers = ("<=>", "<->", "<#>", "embedding", "::vector", "kb_embeddings")
    offenders = [
        (marker, node.value.strip()[:80])
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        for marker in markers
        if marker in node.value
    ]

    assert not offenders, (
        f"routes_crud.py appears to query vectors: {offenders}. Through the "
        f"pooled endpoint the hnsw.iterative_scan setting is unreliable, so "
        f"recall would vary per request. Move the query to the agent, or set the "
        f"GUC inside the same transaction as the search."
    )


@pytest.mark.unit
def test_crud_app_serves_no_graph_routes():
    """/chat must 404 here — proof the graph is not linked into this app."""
    from langgraph_agents.api.crud_app import create_crud_app

    client = TestClient(create_crud_app())
    assert client.post("/chat", json={"query": "hi"}).status_code == 404
