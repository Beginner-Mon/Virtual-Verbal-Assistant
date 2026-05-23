import sys
from pathlib import Path

import pytest

# Make langgraph_agents importable from tests
_project_root = Path(__file__).resolve().parents[2]
_agentic_root = _project_root / "agenticRAG" / "agentic_rag_gemini"
sys.path.insert(0, str(_agentic_root))

# Auto-load .env so HAS_*_KEY checks (and live LLM calls) see the keys.
try:
    from dotenv import load_dotenv
    load_dotenv(_agentic_root / ".env", override=False)
except ImportError:
    pass


@pytest.fixture(autouse=True)
async def _reset_pg_singleton_between_tests():
    """pytest-asyncio uses a fresh event loop per test (function scope).

    The shared PostgresClient holds an asyncpg pool whose connections are bound
    to whichever loop created them. Re-using that pool from a later test's loop
    raises `Event loop is closed`. Clear the singleton before each test so the
    next test rebuilds the pool on its own loop.
    """
    from langgraph_agents import shared
    if shared._pg_client is not None:
        try:
            await shared._pg_client.close()
        except Exception:
            pass
        shared._pg_client = None
    yield
    if shared._pg_client is not None:
        try:
            await shared._pg_client.close()
        except Exception:
            pass
        shared._pg_client = None
