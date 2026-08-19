import sys
from pathlib import Path

import pytest

# Make langgraph_agents importable from tests
_project_root = Path(__file__).resolve().parents[2]
_agentic_root = _project_root / "agenticRAG"
sys.path.insert(0, str(_agentic_root))

# Auto-load .env so HAS_*_KEY checks (and live LLM calls) see the keys.
#
# Goes through the service's own loader rather than naming a path: tests that
# read a different .env from the application are tests of something else.
try:
    from langgraph_agents.shared.env import load_env

    load_env()
except ImportError:
    pass

# Deliberately NO auth configuration here. api/auth.py reads and validates its
# provider config lazily — importing it does nothing — so tests that never
# verify a token need none, and the ones that do set their own (see the
# auth_config fixture in test_auth.py). Everything else substitutes identity
# with app.dependency_overrides[current_user_id], which needs no config at all.
#
# An earlier version put fake COGNITO_* values here. That was the wrong place:
# it made every test in the suite carry credentials to satisfy one module's
# import, which is the smell that led to auth.py validating lazily instead.


@pytest.fixture
def pg_dsn_or_skip():
    """The application's own DSN, or skip.

    Integration tests that need a real database should skip rather than fail on
    a machine that has none — but they must talk to the SAME database the app
    resolves, not a DSN of their own invention. `migrate_messages.py` invented
    its own once and ran against a database that was not this one.
    """
    try:
        from langgraph_agents.db.postgres import get_default_dsn, _LOCAL_DSN

        dsn = get_default_dsn()
    except Exception as exc:                                   # noqa: BLE001
        pytest.skip(f"cannot resolve a DSN: {exc}")

    if dsn == _LOCAL_DSN:
        pytest.skip("no VVA_PG_DSN configured; refusing to guess at localhost")
    return dsn


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
