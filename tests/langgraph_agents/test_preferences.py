"""Unit tests for user_preferences — GET/PATCH /me/preferences (ADR-007).

IDOR-safe: no user_id param, identity from Depends(current_user_id) only.
prefs is UI-only (no PHI), 8KB/depth/prototype guard, version 409.
"""

from unittest.mock import AsyncMock, MagicMock, patch
import json
import pytest
import pytest_asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph_agents.api.schemas import UserPreferencesPatch
from langgraph_agents.api.routes_preferences import _validate_prefs, MAX_PREFS_BYTES


@pytest.fixture(autouse=True)
def auth_env(monkeypatch):
    import langgraph_agents.api.auth as auth_mod
    monkeypatch.setenv("AUTH_PROVIDER", "cognito")
    monkeypatch.setenv("COGNITO_REGION", "us-east-1")
    monkeypatch.setenv("COGNITO_USER_POOL_ID", "us-east-1_testpool")
    monkeypatch.setenv("COGNITO_APP_CLIENT_ID", "test-client-id")
    auth_mod.get_auth_config.cache_clear()
    auth_mod.get_jwks_client.cache_clear()
    yield auth_mod
    auth_mod.get_auth_config.cache_clear()
    auth_mod.get_jwks_client.cache_clear()


# ── Schema / validation ─────────────────────────────────────────────────

@pytest.mark.unit
def test_patch_schema_valid():
    p = UserPreferencesPatch(version=1, avatar_bg="slate")
    assert p.avatar_bg == "slate"

@pytest.mark.unit
def test_patch_schema_rejects_invalid_avatar_bg():
    with pytest.raises(Exception):
        UserPreferencesPatch(version=1, avatar_bg="red")  # type: ignore

@pytest.mark.unit
def test_validate_prefs_ok():
    _validate_prefs({"locale": "vi", "notifications": {"email": True}})

@pytest.mark.unit
def test_validate_prefs_too_large():
    big = "x" * (MAX_PREFS_BYTES + 1)
    with pytest.raises(Exception) as ei:
        _validate_prefs({"big": big})
    assert ei.value.status_code == 413

@pytest.mark.unit
def test_validate_prefs_too_deep():
    with pytest.raises(Exception) as ei:
        _validate_prefs({"a": {"b": {"c": 1}}})
    assert ei.value.status_code == 400

@pytest.mark.unit
def test_validate_prefs_proto_pollution():
    for k in ["__proto__", "constructor", "prototype"]:
        with pytest.raises(Exception) as ei:
            _validate_prefs({k: {}})
        assert ei.value.status_code == 400

@pytest.mark.unit
def test_validate_prefs_phi_guard():
    for k in ["injury_history", "fitness_level", "age"]:
        with pytest.raises(Exception) as ei:
            _validate_prefs({k: "x"})
        assert "PHI" in str(ei.value.detail)

@pytest.mark.unit
def test_validate_prefs_nested_phi_not_blocked():
    # Only top-level PHI keys are blocked — nested is allowed (still UI-only by convention)
    _validate_prefs({"a": {"injury_history": "x"}})


# ── Route: IDOR + auth + version ────────────────────────────────────────

def _app_with_prefs(uid: str):
    """FastAPI with /me/preferences mounted and current_user_id overridden."""
    from langgraph_agents.api.routes_preferences import router as prefs_router
    from langgraph_agents.api.auth import current_user_id, override_user
    app = FastAPI()
    app.include_router(prefs_router)
    app.dependency_overrides[current_user_id] = override_user(uid)
    return app


@pytest.mark.unit
def test_get_requires_auth():
    from langgraph_agents.api.routes_preferences import router as prefs_router
    app = FastAPI()
    app.include_router(prefs_router)
    # no override → no token → 401
    c = TestClient(app)
    assert c.get("/me/preferences").status_code == 401


@pytest.mark.unit
def test_get_autoseeds_and_returns_prefs():
    uid = "00000000-0000-4000-a000-000000000001"
    app = _app_with_prefs(uid)

    fake_row = {
        "avatar_bg": "slate",
        "selected_character_slug": None,
        "display_name": None,
        "prefs": {},
        "version": 1,
        "updated_at": MagicMock(isoformat=lambda: "2026-09-01T00:00:00+00:00"),
    }

    mock_pg = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetchrow = AsyncMock(return_value=fake_row)
    mock_conn.execute = AsyncMock(return_value=None)
    # transaction() is async context manager
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    mock_pg.transaction.return_value = mock_tx

    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=mock_pg):
        c = TestClient(app)
        r = c.get("/me/preferences")
        assert r.status_code == 200
        body = r.json()
        assert body["avatar_bg"] == "slate"
        assert body["version"] == 1
        assert "ETag" in r.headers


@pytest.mark.unit
def test_patch_409_on_stale_version():
    uid = "00000000-0000-4000-a000-000000000002"
    app = _app_with_prefs(uid)

    mock_pg = MagicMock()
    mock_conn = AsyncMock()
    # First fetchrow in PATCH returns None (no row updated due to version mismatch)
    # Second fetchrow returns current version
    mock_conn.fetchrow = AsyncMock(side_effect=[
        None,  # UPDATE ... RETURNING → no row
        {"version": 5},  # SELECT version → current is 5
    ])
    mock_conn.fetchval = AsyncMock(return_value=None)
    mock_conn.execute = AsyncMock(return_value=None)
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    mock_pg.transaction.return_value = mock_tx

    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=mock_pg):
        c = TestClient(app)
        r = c.patch("/me/preferences", json={"avatar_bg": "violet", "version": 1})
        assert r.status_code == 409
        assert r.json()["detail"]["error"] == "version_conflict"


@pytest.mark.unit
def test_patch_rejects_unknown_character():
    uid = "00000000-0000-4000-a000-000000000003"
    app = _app_with_prefs(uid)

    mock_pg = MagicMock()
    mock_conn = AsyncMock()
    mock_conn.fetchval = AsyncMock(return_value=None)  # character not found
    mock_conn.execute = AsyncMock(return_value=None)
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_tx = MagicMock()
    mock_tx.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_tx.__aexit__ = AsyncMock(return_value=None)
    mock_pg.transaction.return_value = mock_tx

    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=mock_pg):
        c = TestClient(app)
        r = c.patch("/me/preferences", json={"selected_character_slug": "fake_slug", "version": 1})
        assert r.status_code == 400
        assert "unknown" in r.json()["detail"].lower()


@pytest.mark.unit
def test_no_user_id_param_on_routes():
    """IDOR fix: routes must not accept user_id from client."""
    from langgraph_agents.api.routes_preferences import router
    import inspect
    for route in router.routes:
        params = list(inspect.signature(route.endpoint).parameters)
        assert "user_id" not in params, f"{route.path} exposes user_id param"
        assert "uid" in params or "user_id" not in params
    # Also ensure no path contains {user_id}
    for route in router.routes:
        assert "{user_id" not in route.path
        assert "{uid" not in route.path


@pytest.mark.unit
def test_prefs_merge_does_not_drop_existing_keys():
    """Client PATCH with partial prefs must not wipe other keys (prefs || jsonb)."""
    # This is a code inspection test — the query uses `prefs || $4::jsonb`
    import pathlib
    src = pathlib.Path("agenticRAG/langgraph_agents/api/routes_preferences.py").read_text()
    assert "prefs ||" in src or "prefs ||" in src.replace(" ", "")
    assert "version + 1" in src
