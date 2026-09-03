"""Unit tests for user preferences — GET/PATCH /me/preferences (ADR-008).

The shape under test changed in plan v3: preferences are one JSONB column on
`users`, the allowed keys are whatever SyncedPrefs declares, writes are
last-write-wins, and GET does not create anything.

Several tests here exist to keep a specific past mistake from coming back and
say so where they do.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from langgraph_agents.api.schemas import SyncedPrefs, UserPreferencesPatch


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


def _app(uid: str):
    """FastAPI with /me/preferences mounted and current_user_id overridden."""
    from langgraph_agents.api.routes_preferences import router as prefs_router
    from langgraph_agents.api.auth import current_user_id, override_user
    app = FastAPI()
    app.include_router(prefs_router)
    app.dependency_overrides[current_user_id] = override_user(uid)
    return app


def _mock_pg(*, fetchrow=None, fetchval=None):
    """A pg client whose transaction() yields a connection we can assert on."""
    conn = AsyncMock()
    conn.fetchrow = AsyncMock(**fetchrow) if fetchrow else AsyncMock(return_value=None)
    conn.fetchval = AsyncMock(**fetchval) if fetchval else AsyncMock(return_value=None)
    conn.execute = AsyncMock(return_value=None)
    tx = MagicMock()
    tx.__aenter__ = AsyncMock(return_value=conn)
    tx.__aexit__ = AsyncMock(return_value=None)
    pg = MagicMock()
    pg.transaction.return_value = tx
    return pg, conn


def _row(prefs: dict | str = "{}", updated="2026-09-02T00:00:00+00:00"):
    return {"preferences": prefs, "updated_at": updated}


# ── Schema: the whitelist is the guard ──────────────────────────────────

@pytest.mark.unit
def test_patch_schema_valid():
    p = UserPreferencesPatch(preferences={"avatar_bg": "slate"})
    assert p.preferences.avatar_bg == "slate"


@pytest.mark.unit
def test_unknown_key_rejected():
    """extra='forbid' — anything SyncedPrefs does not declare is refused."""
    with pytest.raises(Exception):
        SyncedPrefs.model_validate({"locale": "vi"})


@pytest.mark.unit
def test_phi_key_rejected():
    for key in ("injury_history", "fitness_level", "age", "medical_history"):
        with pytest.raises(Exception):
            SyncedPrefs.model_validate({key: "x"})


@pytest.mark.unit
def test_nested_phi_rejected():
    """The hole the old blocklist left open: it only looked at the top level.

    A whitelist closes it without knowing the word 'injury_history' at all —
    the outer key is what fails.
    """
    with pytest.raises(Exception):
        SyncedPrefs.model_validate({"a": {"injury_history": "thoát vị L4"}})


@pytest.mark.unit
def test_avatar_bg_junk_rejected_by_length_not_enum():
    with pytest.raises(Exception):
        SyncedPrefs.model_validate({"avatar_bg": "x" * 40})
    with pytest.raises(Exception):
        SyncedPrefs.model_validate({"avatar_bg": "Slate"})  # pattern is lowercase


@pytest.mark.unit
def test_unknown_colour_is_accepted():
    """D4: the palette lives in avatarPalette.ts and nowhere else.

    If someone reintroduces a Literal enum here, this test fails and points at
    the reason: the frontend deploys through Amplify and the backend through
    CDK, so a colour shipped to the UI first would 422 for whoever picked it.
    The UI looks the id up and falls back to 'slate', so this is inert.
    """
    assert SyncedPrefs.model_validate({"avatar_bg": "neon"}).avatar_bg == "neon"


@pytest.mark.unit
def test_unset_fields_are_not_written():
    """exclude_unset is what makes PATCH a merge rather than a replace."""
    patch = UserPreferencesPatch(preferences={"avatar_bg": "violet"})
    assert patch.preferences.model_dump(exclude_unset=True) == {"avatar_bg": "violet"}


@pytest.mark.unit
def test_explicit_null_clears_character():
    patch = UserPreferencesPatch(preferences={"selected_character_slug": None})
    assert patch.preferences.model_dump(exclude_unset=True) == {
        "selected_character_slug": None
    }


# ── Routes ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_get_requires_auth():
    from langgraph_agents.api.routes_preferences import router as prefs_router
    app = FastAPI()
    app.include_router(prefs_router)
    assert TestClient(app).get("/me/preferences").status_code == 401


@pytest.mark.unit
def test_get_returns_prefs_and_writes_nothing():
    """GET is a safe method again.

    The previous version seeded rows into `users` and `user_preferences` on
    every read, which made a cacheable read a write transaction and created
    rows for tokens that had never done anything.
    """
    uid = "00000000-0000-4000-a000-000000000001"
    pg, conn = _mock_pg(
        fetchrow={"return_value": _row('{"avatar_bg": "violet"}')},
    )
    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=pg):
        r = TestClient(_app(uid)).get("/me/preferences")

    assert r.status_code == 200
    assert r.json()["preferences"]["avatar_bg"] == "violet"
    conn.execute.assert_not_called()


@pytest.mark.unit
def test_get_with_no_row_returns_defaults():
    uid = "00000000-0000-4000-a000-000000000002"
    pg, _ = _mock_pg(fetchrow={"return_value": None})
    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=pg):
        r = TestClient(_app(uid)).get("/me/preferences")

    assert r.status_code == 200
    assert r.json()["preferences"] == {"avatar_bg": None, "selected_character_slug": None}


@pytest.mark.unit
def test_get_drops_stored_keys_that_no_longer_validate():
    """One stale row must not 500 for that user while everyone else is fine."""
    uid = "00000000-0000-4000-a000-000000000003"
    pg, _ = _mock_pg(
        fetchrow={"return_value": _row('{"avatar_bg": "violet", "legacy_flag": true}')},
    )
    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=pg):
        r = TestClient(_app(uid)).get("/me/preferences")

    assert r.status_code == 200
    assert r.json()["preferences"]["avatar_bg"] == "violet"


@pytest.mark.unit
def test_patch_rejects_inactive_character():
    """The regression 863458d introduced: a foreign key cannot see is_active."""
    uid = "00000000-0000-4000-a000-000000000004"
    pg, conn = _mock_pg(fetchval={"return_value": None})
    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=pg):
        r = TestClient(_app(uid)).patch(
            "/me/preferences", json={"preferences": {"selected_character_slug": "off"}},
        )

    assert r.status_code == 400
    assert "inactive" in r.json()["detail"].lower()
    conn.fetchrow.assert_not_called()  # rejected before the write


@pytest.mark.unit
def test_patch_accepts_active_character():
    uid = "00000000-0000-4000-a000-000000000005"
    pg, _ = _mock_pg(
        fetchval={"return_value": 1},
        fetchrow={"return_value": _row('{"selected_character_slug": "bronya"}')},
    )
    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=pg):
        r = TestClient(_app(uid)).patch(
            "/me/preferences",
            json={"preferences": {"selected_character_slug": "bronya"}},
        )

    assert r.status_code == 200
    assert r.json()["preferences"]["selected_character_slug"] == "bronya"


@pytest.mark.unit
def test_patch_rejects_unknown_key():
    uid = "00000000-0000-4000-a000-000000000006"
    pg, _ = _mock_pg()
    with patch("langgraph_agents.api.routes_preferences.get_pg_client", return_value=pg):
        r = TestClient(_app(uid)).patch(
            "/me/preferences", json={"preferences": {"injury_history": "L4"}},
        )
    assert r.status_code == 422


@pytest.mark.unit
def test_no_user_id_param_on_routes():
    """IDOR: routes must not accept a user id from the client."""
    from langgraph_agents.api.routes_preferences import router
    import inspect
    for route in router.routes:
        params = list(inspect.signature(route.endpoint).parameters)
        assert "user_id" not in params, f"{route.path} exposes user_id param"
        assert "{user_id" not in route.path
        assert "{uid" not in route.path


@pytest.mark.unit
def test_write_is_a_merge_and_carries_no_version():
    """Last-write-wins, per key.

    `preferences || $2::jsonb` is what makes two devices changing two different
    preferences both survive. A `version = version + 1` reappearing here would
    mean the optimistic lock is back, and with it a 409 whose resolution was
    always going to be "take the other write".
    """
    import pathlib
    src = pathlib.Path(
        "agenticRAG/langgraph_agents/api/routes_preferences.py"
    ).read_text(encoding="utf-8")
    assert "preferences || $2::jsonb" in src
    # The mechanics of the lock, not the word — the docstring explains why it is
    # gone and should be free to say so.
    assert "version + 1" not in src
    assert "AND version" not in src
    assert "409" not in src
