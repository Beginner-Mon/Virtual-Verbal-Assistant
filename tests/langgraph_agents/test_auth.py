"""Unit tests for api/auth.py — resolve_user_id behaviour.

All Cognito env vars default to empty in the test environment (REQUIRE_AUTH is
not set) so importing auth.py does NOT trigger the startup guard.  Each test
controls env vars and re-imports or patches the module state directly.

No network calls: jwt.decode and PyJWKClient are mocked throughout.
"""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from starlette.requests import Request as StarletteRequest


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_request(auth_header: str | None = None) -> StarletteRequest:
    """Build a minimal Starlette Request with optional Authorization header."""
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [],
    }
    if auth_header is not None:
        scope["headers"] = [(b"authorization", auth_header.encode())]
    return StarletteRequest(scope)


def _reload_auth(
    require_auth: str = "false",
    region: str = "",
    pool_id: str = "",
    client_id: str = "",
):
    """Re-import auth module with the given env vars patched.

    Returns the freshly imported module.
    """
    env_overrides = {
        "REQUIRE_AUTH": require_auth,
        "COGNITO_REGION": region,
        "COGNITO_USER_POOL_ID": pool_id,
        "COGNITO_APP_CLIENT_ID": client_id,
    }
    # Remove cached module so importlib loads it fresh.
    sys.modules.pop("langgraph_agents.api.auth", None)

    with patch.dict("os.environ", env_overrides, clear=False):
        # PyJWKClient tries to fetch JWKS on init when cache_keys=True and
        # keys_to_cache > 0 — patch it to avoid network.
        with patch("jwt.PyJWKClient", autospec=True):
            mod = importlib.import_module("langgraph_agents.api.auth")
    return mod


# ── Test 1: valid id token → sub returned ─────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.unit
async def test_valid_id_token_returns_sub():
    """A well-formed Cognito id-token should resolve to _to_uuid(sub)."""
    from langgraph_agents.db.session_store import _to_uuid
    import uuid

    sub = str(uuid.uuid4())
    fake_claims = {"sub": sub, "token_use": "id"}

    # Patch at the module level so the auth module uses the mock
    with patch("langgraph_agents.api.auth._verify_token", return_value=fake_claims):
        # Ensure the module is loaded
        import langgraph_agents.api.auth as auth_mod

        request = _make_request(f"Bearer fake.jwt.token")
        result = await auth_mod.resolve_user_id(request, "some_fallback")

    assert result == _to_uuid(sub)


# ── Test 2: no token + REQUIRE_AUTH=false → fallback ──────────────────────────

@pytest.mark.asyncio
@pytest.mark.unit
async def test_no_token_require_auth_false_returns_fallback():
    """No token + REQUIRE_AUTH=false → fallback user_id (demo mode)."""
    from langgraph_agents.db.session_store import _to_uuid
    import langgraph_agents.api.auth as auth_mod

    # Ensure flag is false
    original = auth_mod._REQUIRE_AUTH
    try:
        auth_mod._REQUIRE_AUTH = False
        request = _make_request()  # no Authorization header
        result = await auth_mod.resolve_user_id(request, "demo_user")
    finally:
        auth_mod._REQUIRE_AUTH = original

    assert result == _to_uuid("demo_user")


# ── Test 3: no token + REQUIRE_AUTH=true → 401 (IDOR close path) ──────────────

@pytest.mark.asyncio
@pytest.mark.unit
async def test_no_token_require_auth_true_raises_401():
    """REQUIRE_AUTH=true + no token → HTTPException 401. This closes the IDOR."""
    import langgraph_agents.api.auth as auth_mod

    original = auth_mod._REQUIRE_AUTH
    try:
        auth_mod._REQUIRE_AUTH = True
        request = _make_request()  # no Authorization header

        with pytest.raises(HTTPException) as exc_info:
            await auth_mod.resolve_user_id(request, "demo_user")
    finally:
        auth_mod._REQUIRE_AUTH = original

    assert exc_info.value.status_code == 401
    assert "authentication required" in exc_info.value.detail


# ── Test 4a: invalid token + REQUIRE_AUTH=true → 401 ─────────────────────────

@pytest.mark.asyncio
@pytest.mark.unit
async def test_invalid_token_require_auth_true_raises_401():
    """Expired/invalid token + REQUIRE_AUTH=true → 401 (not fallback)."""
    import langgraph_agents.api.auth as auth_mod

    original = auth_mod._REQUIRE_AUTH
    try:
        auth_mod._REQUIRE_AUTH = True
        with patch("langgraph_agents.api.auth._verify_token", return_value=None):
            request = _make_request("Bearer bad.token.here")
            with pytest.raises(HTTPException) as exc_info:
                await auth_mod.resolve_user_id(request, "fallback_user")
    finally:
        auth_mod._REQUIRE_AUTH = original

    assert exc_info.value.status_code == 401


# ── Test 4b: invalid token + REQUIRE_AUTH=false → fallback ────────────────────

@pytest.mark.asyncio
@pytest.mark.unit
async def test_invalid_token_require_auth_false_returns_fallback():
    """Expired/invalid token + REQUIRE_AUTH=false → demo fallback (graceful)."""
    from langgraph_agents.db.session_store import _to_uuid
    import langgraph_agents.api.auth as auth_mod

    original = auth_mod._REQUIRE_AUTH
    try:
        auth_mod._REQUIRE_AUTH = False
        with patch("langgraph_agents.api.auth._verify_token", return_value=None):
            request = _make_request("Bearer expired.token.here")
            result = await auth_mod.resolve_user_id(request, "demo_user_2")
    finally:
        auth_mod._REQUIRE_AUTH = original

    assert result == _to_uuid("demo_user_2")


# ── Test 5: token_use != "id" → rejected (returns None from _verify_token) ───

@pytest.mark.unit
def test_verify_token_rejects_wrong_token_use():
    """token_use != 'id' (e.g. 'access') must return None from _verify_token."""
    import langgraph_agents.api.auth as auth_mod

    fake_signing_key = MagicMock()
    fake_signing_key.key = "fake-key"

    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = fake_signing_key

    # jwt.decode returns claims with wrong token_use
    with patch.object(auth_mod, "_jwks_client", mock_jwks), \
         patch("jwt.decode", return_value={"sub": "some-sub", "token_use": "access"}):
        result = auth_mod._verify_token("fake.jwt.token")

    assert result is None


# ── Test 6: wrong audience/issuer → decode raises → rejected ─────────────────

@pytest.mark.unit
def test_verify_token_rejects_wrong_audience():
    """jwt.decode raising InvalidAudienceError should cause _verify_token → None."""
    import jwt as jwt_lib
    import langgraph_agents.api.auth as auth_mod

    fake_signing_key = MagicMock()
    fake_signing_key.key = "fake-key"

    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = fake_signing_key

    with patch.object(auth_mod, "_jwks_client", mock_jwks), \
         patch("jwt.decode", side_effect=jwt_lib.exceptions.InvalidAudienceError("bad audience")):
        result = auth_mod._verify_token("fake.jwt.token")

    assert result is None


@pytest.mark.unit
def test_verify_token_rejects_wrong_issuer():
    """jwt.decode raising InvalidIssuerError should cause _verify_token → None."""
    import jwt as jwt_lib
    import langgraph_agents.api.auth as auth_mod

    fake_signing_key = MagicMock()
    fake_signing_key.key = "fake-key"

    mock_jwks = MagicMock()
    mock_jwks.get_signing_key_from_jwt.return_value = fake_signing_key

    with patch.object(auth_mod, "_jwks_client", mock_jwks), \
         patch("jwt.decode", side_effect=jwt_lib.exceptions.InvalidIssuerError("bad issuer")):
        result = auth_mod._verify_token("fake.jwt.token")

    assert result is None
