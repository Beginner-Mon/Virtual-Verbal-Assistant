"""Unit tests for api/auth.py — current_user_id, config, token verification.

The tests that used to live here for REQUIRE_AUTH=false (no token → fall back to
the client-supplied user_id) are gone along with the flag. What replaces them is
test_no_bypass_path_exists below, which asserts an absence rather than a
behaviour: the failure this file most needs to catch is somebody reintroducing a
way to be identified without a token.

Provider configuration is set by the auth_config fixture in THIS file, not in
conftest.py. auth.py reads it lazily, so only the tests that actually verify a
token need any, and the rest of the suite carries none.

No network: jwt.decode and the JWKS client are substituted throughout.
"""

from __future__ import annotations

import inspect
import uuid
from unittest.mock import MagicMock, patch

import pytest
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def auth_config(monkeypatch):
    """Fake, fixed provider config for this module only.

    Fake because no test here verifies a real signature — they substitute
    jwt.decode or the JWKS client — so pointing at a live user pool would only
    make the suite depend on the machine running it.

    The cache_clear calls are the load-bearing part: get_auth_config and
    get_jwks_client are lru_cached for the process, so without clearing them a
    value computed by an earlier test would outlive its environment.
    """
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


def _credentials(token: str) -> HTTPAuthorizationCredentials:
    return HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)


def _fake_jwks() -> MagicMock:
    signing_key = MagicMock()
    signing_key.key = "fake-key"
    client = MagicMock()
    client.get_signing_key_from_jwt.return_value = signing_key
    return client


# ── Configuration ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_importing_auth_needs_no_configuration():
    """Importing the module must not require, or read, provider config.

    The point of the lazy split. When this validated at import time, every test
    in the suite had to carry fake credentials to get past `import auth` — and
    conftest.py grew a block of them.
    """
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    agentic_root = Path(__file__).resolve().parents[2] / "agenticRAG"
    probe = textwrap.dedent(f"""
        import os, sys
        sys.path.insert(0, {str(agentic_root)!r})
        for var in ("AUTH_PROVIDER", "COGNITO_REGION", "COGNITO_USER_POOL_ID",
                    "COGNITO_APP_CLIENT_ID"):
            os.environ.pop(var, None)
        import langgraph_agents.api.auth  # must not raise
        print("IMPORT_OK")
    """)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=180,
    )
    assert "IMPORT_OK" in result.stdout, (
        f"importing auth.py without config failed:\n{result.stderr}"
    )


@pytest.mark.unit
def test_verify_auth_config_raises_when_unconfigured(auth_config, monkeypatch):
    """Startup must fail loudly on missing config, not 401 every request later."""
    monkeypatch.delenv("COGNITO_USER_POOL_ID", raising=False)
    auth_config.get_auth_config.cache_clear()

    with pytest.raises(ValueError, match="COGNITO_USER_POOL_ID"):
        auth_config.verify_auth_config()


@pytest.mark.unit
@pytest.mark.parametrize("provider", ["auth0", "clerk"])
def test_verify_auth_config_rejects_non_cognito_provider(
    auth_config, monkeypatch, provider,
):
    """Only cognito is supported, and "clerk" is the case that matters.

    A Clerk branch was removed on 19-08 because its frontend half had already
    been deleted — the backend would happily verify a token the UI had no way to
    obtain. `clerk` is parametrized here rather than left to the generic unknown-
    provider case so that re-adding the branch without re-adding the frontend
    turns this test red instead of silently reopening that path.
    """
    monkeypatch.setenv("AUTH_PROVIDER", provider)
    auth_config.get_auth_config.cache_clear()

    with pytest.raises(ValueError, match="cognito"):
        auth_config.verify_auth_config()


@pytest.mark.unit
def test_cognito_issuer_is_derived_from_the_pool(auth_config):
    """The issuer is what separates a sandbox pool from production."""
    config = auth_config.get_auth_config()
    assert config.issuer == "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_testpool"
    assert config.jwks_url == config.issuer + "/.well-known/jwks.json"
    assert config.audience == "test-client-id"


# ── The happy path ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.unit
async def test_valid_id_token_returns_sub(auth_config):
    from langgraph_agents.db.session_store import _to_uuid

    sub = str(uuid.uuid4())
    with patch.object(auth_config, "_verify_token", return_value={"sub": sub, "token_use": "id"}):
        result = await auth_config.current_user_id(_credentials("fake.jwt.token"))

    assert result == _to_uuid(sub)


# ── No identity without a token ───────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.unit
async def test_missing_credential_is_401(auth_config):
    """HTTPBearer yields None when there is no usable Authorization header."""
    with pytest.raises(HTTPException) as exc_info:
        await auth_config.current_user_id(None)

    assert exc_info.value.status_code == 401
    assert exc_info.value.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.asyncio
@pytest.mark.unit
async def test_invalid_token_is_401_not_a_fallback(auth_config):
    """A token that does not verify must 401 — never degrade to some other id."""
    with patch.object(auth_config, "_verify_token", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            await auth_config.current_user_id(_credentials("bad.token.here"))

    assert exc_info.value.status_code == 401


@pytest.mark.unit
@pytest.mark.parametrize("header", [
    None,                       # no Authorization header at all
    "",                         # present but empty
    "Basic dXNlcjpwYXNz",       # wrong scheme
    "Bearer",                   # scheme with no token
])
def test_malformed_authorization_header_is_401(auth_config, header):
    """End-to-end through the scheme, since header parsing is HTTPBearer's job."""
    app = FastAPI()

    @app.get("/probe")
    async def probe(uid: str = Depends(auth_config.current_user_id)):
        return {"uid": uid}

    # No `with`: the lifespan is irrelevant here and only costs a DB round trip.
    client = TestClient(app)
    headers = {} if header is None else {"Authorization": header}
    assert client.get("/probe", headers=headers).status_code == 401


@pytest.mark.unit
def test_openapi_declares_the_security_scheme(auth_config):
    """Routes must advertise that they need a credential.

    Not cosmetic: this is what puts the Authorize button in /docs, which is how
    the API gets tried from a browser now that there is no anonymous path.
    """
    app = FastAPI()

    @app.get("/probe")
    async def probe(uid: str = Depends(auth_config.current_user_id)):
        return {"uid": uid}

    schema = TestClient(app).get("/openapi.json").json()
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    assert schema["paths"]["/probe"]["get"]["security"]


# ── The absence test ──────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_bypass_path_exists(auth_config):
    """auth.py must expose no way to be identified without a verified token.

    Three assertions because there are three separate ways the bypass could come
    back: a flag, a caller-supplied fallback argument, or a resolver that quietly
    accepts one.
    """
    assert not hasattr(auth_config, "_REQUIRE_AUTH"), (
        "REQUIRE_AUTH is back. Environments are separated by trust root "
        "(which user pool), not by a switch that turns verification off."
    )

    assert not hasattr(auth_config, "resolve_user_id"), (
        "resolve_user_id is back. Use current_user_id, which cannot be handed a "
        "fallback identity."
    )

    params = list(inspect.signature(auth_config.current_user_id).parameters)
    assert params == ["credentials"], (
        f"current_user_id takes {params}. Its only input must be the injected "
        "credential — any extra parameter is a place for a client-supplied id "
        "to get in."
    )


# ── Token content checks ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_verify_token_rejects_wrong_token_use(auth_config):
    """token_use != 'id' (e.g. an access token) must return None."""
    with patch.object(auth_config, "get_jwks_client", return_value=_fake_jwks()), \
         patch("jwt.decode", return_value={"sub": "some-sub", "token_use": "access"}):
        assert auth_config._verify_token("fake.jwt.token") is None


@pytest.mark.unit
def test_verify_token_rejects_token_without_sub(auth_config):
    """No sub means no identity, even if the signature checks out."""
    with patch.object(auth_config, "get_jwks_client", return_value=_fake_jwks()), \
         patch("jwt.decode", return_value={"token_use": "id"}):
        assert auth_config._verify_token("fake.jwt.token") is None


@pytest.mark.unit
@pytest.mark.parametrize("error_name", ["InvalidAudienceError", "InvalidIssuerError",
                                        "ExpiredSignatureError", "InvalidSignatureError"])
def test_verify_token_rejects_bad_claims(auth_config, error_name):
    """Wrong audience, wrong issuer, expired, or forged → None.

    Wrong issuer is the one that separates the environments: a token minted by
    the `ampx sandbox` pool carries a different issuer, so production rejects it.
    """
    import jwt as jwt_lib

    error_cls = getattr(jwt_lib.exceptions, error_name)
    with patch.object(auth_config, "get_jwks_client", return_value=_fake_jwks()), \
         patch("jwt.decode", side_effect=error_cls("rejected")):
        assert auth_config._verify_token("fake.jwt.token") is None


@pytest.mark.unit
def test_jwks_outage_is_401_and_logged_as_unavailable(auth_config):
    """A JWKS failure fails closed, and is logged as an outage, not a bad token.

    The distinction is the point. Conflating "we could not reach Cognito" with
    "your token is invalid" is how an identity-provider outage presents as every
    user simultaneously holding a bad credential, with nothing in the logs
    pointing at the real cause.
    """
    from jwt import PyJWKClientError

    broken = MagicMock()
    broken.get_signing_key_from_jwt.side_effect = PyJWKClientError("cannot fetch keys")

    with patch.object(auth_config, "get_jwks_client", return_value=broken), \
         patch.object(auth_config.logger, "warning") as mock_warning:
        assert auth_config._verify_token("fake.jwt.token") is None

    assert mock_warning.called
    assert mock_warning.call_args[0][0] == "token_verification_unavailable"
