"""JWT identity, from AWS Cognito.

Identity comes from the verified token and from nowhere else. There is no
environment variable, and no branch, that lets a caller name itself: a request
without a valid Bearer token is a 401 in every environment.

That is deliberate. The previous design carried a ``REQUIRE_AUTH`` flag whose
"false" setting made the client-supplied ``user_id`` the identity — an
auth-disabling code path shipped inside the production artifact, one wrong
environment variable away from being live, and silent when it was. Environments
are separated here by TRUST ROOT instead: development points
``COGNITO_USER_POOL_ID`` at an ``ampx sandbox`` pool, production points it at the
production pool. Same code on both sides, so what runs in development is what
runs in production, and a dev-pool token is rejected by production because the
issuer does not match.

Importing this module does nothing. Configuration is read and validated by
``verify_auth_config()``, which each app calls from its lifespan, and the JWKS
client is built on first use. That split matters: a module that validates
production configuration at import time forces every importer — tests included —
to satisfy production preconditions, which is how a test suite ends up carrying
fake credentials just to get past an import. Startup still fails loudly on bad
config (on Lambda, an INIT failure), because the lifespan runs at boot.

Tests substitute identity through FastAPI's ``dependency_overrides`` — a seam
that lives only inside the test process and cannot be switched on by
configuration:

    app.dependency_overrides[current_user_id] = lambda: "00000000-...-0001"

A Clerk branch lived here until 19-08. It was removed rather than documented:
the frontend half had already been deleted (see the note at the foot of
components/AuthGuard.tsx), so the only reachable configuration authenticated
against a provider the UI could no longer obtain a token from — and no
deployment ever set it, since infra/infra/crud_api_stack.py hardcodes
AUTH_PROVIDER=cognito. Dead code in an auth path is worse than dead code
elsewhere: it reads as a supported option.

Configuration:
    AUTH_PROVIDER   — must be "cognito" (default). Kept as an explicit check so a
                      leftover value fails at startup instead of being ignored.
    COGNITO_REGION, COGNITO_USER_POOL_ID, COGNITO_APP_CLIENT_ID — all required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

from langgraph_agents.db.postgres import bind_request_user
from langgraph_agents.db.session_store import _to_uuid
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.api.auth")

_UNAUTHENTICATED = HTTPException(
    status_code=401,
    detail="authentication required",
    headers={"WWW-Authenticate": "Bearer"},
)


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuthConfig:
    provider: str
    issuer: str
    jwks_url: str
    audience: Optional[str]


@lru_cache(maxsize=1)
def get_auth_config() -> AuthConfig:
    """Read and validate provider configuration. Cached for the process.

    Raises ValueError when the selected provider is not fully configured. There
    is no partially-configured mode: a service that cannot verify a token has
    nothing useful to do, and the failure belongs at startup rather than spread
    across every request as a 401 that looks like a user problem.
    """
    provider = os.getenv("AUTH_PROVIDER", "cognito").strip().lower()
    if provider != "cognito":
        raise ValueError(
            f"AUTH_PROVIDER must be 'cognito' (got {provider!r}). A Clerk branch "
            "used to live here and was removed on 19-08: the frontend half had "
            "already been deleted, so the backend could verify a Clerk token that "
            "the UI had no way to obtain. The variable is kept only so a stale "
            "value fails loudly instead of being ignored."
        )

    region = os.getenv("COGNITO_REGION", "")
    pool_id = os.getenv("COGNITO_USER_POOL_ID", "")
    client_id = os.getenv("COGNITO_APP_CLIENT_ID", "")
    if not all([region, pool_id, client_id]):
        raise ValueError(
            "AUTH_PROVIDER=cognito requires: COGNITO_REGION, COGNITO_USER_POOL_ID, "
            "COGNITO_APP_CLIENT_ID"
        )
    issuer = f"https://cognito-idp.{region}.amazonaws.com/{pool_id}"
    return AuthConfig(
        provider=provider,
        issuer=issuer,
        jwks_url=f"{issuer}/.well-known/jwks.json",
        audience=client_id,
    )


def verify_auth_config() -> None:
    """Fail now if this process cannot verify tokens. Call from app lifespan.

    Exists so the failure lands at startup — visible in a deploy, an INIT failure
    on Lambda — instead of arriving later as a 401 on every request, which is
    indistinguishable from users holding bad tokens.
    """
    config = get_auth_config()
    logger.info("auth_config_ok", extra={
        "provider": config.provider, "issuer": config.issuer,
    })


@lru_cache(maxsize=1)
def get_jwks_client() -> PyJWKClient:
    """The JWKS client, built on first use.

    PyJWKClient fetches the key set over HTTPS on the first verification and
    caches keys in-process afterwards. Outside a VPC this needs no NAT — same
    reasoning as infra/infra/character_stack.py.
    """
    return PyJWKClient(get_auth_config().jwks_url, cache_keys=True)


# ── Verification ──────────────────────────────────────────────────────────────


def _verify_token(token: str) -> Optional[dict]:
    """Verify the configured provider's session/ID token.

    Returns the claims, or None when the token is not acceptable. Both failure
    kinds return None — the caller 401s either way, which is the right answer
    when we cannot prove who is calling — but they are logged differently on
    purpose. See the except clauses.
    """
    try:
        config = get_auth_config()
        signing_key = get_jwks_client().get_signing_key_from_jwt(token)

        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=config.audience,
            issuer=config.issuer,
        )
        if claims.get("token_use") != "id":
            logger.debug("token_rejected", extra={"reason": "token_use_not_id"})
            return None

        if not claims.get("sub"):
            logger.debug("token_rejected", extra={"reason": "no_sub"})
            return None
        return claims

    except jwt.exceptions.InvalidTokenError as exc:
        # The token itself is bad — expired, wrong audience, wrong issuer, bad
        # signature. Routine, and the correct answer is simply "no".
        logger.debug("token_rejected", extra={"error": str(exc)})
        return None

    except Exception as exc:
        # Everything else means we could not REACH a verdict: the JWKS fetch
        # failed, the key set is malformed, the provider is misconfigured. The
        # response is still a 401 (fail closed), but this must not be logged as
        # "bad token" — that conflation is how a Cognito outage looks like every
        # user suddenly holding an invalid credential, with nothing in the logs
        # to say otherwise.
        logger.warning(
            "token_verification_unavailable",
            extra={"error": str(exc), "error_type": type(exc).__name__},
        )
        return None


# ── The dependency ────────────────────────────────────────────────────────────

# auto_error=False so a missing credential reaches our own handler: HTTPBearer's
# built-in error is a 403, and "you did not authenticate" is a 401.
#
# Public because billing.py declares the same parameter: a dependency that wraps
# current_user_id has to take credentials itself and pass them on, so it needs
# this exact scheme object. One instance, so the OpenAPI document describes one
# security scheme rather than two identical ones.
bearer_scheme = HTTPBearer(
    auto_error=False,
    description="Cognito ID token. Locally, mint one with scripts/dev_token.ps1.",
)


async def current_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> str:
    """FastAPI dependency: the calling user's id, from the verified token only.

    Use as ``uid: str = Depends(current_user_id)``. Never accept a user id from
    the request itself — a path, query or body value is input, not identity.

    Declared through HTTPBearer rather than by reading the header directly, so
    OpenAPI records that these routes need a credential: /docs then offers an
    Authorize button, which is the difference between being able to try the API
    from the browser and having to reach for curl.

    Raises:
        HTTPException(401): no Bearer token, or one that does not verify.
    """
    if credentials is None:
        raise _UNAUTHENTICATED

    claims = _verify_token(credentials.credentials)
    if claims is None:
        raise _UNAUTHENTICATED

    # Cognito's sub is already a UUID; _to_uuid is a no-op for those and keeps
    # the return type consistent for providers whose sub is not.
    user_id = _to_uuid(claims["sub"])

    # Row-level security reads the user from the database session, and this is
    # the only place that value is allowed to originate: it has just been taken
    # from a verified signature. Binding it here rather than in each handler
    # means a handler cannot forget, and cannot substitute something the caller
    # supplied. See db/postgres.py.
    bind_request_user(user_id)
    return user_id


def override_user(user_id: str):
    """Build a replacement for current_user_id, for dependency_overrides.

        app.dependency_overrides[current_user_id] = override_user("0000-...-0001")

    Use this rather than `lambda: uid`. Overriding the dependency replaces
    everything it does, including binding the user for row-level security — so a
    bare lambda produces a request that is authenticated as far as the handler is
    concerned and anonymous as far as the database is concerned. Every query then
    fails with `unrecognized configuration parameter "app.user_id"`, which points
    at the database rather than at the override that caused it.
    """
    async def _override() -> str:
        bind_request_user(user_id)
        return user_id

    return _override
