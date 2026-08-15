"""JWT auth resolution for Clerk Development or legacy Cognito.

Behaviour (controlled by REQUIRE_AUTH env var, default false):
  - Bearer token present + valid  → user_id = sub from JWT (client value ignored).
  - No/invalid token + REQUIRE_AUTH=false → fallback to client-supplied user_id.
  - No/invalid token + REQUIRE_AUTH=true  → HTTP 401.

Set ``AUTH_PROVIDER=clerk`` for the zero-cost development setup.  Cognito stays
available as a compatibility path for existing deployments while they migrate.
At startup, required provider configuration is validated when REQUIRE_AUTH=true.
"""

from __future__ import annotations

import os
from typing import Optional

import jwt
from fastapi import HTTPException, Request
from jwt import PyJWKClient, PyJWKClientError

from langgraph_agents.db.session_store import _to_uuid

# ── Config from env ────────────────────────────────────────────────────────────

_REQUIRE_AUTH: bool = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
_AUTH_PROVIDER: str = os.getenv("AUTH_PROVIDER", "cognito").strip().lower()
_REGION: str = os.getenv("COGNITO_REGION", "")
_POOL_ID: str = os.getenv("COGNITO_USER_POOL_ID", "")
_CLIENT_ID: str = os.getenv("COGNITO_APP_CLIENT_ID", "")
_CLERK_ISSUER: str = os.getenv("CLERK_ISSUER", "").rstrip("/")
_CLERK_JWKS_URL: str = os.getenv("CLERK_JWKS_URL", "")
_CLERK_AUDIENCE: str = os.getenv("CLERK_AUDIENCE", "")
_CLERK_AUTHORIZED_PARTIES: set[str] = {
    value.strip()
    for value in os.getenv("CLERK_AUTHORIZED_PARTIES", "").split(",")
    if value.strip()
}

# ── Startup guard ──────────────────────────────────────────────────────────────
# Fail at import time (module load = app startup) when REQUIRE_AUTH=true but
# Cognito config is incomplete — avoids silently falling back to no-auth.
if _AUTH_PROVIDER not in {"clerk", "cognito"}:
    raise ValueError("AUTH_PROVIDER must be 'clerk' or 'cognito'")

if _REQUIRE_AUTH:
    if _AUTH_PROVIDER == "clerk" and not _CLERK_ISSUER:
        raise ValueError("REQUIRE_AUTH=true with AUTH_PROVIDER=clerk requires CLERK_ISSUER")
    if _AUTH_PROVIDER == "cognito" and not all([_REGION, _POOL_ID, _CLIENT_ID]):
        raise ValueError(
            "REQUIRE_AUTH=true with AUTH_PROVIDER=cognito requires: "
            "COGNITO_REGION, COGNITO_USER_POOL_ID, COGNITO_APP_CLIENT_ID"
        )

# ── JWKS client (module-level cache; PyJWKClient caches keys internally) ──────
_jwks_client: Optional[PyJWKClient] = None

if _AUTH_PROVIDER == "clerk" and _CLERK_ISSUER:
    _JWKS_URL = _CLERK_JWKS_URL or f"{_CLERK_ISSUER}/.well-known/jwks.json"
    _ISSUER = _CLERK_ISSUER
    _jwks_client = PyJWKClient(_JWKS_URL, cache_keys=True)
elif _AUTH_PROVIDER == "cognito" and _REGION and _POOL_ID:
    _JWKS_URL = (
        f"https://cognito-idp.{_REGION}.amazonaws.com/{_POOL_ID}/.well-known/jwks.json"
    )
    _ISSUER = f"https://cognito-idp.{_REGION}.amazonaws.com/{_POOL_ID}"
    _jwks_client = PyJWKClient(_JWKS_URL, cache_keys=True)
else:
    _JWKS_URL = ""
    _ISSUER = ""


def _verify_token(token: str) -> Optional[dict]:
    """Verify the configured provider's session/ID token."""
    if _jwks_client is None:
        # No Cognito config — can't verify (only safe when REQUIRE_AUTH=false).
        return None
    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        if _AUTH_PROVIDER == "clerk":
            decode_kwargs = {
                "algorithms": ["RS256"],
                "issuer": _ISSUER,
                "options": {"verify_aud": bool(_CLERK_AUDIENCE)},
            }
            if _CLERK_AUDIENCE:
                decode_kwargs["audience"] = _CLERK_AUDIENCE
            claims = jwt.decode(token, signing_key.key, **decode_kwargs)
            if _CLERK_AUTHORIZED_PARTIES and claims.get("azp") not in _CLERK_AUTHORIZED_PARTIES:
                return None
        else:
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=_CLIENT_ID,
                issuer=_ISSUER,
            )
            if claims.get("token_use") != "id":
                return None
        if not claims.get("sub"):
            return None
        return claims
    except (
        jwt.exceptions.InvalidTokenError,
        PyJWKClientError,
        Exception,  # network errors, malformed JWKS, etc.
    ):
        return None


async def resolve_user_id(
    request: Request,
    fallback_user_id: Optional[str],
) -> str:
    """Extract + verify the Bearer token; return a UUID string for the caller.

    Args:
        request:          FastAPI Request (used to read the Authorization header).
        fallback_user_id: user_id supplied by the client (body / query / path).
                          Used only when there is no valid token AND
                          REQUIRE_AUTH=false.

    Returns:
        A UUID string (via _to_uuid) representing the authenticated or demo user.

    Raises:
        HTTPException(401): when token is absent/invalid AND REQUIRE_AUTH=true.
    """
    auth_header: str = request.headers.get("Authorization", "")

    if auth_header.startswith("Bearer "):
        token = auth_header[len("Bearer "):]
        claims = _verify_token(token)
        if claims is not None:
            # Valid token — sub is already a UUID from Cognito; _to_uuid is a
            # no-op for valid UUIDs but keeps the return type consistent.
            return _to_uuid(claims["sub"])
        # Token present but invalid — treat same as absent.

    # No valid token
    if _REQUIRE_AUTH:
        raise HTTPException(status_code=401, detail="authentication required")

    # Demo / dev fallback — honour client-supplied user_id
    return _to_uuid(fallback_user_id or "anonymous")
