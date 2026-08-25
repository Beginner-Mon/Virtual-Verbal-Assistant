"""Character catalog — local shim for zero-cost dev.

Mirrors infra/lambda/characters/handler.py but as a FastAPI router backed by
asyncpg (get_pg_client) instead of pg8000. Public (no auth) — same as the
prod REST API (rest_api_stack.py:208-211).

Prod stays on the Lambda (vva-characters) + REST API Gateway; this router
exists so `VITE_API_GATEWAY_URL=http://localhost:8000` serves /characters
without deploying anything, fixing MotionContext 404 on single-port dev.

Routes:
    GET /characters
    GET /characters/{slug}
    GET /characters/{slug}/avatar-profile

`persona` is never returned (LLM system prompt) — same as the Lambda.
"""

from __future__ import annotations

import re

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.api.characters")

router = APIRouter(tags=["characters"])

_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Same allowlist as infra/lambda/characters/handler.py — adding a column must
# not leak it by default (e.g. `persona`).
_PUBLIC_COLUMNS = (
    "slug, display_name, description, "
    "vrm_url, thumbnail_url, vrm_metadata, "
    "voice_language, sort_order, ui_strings"
)

_CACHE_SECONDS = 300


def _cache_headers() -> dict:
    return {"Cache-Control": f"public, max-age={_CACHE_SECONDS}"}


@router.get("/characters")
async def list_characters():
    """List active characters — public, no auth."""
    pg = get_pg_client()
    await pg.connect()
    # Bypass RLS user scope: characters is public catalog, not per-user.
    # get_pg_client().fetch goes through _scoped() which sets app.user_id
    # when bound; for public reads we use the raw transaction to avoid
    # requiring a token and to avoid RLS filtering.
    async with pg._raw_transaction() as conn:
        rows = await conn.fetch(
            f"SELECT {_PUBLIC_COLUMNS} FROM characters "
            "WHERE is_active ORDER BY sort_order, slug"
        )
    # asyncpg Record -> dict, decode JSONB strings (asyncpg returns str)
    characters = []
    for r in rows:
        d = dict(r)
        for k in ("vrm_metadata", "ui_strings"):
            if isinstance(d.get(k), str):
                try:
                    d[k] = json.loads(d[k])
                except Exception:
                    pass
        characters.append(d)
    return JSONResponse(
        content={"characters": characters, "total": len(characters)},
        headers=_cache_headers(),
    )


@router.get("/characters/{slug}")
async def get_character(slug: str):
    """One character by slug — public."""
    if not _SLUG_RE.match(slug):
        raise HTTPException(status_code=400, detail="Invalid character slug")
    pg = get_pg_client()
    await pg.connect()
    async with pg._raw_transaction() as conn:
        row = await conn.fetchrow(
            f"SELECT {_PUBLIC_COLUMNS}, avatar_profile FROM characters "
            "WHERE slug = $1 AND is_active",
            slug,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Character not found")
    result = dict(row)
    for k in ("vrm_metadata", "ui_strings", "avatar_profile"):
        if isinstance(result.get(k), str):
            try:
                result[k] = json.loads(result[k])
            except Exception:
                pass
    return JSONResponse(content=result, headers=_cache_headers())


@router.get("/characters/{slug}/avatar-profile")
async def get_avatar_profile(slug: str):
    """Avatar profile JSONB — public."""
    if not _SLUG_RE.match(slug):
        raise HTTPException(status_code=400, detail="Invalid character slug")
    pg = get_pg_client()
    await pg.connect()
    async with pg._raw_transaction() as conn:
        row = await conn.fetchrow(
            "SELECT avatar_profile FROM characters WHERE slug = $1 AND is_active",
            slug,
        )
    if row is None:
        raise HTTPException(status_code=404, detail="Character not found")
    profile = row["avatar_profile"] or {}
    if isinstance(profile, str):
        try:
            profile = json.loads(profile)
        except Exception:
            pass
    # handler.py returns the profile object directly, not wrapped
    if isinstance(profile, dict):
        return JSONResponse(content=profile, headers=_cache_headers())
    return JSONResponse(content={"avatar_profile": profile}, headers=_cache_headers())
