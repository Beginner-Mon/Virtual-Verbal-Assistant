"""User preferences — cross-device synced UI prefs (avatar_bg, default character).

GET  /me/preferences  → current user's synced prefs (UI-only, no PHI)
PATCH /me/preferences → merge patch with optimistic lock (version)

Identity never comes from the request. Both routes use Depends(current_user_id)
which reads it from the verified Cognito ID token (api/auth.py:207). No user_id
path/query/body parameter exists to spoof — this is the IDOR fix: /me means
"whoever the token says you are". Row-level security on user_preferences
(current_setting('app.user_id')) is the second fence.

prefs JSONB is UI-only (notifications, locale). 8KB max, depth <=2, reject
__proto__/constructor. PHI (injury_history etc) belongs in user_memory.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, Response

from langgraph_agents.api.auth import current_user_id
from langgraph_agents.api.schemas import UserPreferencesOut, UserPreferencesPatch
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.api.preferences")

router = APIRouter(tags=["preferences"])

# Keep in sync with avatarPalette.ts + 009 CHECK
ALLOWED_AVATAR_BG = {"slate", "violet", "blue", "emerald", "amber", "rose", "cyan", "indigo"}
MAX_PREFS_BYTES = 8 * 1024
MAX_PREFS_DEPTH = 2
FORBIDDEN_PREFS_KEYS = {"__proto__", "constructor", "prototype"}


def _validate_prefs(prefs: dict) -> None:
    """Reject oversized, too-deep, or prototype-polluting payloads."""
    raw = json.dumps(prefs, ensure_ascii=False)
    if len(raw.encode("utf-8")) > MAX_PREFS_BYTES:
        raise HTTPException(413, f"prefs too large (>{MAX_PREFS_BYTES} bytes)")

    def _check(obj, depth: int):
        if depth > MAX_PREFS_DEPTH:
            raise HTTPException(400, f"prefs too deep (>{MAX_PREFS_DEPTH})")
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in FORBIDDEN_PREFS_KEYS:
                    raise HTTPException(400, f"forbidden prefs key: {k}")
                _check(v, depth + 1)
        elif isinstance(obj, list):
            for v in obj:
                _check(v, depth + 1)

    _check(prefs, 0)
    # PHI guard — prefs must not smuggle health data
    _phi_keys = {"injury_history", "fitness_level", "age", "medical_history", "phi"}
    for k in prefs:
        if k in _phi_keys:
            raise HTTPException(400, f"prefs must not contain PHI key: {k} — use user_memory")


def _row_to_out(row) -> UserPreferencesOut:
    return UserPreferencesOut(
        avatar_bg=row["avatar_bg"],
        selected_character_slug=row["selected_character_slug"],
        display_name=row["display_name"],
        prefs=row["prefs"] if isinstance(row["prefs"], dict) else {},
        version=row["version"],
        updated_at=row["updated_at"].isoformat() if hasattr(row["updated_at"], "isoformat") else str(row["updated_at"]),
    )


@router.get("/me/preferences", response_model=UserPreferencesOut)
async def get_preferences(uid: str = Depends(current_user_id), response: Response = None):
    """Return the calling user's synced prefs. Creates a row on first read."""
    pg = get_pg_client()
    # Ensure row exists (users row is created by /me/memory or /chat; this is idempotent)
    async with pg.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING", uid,
        )
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1::uuid) ON CONFLICT (user_id) DO NOTHING",
            uid,
        )
        row = await conn.fetchrow(
            "SELECT user_id, avatar_bg, selected_character_slug, display_name, prefs, version, updated_at "
            "FROM user_preferences WHERE user_id = $1::uuid",
            uid,
        )
    if not row:
        raise HTTPException(500, "failed to load preferences")
    out = _row_to_out(row)
    if response is not None:
        response.headers["ETag"] = f'W/"{out.version}"'
    return out


@router.patch("/me/preferences", response_model=UserPreferencesOut)
async def patch_preferences(body: UserPreferencesPatch, uid: str = Depends(current_user_id), response: Response = None):
    """Merge-patch the calling user's prefs. Optimistic lock on version.

    avatar_bg / selected_character_slug / display_name are replaced when present.
    prefs is merged: stored prefs || patch.prefs (shallow JSONB concat).
    409 if version is stale — caller should GET then retry.
    """
    if body.prefs is not None:
        _validate_prefs(body.prefs)

    # Validate character slug when provided (and not clearing with None via missing field)
    # None in body means "not sent"; to clear, client sends null explicitly which
    # Pydantic maps to None but we need to distinguish. We treat None as "no change"
    # unless the client explicitly wants to clear — use a sentinel: if the field was
    # set to None explicitly, the model still has None, so we check model_fields_set.
    # For now, only set when non-None to avoid ambiguity.
    wants_clear_character = "selected_character_slug" in body.model_fields_set and body.selected_character_slug is None

    pg = get_pg_client()
    async with pg.transaction() as conn:
        await conn.execute(
            "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING", uid,
        )
        await conn.execute(
            "INSERT INTO user_preferences (user_id) VALUES ($1::uuid) ON CONFLICT (user_id) DO NOTHING",
            uid,
        )

        if body.selected_character_slug is not None:
            exists = await conn.fetchval(
                "SELECT 1 FROM characters WHERE slug = $1 AND is_active = true",
                body.selected_character_slug,
            )
            if not exists:
                raise HTTPException(400, f"unknown or inactive character: {body.selected_character_slug}")

        # Build dynamic update — prefs merged as JSONB ||, other cols COALESCE
        prefs_json = json.dumps(body.prefs) if body.prefs is not None else None

        # Use version in WHERE for optimistic lock
        if wants_clear_character:
            query = """
                UPDATE user_preferences
                SET selected_character_slug = NULL,
                    avatar_bg = COALESCE($2, avatar_bg),
                    display_name = COALESCE($3, display_name),
                    prefs = CASE WHEN $4::jsonb IS NULL THEN prefs ELSE prefs || $4::jsonb END,
                    version = version + 1,
                    updated_at = now()
                WHERE user_id = $1::uuid AND version = $5
                RETURNING user_id, avatar_bg, selected_character_slug, display_name, prefs, version, updated_at
            """
            row = await conn.fetchrow(query, uid, body.avatar_bg, body.display_name, prefs_json, body.version)
        else:
            # When selected_character_slug not sent, keep existing
            if body.selected_character_slug is not None:
                query = """
                    UPDATE user_preferences
                    SET avatar_bg = COALESCE($2, avatar_bg),
                        selected_character_slug = $3,
                        display_name = COALESCE($4, display_name),
                        prefs = CASE WHEN $5::jsonb IS NULL THEN prefs ELSE prefs || $5::jsonb END,
                        version = version + 1,
                        updated_at = now()
                    WHERE user_id = $1::uuid AND version = $6
                    RETURNING user_id, avatar_bg, selected_character_slug, display_name, prefs, version, updated_at
                """
                row = await conn.fetchrow(query, uid, body.avatar_bg, body.selected_character_slug, body.display_name, prefs_json, body.version)
            else:
                query = """
                    UPDATE user_preferences
                    SET avatar_bg = COALESCE($2, avatar_bg),
                        display_name = COALESCE($3, display_name),
                        prefs = CASE WHEN $4::jsonb IS NULL THEN prefs ELSE prefs || $4::jsonb END,
                        version = version + 1,
                        updated_at = now()
                    WHERE user_id = $1::uuid AND version = $5
                    RETURNING user_id, avatar_bg, selected_character_slug, display_name, prefs, version, updated_at
                """
                row = await conn.fetchrow(query, uid, body.avatar_bg, body.display_name, prefs_json, body.version)

        if not row:
            # Distinguish 409 vs missing row — if current version differs, it's a conflict
            current = await conn.fetchrow(
                "SELECT version FROM user_preferences WHERE user_id = $1::uuid", uid,
            )
            if current and current["version"] != body.version:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "version_conflict",
                        "current_version": current["version"],
                        "expected": body.version,
                    },
                )
            raise HTTPException(404, "preferences not found")

    out = _row_to_out(row)
    if response is not None:
        response.headers["ETag"] = f'W/"{out.version}"'
    logger.info("prefs_patched", extra={"user_id": uid, "version": out.version})
    return out
