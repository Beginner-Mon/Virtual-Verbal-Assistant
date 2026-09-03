"""User preferences — cross-device synced UI prefs, one JSONB column on `users`.

GET   /me/preferences  → the calling user's synced prefs (UI-only, no PHI)
PATCH /me/preferences  → shallow merge over that column, last write wins

Identity never comes from the request. Both routes use Depends(current_user_id)
which reads it from the verified Cognito ID token (api/auth.py). No user_id
path/query/body parameter exists to spoof — this is the IDOR fix: /me means
"whoever the token says you are". Row-level security on `users`
(current_setting('app.user_id'), migration 007) is the second fence.

Three things this file no longer does, each on purpose:

*No validation of its own.* SyncedPrefs in api/schemas.py declares the whole set
of allowed keys and forbids the rest, so the size, depth, prototype-pollution and
PHI-keyword checks that used to live here have nothing left to catch.

*No writes on GET.* Preferences live on a row five other code paths already
create, so reading them creates nothing. A user with no row yet reads defaults.

*No version column.* Writes are last-write-wins. The merge is per key, so two
devices changing two different preferences never disagreed in the first place.

See docs/plans/preferences-v3-plan.md, ADR-008.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from langgraph_agents.api.auth import current_user_id
from langgraph_agents.api.schemas import (
    SyncedPrefs,
    UserPreferencesOut,
    UserPreferencesPatch,
)
from langgraph_agents.shared import get_pg_client
from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.api.preferences")

router = APIRouter(tags=["preferences"])

_SELECT = "SELECT preferences, updated_at FROM users WHERE id = $1::uuid"


def _read_prefs(raw) -> SyncedPrefs:
    """Parse a stored preferences object leniently.

    SyncedPrefs forbids unknown keys, which is what a write needs and the
    opposite of what a read needs: a row written before a key was renamed, or
    before a constraint tightened, would otherwise make GET fail for that one
    user and no other — the kind of outage that looks like an auth bug. Keys that
    no longer validate are dropped with a log line instead.

    asyncpg has no JSONB codec registered on this pool (db/postgres.py builds it
    with only a pgvector init), so JSONB arrives as text.
    """
    if isinstance(raw, str):
        raw = json.loads(raw or "{}")
    if not isinstance(raw, dict):
        return SyncedPrefs()

    safe: dict = {}
    for key, value in raw.items():
        try:
            SyncedPrefs.model_validate({key: value})
        except ValidationError:
            logger.warning("prefs_key_dropped", extra={"key": key})
            continue
        safe[key] = value
    return SyncedPrefs.model_validate(safe)


def _to_out(row) -> UserPreferencesOut:
    if row is None:
        return UserPreferencesOut()
    updated = row["updated_at"]
    return UserPreferencesOut(
        preferences=_read_prefs(row["preferences"]),
        updated_at=updated.isoformat() if hasattr(updated, "isoformat") else (
            str(updated) if updated is not None else None
        ),
    )


@router.get("/me/preferences", response_model=UserPreferencesOut)
async def get_preferences(uid: str = Depends(current_user_id)):
    """Return the calling user's synced prefs. Creates nothing."""
    pg = get_pg_client()
    async with pg.transaction() as conn:
        row = await conn.fetchrow(_SELECT, uid)
    return _to_out(row)


@router.patch("/me/preferences", response_model=UserPreferencesOut)
async def patch_preferences(
    body: UserPreferencesPatch, uid: str = Depends(current_user_id),
):
    """Merge the given keys into the calling user's prefs. Last write wins.

    Only keys present in the request are written — `exclude_unset` is what makes
    that true, and it is also what lets `selected_character_slug: null` mean
    "clear it" while an absent field means "leave it alone".
    """
    patch = body.preferences.model_dump(exclude_unset=True)

    pg = get_pg_client()
    async with pg.transaction() as conn:
        if not patch:
            return _to_out(await conn.fetchrow(_SELECT, uid))

        slug = patch.get("selected_character_slug")
        if slug is not None:
            # The only guard on this value. A foreign key would check that the
            # row exists and stop there; `characters` soft-deletes, so the case
            # that actually happens is a slug that exists and is switched off —
            # which would be stored happily, then 404 on GET /characters/{slug}
            # at next load and drop the user back to the default character while
            # the star in the picker still pointed elsewhere.
            active = await conn.fetchval(
                "SELECT 1 FROM characters WHERE slug = $1 AND is_active", slug,
            )
            if not active:
                raise HTTPException(400, f"unknown or inactive character: {slug}")

        # Same idiom as routes_crud.py and db/session_store.py: the row is
        # normally already there, and this costs nothing when it is.
        await conn.execute(
            "INSERT INTO users (id) VALUES ($1::uuid) ON CONFLICT (id) DO NOTHING", uid,
        )
        row = await conn.fetchrow(
            "UPDATE users SET preferences = preferences || $2::jsonb, updated_at = now() "
            "WHERE id = $1::uuid "
            "RETURNING preferences, updated_at",
            uid,
            json.dumps(patch),
        )

    if row is None:
        raise HTTPException(500, "failed to write preferences")

    logger.info("prefs_patched", extra={"user_id": uid, "keys": sorted(patch)})
    return _to_out(row)
