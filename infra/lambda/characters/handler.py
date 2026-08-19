"""Lambda handler: the character catalog, served behind CloudFront.

Routes (Lambda Function URL, so the path arrives in event["rawPath"]):

    GET /characters                          list active characters
    GET /characters/{slug}                   one character
    GET /characters/{slug}/avatar-profile    just the avatar_profile JSONB

One function rather than three: these are three reads of the same resource,
and CloudFront caches the responses at the edge, so the origin is hit about
once per TTL. Splitting them would triple the CDK surface to serve a catalog
of four rows that changes approximately never.

`persona` is never returned. It is the LLM system prompt — the backend reads
it straight from the DB at startup (nodes/_persona_loader.preload_personas_from_db)
and no client has any use for it.
"""

from __future__ import annotations

import logging
import re

from shared.db import get_connection, fetch_all, fetch_one
from shared.response import success as _success, error

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Browser cache lifetime for catalog responses. Deliberately set here and not in
# shared/response.py: that helper is shared with the session CRUD handlers, whose
# responses are per-user and must never be stored anywhere. Only this catalog is
# public and identical for every viewer.
#
# CloudFront caches at the edge regardless (its cache policy decides that). What
# this adds is the browser's own cache, which turns a page reload from five
# requests into zero. Matched to the distribution's 300s TTL so the two layers
# cannot disagree about how stale a response may be.
_CACHE_SECONDS = 300


def success(body: dict, status_code: int = 200) -> dict:
    resp = _success(body, status_code)
    resp["headers"] = {**resp["headers"], "Cache-Control": f"public, max-age={_CACHE_SECONDS}"}
    return resp

# Same shape the backend enforces on ChatRequest.persona_id, because a slug is
# exactly what the frontend sends back as persona_id once a character is picked.
_SLUG_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

# Every column the client is allowed to see. Listed explicitly rather than
# SELECT * so that adding a column to `characters` can never leak it by default.
#
# `ui_strings` is the character's chat-surface copy — greeting, stage labels,
# error lines. It is deliberately a separate column from `persona`: persona is
# the system prompt and must never reach a browser, while this exists only to be
# rendered in one. Keep prompt text out of it.
_PUBLIC_COLUMNS = """
    slug, display_name, description,
    vrm_url, thumbnail_url, vrm_metadata,
    voice_language, sort_order, ui_strings
"""


def _list_characters(cur) -> dict:
    cur.execute(
        f"SELECT {_PUBLIC_COLUMNS} FROM characters "
        "WHERE is_active ORDER BY sort_order, slug"
    )
    rows = fetch_all(cur)
    return success({"characters": rows, "total": len(rows)})


def _get_character(cur, slug: str) -> dict:
    cur.execute(
        f"SELECT {_PUBLIC_COLUMNS}, avatar_profile FROM characters "
        "WHERE slug = %s AND is_active",
        (slug,),
    )
    row = fetch_one(cur)
    if row is None:
        return error("Character not found", 404)
    return success(row)


def _get_avatar_profile(cur, slug: str) -> dict:
    cur.execute(
        "SELECT avatar_profile FROM characters WHERE slug = %s AND is_active",
        (slug,),
    )
    row = fetch_one(cur)
    if row is None:
        return error("Character not found", 404)
    return success(row["avatar_profile"] or {})


def handler(event, context):
    method = (
        event.get("requestContext", {})
        .get("http", {})
        .get("method", "GET")
        .upper()
    )
    if method not in ("GET", "HEAD"):
        return error("Method not allowed", 405)

    # Trailing slashes are stripped so /characters and /characters/ are one route.
    raw_path = (event.get("rawPath") or "/characters").rstrip("/") or "/characters"
    segments = [s for s in raw_path.split("/") if s]

    # segments is one of:
    #   ["characters"]                            → list
    #   ["characters", slug]                      → detail
    #   ["characters", slug, "avatar-profile"]     → profile
    if not segments or segments[0] != "characters":
        return error(f"Unknown path: {raw_path}", 404)

    slug = segments[1] if len(segments) > 1 else None
    if slug is not None and not _SLUG_RE.match(slug):
        # Rejected before it reaches SQL. The query is parameterised anyway, so
        # this is about returning an honest 400 rather than an empty 404.
        return error("Invalid character slug", 400)

    if len(segments) > 3 or (len(segments) == 3 and segments[2] != "avatar-profile"):
        return error(f"Unknown path: {raw_path}", 404)

    try:
        conn = get_connection()
        cur = conn.cursor()
        try:
            if slug is None:
                return _list_characters(cur)
            if len(segments) == 3:
                return _get_avatar_profile(cur, slug)
            return _get_character(cur, slug)
        finally:
            cur.close()

    except Exception as exc:
        logger.exception("characters handler failed path=%s", raw_path)
        return error(f"Internal server error: {exc}", 500)
