"""Lambda handler: GET /sessions/{session_id}/messages — load chat history.

Path params:
    session_id (required) — session UUID

Query params:
    user_id  (required) — ownership check
    cursor   (optional) — ISO-8601 timestamp for cursor-based pagination
    limit    (optional) — messages per page (default: 20, max: 100)

Ported from agenticRAG/langgraph_agents/api/main.py POST /sessions/{id}/resume.
Changes from original:
    - POST → GET (idempotent read-only)
    - No Redis STM populate (Lambda has no Redis access)
    - Cursor-based pagination instead of loading entire message history
    - Uses normalized messages table

Pagination pattern (matches REUPDATE_PLAN.md §6.10.1):
    First load: GET /sessions/{id}/messages?user_id=xxx
        → Returns 20 most recent messages (newest first)
        → Response includes next_cursor (created_at of oldest returned message)

    Scroll up: GET /sessions/{id}/messages?user_id=xxx&cursor=2026-05-28T10:30:00Z
        → Returns 20 messages older than cursor
        → next_cursor is null when no more messages exist
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from shared.db import get_connection, to_uuid, fetch_all
from shared.response import success, error

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    path_params = event.get("pathParameters") or {}
    query_params = event.get("queryStringParameters") or {}

    session_id = path_params.get("session_id")
    user_id = query_params.get("user_id")

    if not session_id:
        return error("session_id path parameter is required", 400)
    if not user_id:
        return error("user_id query parameter is required", 400)

    cursor = query_params.get("cursor")  # ISO-8601 timestamp or None

    try:
        limit = min(int(query_params.get("limit", "20")), 100)
    except (ValueError, TypeError):
        limit = 20

    user_uuid = to_uuid(user_id)

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Verify session ownership
        cur.execute(
            """
            SELECT session_id::text FROM conversations
            WHERE session_id = %s::uuid AND user_id = %s::uuid
            """,
            (session_id, user_uuid),
        )
        session_row = cur.fetchone()
        if session_row is None:
            cur.close()
            return error("Session not found or not owned by user", 404)

        # Cursor-based pagination: load messages older than cursor.
        # Uses idx_messages_session_pagination index (session_id, created_at DESC).
        # Fetch limit+1 to check if more messages exist beyond this page.
        if cursor:
            cur.execute(
                """
                SELECT id::text, role, content, metadata, created_at
                FROM messages
                WHERE session_id = %s::uuid
                  AND created_at < %s::timestamptz
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, cursor, limit + 1),
            )
        else:
            # First load: most recent messages
            cur.execute(
                """
                SELECT id::text, role, content, metadata, created_at
                FROM messages
                WHERE session_id = %s::uuid
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (session_id, limit + 1),
            )

        rows = fetch_all(cur)
        cur.close()

        # Determine if there are more messages (we fetched limit+1)
        has_more = len(rows) > limit
        if has_more:
            rows = rows[:limit]

        # next_cursor = created_at of the oldest message in this page
        next_cursor = rows[-1]["created_at"] if rows and has_more else None

        return success({
            "session_id": session_id,
            "messages": rows,
            "next_cursor": next_cursor,
            "has_more": has_more,
        })

    except Exception as exc:
        logger.exception("resume_session failed")
        return error(f"Internal server error: {exc}", 500)
