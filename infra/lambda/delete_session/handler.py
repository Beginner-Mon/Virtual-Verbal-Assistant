"""Lambda handler: DELETE /sessions/{session_id} — delete a session.

Path params:
    session_id (required) — session UUID

Query params:
    user_id (required) — ownership check

Ported from agenticRAG/langgraph_agents/api/main.py DELETE /sessions endpoint.
Changes from original:
    - No Redis STM cleanup (Lambda has no Redis access)
    - Messages auto-deleted via ON DELETE CASCADE on messages.session_id FK
    - user_id moved to query param for cleaner REST URL structure
"""

from __future__ import annotations

import logging

from shared.db import get_connection, to_uuid
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

    user_uuid = to_uuid(user_id)

    try:
        conn = get_connection()
        cur = conn.cursor()

        # Verify ownership then delete. ON DELETE CASCADE on messages table
        # ensures all message rows for this session are also removed.
        cur.execute(
            """
            DELETE FROM conversations
            WHERE session_id = %s::uuid AND user_id = %s::uuid
            RETURNING session_id::text
            """,
            (session_id, user_uuid),
        )
        deleted_row = cur.fetchone()
        cur.close()

        if deleted_row is None:
            return error("Session not found or not owned by user", 404)

        logger.info("session_deleted session_id=%s user_id=%s", session_id, user_id)

        return success({
            "deleted": session_id,
            "message": "Session and all associated messages deleted",
        })

    except Exception as exc:
        logger.exception("delete_session failed")
        return error(f"Internal server error: {exc}", 500)
