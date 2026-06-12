"""Lambda handler: GET /sessions — list user sessions.

Query params:
    user_id (required) — user identifier (UUID or plain string)
    limit   (optional) — max sessions to return (default: 50)

Ported from agenticRAG/langgraph_agents/api/main.py GET /sessions endpoint.
Changes from original:
    - No Redis cache
    - Uses normalized schema (separate messages table)
    - Runs on Lambda with pg8000 via RDS Proxy
"""

from __future__ import annotations

import logging

from shared.db import get_connection, to_uuid, fetch_all
from shared.response import success, error

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handler(event, context):
    params = event.get("queryStringParameters") or {}
    user_id = params.get("user_id")

    if not user_id:
        return error("user_id query parameter is required", 400)

    try:
        limit = min(int(params.get("limit", "50")), 200)
    except (ValueError, TypeError):
        limit = 50

    user_uuid = to_uuid(user_id)

    try:
        conn = get_connection()
        cur = conn.cursor()
        cur.execute(
            """
            SELECT c.session_id::text AS session_id,
                   c.created_at,
                   c.updated_at,
                   COUNT(m.id)::int AS message_count,
                   COALESCE(
                     SUBSTRING(
                       (SELECT content FROM messages
                        WHERE session_id = c.session_id AND role = 'user'
                        ORDER BY created_at ASC LIMIT 1
                       ), 1, 80
                     ),
                     '(empty)'
                   ) AS first_user_message_preview
            FROM conversations c
            LEFT JOIN messages m ON m.session_id = c.session_id
            WHERE c.user_id = %s::uuid
            GROUP BY c.session_id, c.created_at, c.updated_at
            ORDER BY c.updated_at DESC
            LIMIT %s
            """,
            (user_uuid, limit),
        )
        rows = fetch_all(cur)
        cur.close()

        return success({
            "sessions": rows,
            "total": len(rows),
        })

    except Exception as exc:
        logger.exception("list_sessions failed")
        return error(f"Internal server error: {exc}", 500)
