"""SSE event encoding + EventSourceResponse helper.

Wraps sse-starlette for consistent event format across endpoints.
Frontend EventSource parses `event:` + `data:` fields automatically.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from sse_starlette.sse import EventSourceResponse


def encode_event(event_type: str, data: Any) -> dict:
    """Build a dict that sse-starlette serializes to:

        event: {event_type}
        data: {json.dumps(data)}

    """
    return {
        "event": event_type,
        "data": json.dumps(data, ensure_ascii=False),
    }


def stream_response(generator: AsyncIterator[dict]) -> EventSourceResponse:
    """Wrap an async generator of encoded events into a streaming response."""
    return EventSourceResponse(generator)
