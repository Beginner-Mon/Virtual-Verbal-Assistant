"""API Gateway proxy response helpers for VVA Lambda functions.

Under a Lambda proxy integration the function must return
``{statusCode, headers, body}`` with **body as a JSON string**. AWS documents a
malformed proxy response as the number one cause of a 502 from API Gateway, so
the shape below is load-bearing: `json.dumps` here, not a dict.

CORS is also the function's job under proxy integration. API Gateway answers the
OPTIONS preflight from the stage's CORS configuration, but the headers on the
actual GET response come from here — the console's "Enable CORS" button does not
apply to proxy integrations.
"""

from __future__ import annotations

import json
import logging
import os

logger = logging.getLogger(__name__)

# The origins allowed to read these responses.
#
# This was `"*"` until 20-08, and the wildcard was invisible: CloudFront sat in
# front of /characters* with a ResponseHeadersPolicy whose origin_override=True
# replaced whatever the function sent. Moving the catalog behind API Gateway
# removed that cover and would have made the wildcard the real policy — a value
# nobody chose, that merely surfaced.
#
# Read from the environment rather than hardcoded so it matches the one list the
# rest of the service already uses (api/crud_app.py:39-47 does the same). A
# catalog is public data, so a wildcard would not have been a vulnerability; it
# would have been an inconsistency that outlives the reason for it.
_DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def allowed_origins() -> list[str]:
    return [
        o.strip()
        for o in os.getenv("ALLOWED_ORIGINS", _DEFAULT_ORIGINS).split(",")
        if o.strip()
    ] or [_DEFAULT_ORIGINS.split(",")[0]]


# The Origin of the request being served, set once per invocation by
# begin_request(). A module global rather than a parameter on success()/error():
# threading `event` through every call site would touch a dozen returns to carry
# one header, and a Lambda execution environment handles exactly one request at a
# time — the concurrency hazard that makes this pattern wrong in a web server
# does not exist here.
_request_origin: str | None = None


def begin_request(event: dict | None) -> None:
    """Record the caller's Origin. Call once, first thing in the handler.

    Not calling it is safe: responses then carry the first configured origin,
    which is wrong for the caller but is a legible wrongness — the browser
    console names an origin that does not match yours, rather than omitting the
    header and reading as a network fault.
    """
    global _request_origin
    headers = (event or {}).get("headers") or {}
    # Header names are case-insensitive and neither payload format normalises
    # them, so check both spellings.
    _request_origin = headers.get("origin") or headers.get("Origin")


def _headers() -> dict:
    allowed = allowed_origins()
    # Echo rather than list: the CORS spec permits exactly one origin here.
    origin = _request_origin if _request_origin in allowed else allowed[0]
    return {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Headers": "Content-Type,Authorization",
        "Access-Control-Allow-Methods": "GET,HEAD,OPTIONS",
        # Without this a shared cache could hand one origin's response to a
        # different origin: the body is identical and only this header differs.
        "Vary": "Origin",
    }


def success(body: dict, status_code: int = 200) -> dict:
    """Return a successful API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": _headers(),
        "body": json.dumps(body, ensure_ascii=False),
    }


def error(message: str, status_code: int = 400) -> dict:
    """Return an error API Gateway proxy response."""
    logger.warning("lambda_error status=%d message=%s", status_code, message)
    return {
        "statusCode": status_code,
        "headers": _headers(),
        "body": json.dumps({"error": message}, ensure_ascii=False),
    }
