"""API Gateway proxy response helpers for VVA Lambda functions.

All Lambda functions behind API Gateway (proxy integration) must return
a dict with statusCode, headers, and body. These helpers standardize
the response format and include CORS headers.
"""

from __future__ import annotations

import json
import logging

logger = logging.getLogger(__name__)

# CORS headers — allow all origins for dev. Lock down in production.
_CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
    "Access-Control-Allow-Methods": "GET,DELETE,OPTIONS",
}


def success(body: dict, status_code: int = 200) -> dict:
    """Return a successful API Gateway proxy response."""
    return {
        "statusCode": status_code,
        "headers": _CORS_HEADERS,
        "body": json.dumps(body, ensure_ascii=False),
    }


def error(message: str, status_code: int = 400) -> dict:
    """Return an error API Gateway proxy response."""
    logger.warning("lambda_error status=%d message=%s", status_code, message)
    return {
        "statusCode": status_code,
        "headers": _CORS_HEADERS,
        "body": json.dumps({"error": message}, ensure_ascii=False),
    }
