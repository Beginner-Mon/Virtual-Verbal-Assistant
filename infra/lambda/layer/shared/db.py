"""Database connection helper for VVA Lambda functions.

Connects to RDS via RDS Proxy using IAM Authentication.
DB connection params are read from SSM Parameter Store at cold start.
Auth token is refreshed every ~14 minutes (token lifetime = 15 min).

Environment variables (set by CDK LambdaStack):
    DB_PARAM_PREFIX  — SSM parameter path prefix (e.g., /vva/db)
"""

from __future__ import annotations

import os
import ssl
import time
import uuid
from datetime import datetime, date
from decimal import Decimal

import boto3
import pg8000.dbapi


# ── Module-level cache (persisted across warm Lambda invocations) ────

_conn: pg8000.dbapi.Connection | None = None
_config: dict | None = None
_token: str | None = None
_token_ts: float = 0.0

_TOKEN_REFRESH_SECS = 14 * 60  # Refresh token every 14 min (expires at 15)


def _get_config() -> dict:
    """Read DB connection params from SSM Parameter Store (cached).

    Reads all parameters under DB_PARAM_PREFIX (e.g., /vva/db/) and
    returns them as a dict keyed by the last path segment:
        /vva/db/proxy_endpoint → config["proxy_endpoint"]
        /vva/db/port           → config["port"]
        /vva/db/name           → config["name"]
        /vva/db/username       → config["username"]
    """
    global _config
    if _config is not None:
        return _config

    ssm = boto3.client("ssm")
    prefix = os.environ["DB_PARAM_PREFIX"]
    resp = ssm.get_parameters_by_path(Path=prefix + "/", Recursive=False)

    _config = {}
    for param in resp["Parameters"]:
        # "/vva/db/proxy_endpoint" → "proxy_endpoint"
        key = param["Name"].rsplit("/", 1)[-1]
        _config[key] = param["Value"]

    return _config


def _get_auth_token(config: dict) -> str:
    """Generate a short-lived IAM auth token for RDS Proxy (cached ~14 min).

    The token is valid for 15 minutes. We refresh at 14 minutes to avoid
    connection failures near expiry boundaries.
    """
    global _token, _token_ts

    now = time.time()
    if _token is not None and (now - _token_ts) < _TOKEN_REFRESH_SECS:
        return _token

    rds_client = boto3.client("rds")
    _token = rds_client.generate_db_auth_token(
        DBHostname=config["proxy_endpoint"],
        Port=int(config.get("port", 5432)),
        DBUsername=config["username"],
        Region=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION")),
    )
    _token_ts = now
    return _token


def get_connection() -> pg8000.dbapi.Connection:
    """Return a reusable DB connection to RDS Proxy (IAM Auth + TLS).

    The connection is cached in module scope. If the cached connection is
    stale (Lambda container freeze/thaw or proxy idle timeout), a fresh
    connection is created transparently. Auth token is refreshed every ~14 min.
    """
    global _conn

    # Fast path: reuse existing connection
    if _conn is not None:
        try:
            # Lightweight liveness check — detects stale connections from
            # Lambda container freeze/thaw or RDS Proxy idle disconnect.
            cur = _conn.cursor()
            cur.execute("SELECT 1")
            cur.fetchone()
            cur.close()
            return _conn
        except Exception:
            # Connection dead — recreate below
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None

    config = _get_config()
    token = _get_auth_token(config)

    # TLS context for RDS Proxy (require_tls=True on proxy).
    # ssl.create_default_context() trusts the Amazon Root CA that signs
    # RDS certificates — no custom CA bundle needed in Lambda runtime.
    ssl_context = ssl.create_default_context()

    _conn = pg8000.dbapi.connect(
        host=config["proxy_endpoint"],
        port=int(config.get("port", 5432)),
        database=config.get("name", "vva"),
        user=config["username"],
        password=token,                 # IAM auth token instead of password
        ssl_context=ssl_context,
    )
    _conn.autocommit = True
    return _conn


# ── Query helpers ───────────────────────────────────────────────────


def to_uuid(value: str) -> str:
    """Coerce arbitrary string into a deterministic UUID string.

    Mirrors the logic in agenticRAG/langgraph_agents/db/session_store.py:
      - Already valid UUID → returned unchanged
      - Any other string   → uuid5(NAMESPACE_DNS, value) → deterministic
    """
    try:
        return str(uuid.UUID(value))
    except (ValueError, TypeError):
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(value)))


def _serialize_value(val):
    """Convert Python types to JSON-safe values."""
    if val is None:
        return val
    if isinstance(val, (datetime, date)):
        return val.isoformat()
    if isinstance(val, uuid.UUID):
        return str(val)
    if isinstance(val, Decimal):
        return int(val) if val == int(val) else float(val)
    return val


def fetch_all(cursor) -> list[dict]:
    """Fetch all rows from a cursor as a list of dicts with JSON-safe values."""
    if cursor.description is None:
        return []
    columns = [desc[0] for desc in cursor.description]
    return [
        {col: _serialize_value(row[i]) for i, col in enumerate(columns)}
        for row in cursor.fetchall()
    ]


def fetch_one(cursor) -> dict | None:
    """Fetch one row from a cursor as a dict with JSON-safe values."""
    if cursor.description is None:
        return None
    row = cursor.fetchone()
    if row is None:
        return None
    columns = [desc[0] for desc in cursor.description]
    return {col: _serialize_value(row[i]) for i, col in enumerate(columns)}
