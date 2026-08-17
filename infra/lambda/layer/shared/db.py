"""Database connection helper for VVA Lambda functions.

Supports the project's two infrastructure tracks, selected by DB_MODE:

    DB_MODE=rds   (default, Track 1 — frozen)
        Lambda inside the VPC → RDS Proxy, IAM auth, connection params from
        four plain SSM parameters under DB_PARAM_PREFIX. Auth token refreshed
        every ~14 min (lifetime 15 min).

    DB_MODE=neon  (Track 2 — active)
        Lambda outside any VPC → Neon over TLS, one SSM SecureString holding
        the whole DSN. No IAM auth, no RDS Proxy, no NAT gateway, and no
        interface VPC endpoints: a function outside the VPC reaches SSM over
        the ordinary internet path, so the endpoint costs that motivated
        Track 2 never arise.

One module, two branches, deliberately: forking the file would leave two
copies of the connection cache and the type serializers to keep in sync.

Environment variables:
    DB_MODE          — "rds" | "neon" (default "rds")
    DB_PARAM_PREFIX  — rds mode: SSM path prefix (e.g., /vva/db)
    NEON_DSN_PARAM   — neon mode: SSM SecureString name (e.g., /vva/neon/dsn)
"""

from __future__ import annotations

import os
import ssl
import time
import uuid
from datetime import datetime, date
from decimal import Decimal
from urllib.parse import urlsplit, unquote

import boto3
import pg8000.dbapi


# ── Module-level cache (persisted across warm Lambda invocations) ────

_conn: pg8000.dbapi.Connection | None = None
_config: dict | None = None
_token: str | None = None
_token_ts: float = 0.0

_TOKEN_REFRESH_SECS = 14 * 60  # Refresh token every 14 min (expires at 15)


def _mode() -> str:
    return os.environ.get("DB_MODE", "rds").strip().lower()


def _parse_dsn(dsn: str) -> dict:
    """Split a postgresql:// DSN into pg8000's separate connect kwargs.

    pg8000 takes host/port/database/user/password individually — it has no
    URL form — so a DSN copied from the Neon dashboard has to be taken apart.
    Credentials are percent-decoded: Neon passwords routinely contain
    characters that are escaped in a URL and must not reach the server that way.
    """
    parts = urlsplit(dsn)
    if parts.scheme not in ("postgresql", "postgres"):
        raise ValueError(f"unsupported DSN scheme: {parts.scheme!r}")

    database = parts.path.lstrip("/") or "neondb"
    return {
        "host": parts.hostname,
        "port": parts.port or 5432,
        "database": database,
        "user": unquote(parts.username or ""),
        "password": unquote(parts.password or ""),
    }


def _get_config() -> dict:
    """Read DB connection params from SSM Parameter Store (cached).

    rds mode — reads every parameter under DB_PARAM_PREFIX and keys them by
    the last path segment:
        /vva/db/proxy_endpoint → config["proxy_endpoint"]
        /vva/db/port           → config["port"]
        /vva/db/name           → config["name"]
        /vva/db/username       → config["username"]

    neon mode — reads one SecureString (WithDecryption) and splits the DSN.
    """
    global _config
    if _config is not None:
        return _config

    ssm = boto3.client("ssm")

    if _mode() == "neon":
        name = os.environ["NEON_DSN_PARAM"]
        resp = ssm.get_parameter(Name=name, WithDecryption=True)
        _config = _parse_dsn(resp["Parameter"]["Value"])
        return _config

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

    # TLS for both tracks. ssl.create_default_context() trusts the system CA
    # bundle, which covers the Amazon Root CA signing RDS certificates and the
    # public CA behind Neon alike — no custom bundle in either mode.
    ssl_context = ssl.create_default_context()

    if _mode() == "neon":
        _conn = pg8000.dbapi.connect(ssl_context=ssl_context, **config)
    else:
        token = _get_auth_token(config)
        _conn = pg8000.dbapi.connect(
            host=config["proxy_endpoint"],
            port=int(config.get("port", 5432)),
            database=config.get("name", "vva"),
            user=config["username"],
            password=token,             # IAM auth token instead of password
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
