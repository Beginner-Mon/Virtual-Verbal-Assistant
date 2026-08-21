"""Short-term memory store — one interface, three backends.

STM is a CACHE over PostgreSQL, not a source of truth. `messages` holds the real
conversation; this holds the last few turns so a request does not have to
re-read and re-shape them. Everything here is therefore allowed to fail: a miss
costs a database round-trip, never a wrong answer.

    stm = get_stm()
    recent = await stm.get(session_id)          # list | None
    await stm.set(session_id, pairs)            # with TTL
    await stm.delete(session_id)

Backends, chosen by ``STM_BACKEND``:

    redis     (default) — local development; docker-compose already runs one
    dynamodb            — deployed; no VPC, so no NAT gateway
    none                — no cache at all; every read is a miss

Two backends rather than one, for the same reason
``infra/lambda/layer/shared/db.py`` branches on ``DB_MODE``: one module with two
branches cannot drift the way two modules can.

Why not ElastiCache
-------------------
Measured 21-08: Serverless for Redis OSS meters a 1 GB minimum ($91/month);
Valkey meters 100 MB ($6/month). Either way it lives inside the VPC, and this
service needs the public internet (DeepSeek, Neon, Cognito JWKS), so joining the
VPC to reach it adds a NAT gateway at ~$33/month. **The NAT costs more than the
cache.** DynamoDB is reachable from outside a VPC and costs cents at this volume.

Why the shape below is enough
-----------------------------
The STM value is ONE small JSON blob per session — at most ``_STM_MAX`` (3) Q&A
pairs, read and written whole. Nothing here uses a Redis data structure: no
LIST, no SORTED SET, no pub/sub, no atomic operation. The only Redis feature in
play is ``SETEX``, i.e. a TTL, and DynamoDB has one natively. That is what makes
this a backend swap rather than a rewrite of the memory tier.

The DynamoDB TTL trap
---------------------
DynamoDB TTL is a background sweeper: AWS documents deletion "within a few days"
of expiry, typically within 48 hours — NOT at the timestamp. Ported naively, a
session's short-term memory that should have gone cold after two hours would
still be served two DAYS later, and the agent would answer using context it was
supposed to have forgotten. Silently: no error, no log, just a wrong answer.

So ``expires_at`` is written as a normal attribute and checked on every read,
and an item past its time is treated as absent. ``GetItem`` takes no filter
expression, so this has to happen in application code. DynamoDB's own TTL is
left enabled purely to reclaim storage — it decides nothing.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Optional

from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.shared.stm")

# 2 hours, matching the setex() this replaces.
DEFAULT_TTL_SECONDS = 7200

_DEFAULT_REDIS_URL = "redis://localhost:6379/0"
_DEFAULT_TABLE = "vva-stm"


def _key(session_id: str) -> str:
    """The cache key. Kept identical to the Redis key this replaces so that a
    running local Redis does not lose its contents when this module lands."""
    return f"stm:{session_id}"


class _Store:
    """Common behaviour: never raise, and never log the same outage twice.

    Both matter on the deployed path. Never raising is what lets the caller
    treat an unreachable cache as an ordinary miss — the PostgreSQL fallback in
    nodes/memory.py already handles that case. Logging once is what stops a
    dead backend from writing a line per request into CloudWatch, which is
    billed per GB ingested and would turn a degraded cache into a bill.
    """

    def __init__(self) -> None:
        self._reported: set[str] = set()

    def _degrade(self, op: str, exc: BaseException) -> None:
        if op not in self._reported:
            self._reported.add(op)
            logger.warning("stm_unavailable", extra={
                "op": op,
                "backend": type(self).__name__,
                "error": str(exc),
                "error_type": type(exc).__name__,
                "note": "further failures of this op are not logged; reads fall back to PostgreSQL",
            })

    async def get(self, session_id: str) -> Optional[list[dict]]:
        raise NotImplementedError

    async def set(self, session_id: str, value: list[dict],
                  ttl: int = DEFAULT_TTL_SECONDS) -> None:
        raise NotImplementedError

    async def delete(self, session_id: str) -> None:
        raise NotImplementedError


class NullStore(_Store):
    """No cache. Every read misses, every write is dropped.

    Not a test double — a supported deployment. It is what makes the Redis
    decision non-blocking: the agent runs correctly with no cache at all, just
    with an extra PostgreSQL read per turn, so the cache can be chosen after
    measuring rather than before.
    """

    async def get(self, session_id: str) -> Optional[list[dict]]:
        return None

    async def set(self, session_id: str, value: list[dict],
                  ttl: int = DEFAULT_TTL_SECONDS) -> None:
        return None

    async def delete(self, session_id: str) -> None:
        return None


class RedisStore(_Store):
    """Local development. One client for the process, as api/main.py already does.

    The four call sites this replaces each opened and closed their own
    connection — four connect/close cycles per turn against a cache whose whole
    point is to be cheaper than the database.
    """

    def __init__(self, url: str) -> None:
        super().__init__()
        self._url = url
        self._client = None

    def _redis(self):
        if self._client is None:
            import redis.asyncio as aioredis
            self._client = aioredis.from_url(
                self._url,
                decode_responses=True,
                # Short, and deliberately so: this is a cache. Waiting five
                # seconds to discover it is down costs more than the miss it is
                # trying to avoid.
                socket_connect_timeout=2,
                socket_timeout=2,
            )
        return self._client

    async def get(self, session_id: str) -> Optional[list[dict]]:
        try:
            raw = await self._redis().get(_key(session_id))
            return json.loads(raw) if raw else None
        except Exception as exc:                                    # noqa: BLE001
            self._degrade("get", exc)
            return None

    async def set(self, session_id: str, value: list[dict],
                  ttl: int = DEFAULT_TTL_SECONDS) -> None:
        try:
            await self._redis().setex(_key(session_id), ttl, json.dumps(value))
        except Exception as exc:                                    # noqa: BLE001
            self._degrade("set", exc)

    async def delete(self, session_id: str) -> None:
        try:
            await self._redis().delete(_key(session_id))
        except Exception as exc:                                    # noqa: BLE001
            self._degrade("delete", exc)


class DynamoStore(_Store):
    """Deployed path. Outside the VPC, so no NAT gateway.

    boto3 rather than an async client: it is already a dependency, the Lambda
    runtime ships it, and a single-item GetItem is a few milliseconds. The
    thread hop through asyncio.to_thread costs less than adding aioboto3 to an
    image whose size is already the thing being optimised.
    """

    def __init__(self, table_name: str) -> None:
        super().__init__()
        self._table_name = table_name
        self._table = None

    def _get_table(self):
        if self._table is None:
            import boto3
            self._table = boto3.resource("dynamodb").Table(self._table_name)
        return self._table

    async def get(self, session_id: str) -> Optional[list[dict]]:
        try:
            item = (await asyncio.to_thread(
                lambda: self._get_table().get_item(Key={"session_id": session_id})
            )).get("Item")
            if not item:
                return None

            # The reason this method is not a one-liner. See "The DynamoDB TTL
            # trap" in the module docstring: the sweeper runs days late, so an
            # expired item is still readable and would resurrect a conversation
            # that was supposed to be forgotten.
            expires_at = item.get("expires_at")
            if expires_at is not None and float(expires_at) <= time.time():
                return None

            payload = item.get("value")
            return json.loads(payload) if payload else None
        except Exception as exc:                                    # noqa: BLE001
            self._degrade("get", exc)
            return None

    async def set(self, session_id: str, value: list[dict],
                  ttl: int = DEFAULT_TTL_SECONDS) -> None:
        try:
            # Integer seconds: DynamoDB ignores a TTL attribute that is not a
            # Number, and does so without complaining — the item simply never
            # expires. Nothing surfaces that except a table that keeps growing.
            expires_at = int(time.time()) + int(ttl)
            await asyncio.to_thread(
                lambda: self._get_table().put_item(Item={
                    "session_id": session_id,
                    "value": json.dumps(value),
                    "expires_at": expires_at,
                })
            )
        except Exception as exc:                                    # noqa: BLE001
            self._degrade("set", exc)

    async def delete(self, session_id: str) -> None:
        try:
            await asyncio.to_thread(
                lambda: self._get_table().delete_item(Key={"session_id": session_id})
            )
        except Exception as exc:                                    # noqa: BLE001
            self._degrade("delete", exc)


_store: Optional[_Store] = None


def _build_store() -> _Store:
    backend = os.getenv("STM_BACKEND", "redis").strip().lower()

    if backend == "none":
        logger.info("stm_backend", extra={"backend": "none"})
        return NullStore()

    if backend == "dynamodb":
        table = os.getenv("STM_TABLE", _DEFAULT_TABLE)
        logger.info("stm_backend", extra={"backend": "dynamodb", "table": table})
        return DynamoStore(table)

    if backend == "redis":
        url = os.getenv("REDIS_URL", _DEFAULT_REDIS_URL)
        logger.info("stm_backend", extra={"backend": "redis", "url": url})
        return RedisStore(url)

    # Loud, and not survivable. A typo here would otherwise pick the default and
    # run for weeks with a cache nobody meant to use — or, worse on the deployed
    # path, try to reach a Redis on localhost that does not exist and degrade
    # every read while looking healthy.
    raise ValueError(
        f"STM_BACKEND={backend!r} is not a backend. Use 'redis' (default, local), "
        f"'dynamodb' (deployed), or 'none' (no cache)."
    )


def get_stm() -> _Store:
    """The process-wide STM store. Built on first use."""
    global _store
    if _store is None:
        _store = _build_store()
    return _store


def reset_stm() -> None:
    """Drop the cached store so the next get_stm() re-reads configuration.

    For tests that change STM_BACKEND. Production has one configuration for the
    life of the process and must not call this.
    """
    global _store
    _store = None
