"""The one place that loads `.env`.

WHY THIS MODULE EXISTS

There were four `.env` loaders — `llm.py`, `db/postgres.py`,
`local_tests/run_ollama_node_smoke.py` and `db/migrations/migrate_messages.py` —
each with its own search order, and one of them (the migration) looked only at
the repo root. Which file won therefore depended on **which module got imported
first**, and the losing case was silent: no `VVA_PG_DSN` meant the app quietly
fell back to the local Postgres container while migrations ran against Neon.
Nothing logged, nothing raised, and the two databases drift apart.

One loader, one canonical path, one warning when it is not there.

CANONICAL PATH: `agenticRAG/.env`

The legacy path `agenticRAG/agentic_rag_gemini/.env` is still accepted, and
loading it logs a warning telling you to move the file.

That package was deleted on 10/08/2026, so this looks like dead code — it is not.
`.env` is gitignored, and `git rm` does not touch ignored files: anyone pulling
that deletion keeps their old `.env` sitting in a directory git now considers
gone. Without this fallback their next start-up would silently drop to the local
database. The warning is how they find out where their config is coming from.

Environment variables already set in the process ALWAYS win: `override=False`
throughout. A container or a systemd unit passing real environment variables is
the correct production setup, and a stray `.env` on the image must not silently
replace it.
"""

from __future__ import annotations

import os
from pathlib import Path

from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.env")

# .../agenticRAG/langgraph_agents/shared/env.py → .../agenticRAG
AGENTIC_ROOT = Path(__file__).resolve().parents[2]

CANONICAL_ENV = AGENTIC_ROOT / ".env"
LEGACY_ENV = AGENTIC_ROOT / "agentic_rag_gemini" / ".env"

_loaded: Path | None = None
_done = False


def load_env(*, force: bool = False) -> Path | None:
    """Load the project `.env`. Idempotent; safe to call from any module.

    Returns the file that was loaded, or None when there was none — which is not
    an error. Docker, systemd and CI all supply configuration through real
    environment variables, and raising here would break precisely those correct
    setups. What is an error is *silently* running on defaults, and that is
    reported where it does damage: see `_resolve_dsn` in db/postgres.py, which
    now says so when it falls back to the local container.
    """
    global _loaded, _done
    if _done and not force:
        return _loaded

    _done = True

    try:
        from dotenv import load_dotenv
    except ImportError:
        logger.warning(
            "python-dotenv is not installed; .env files will be ignored. "
            "Configuration must come from real environment variables."
        )
        return None

    if CANONICAL_ENV.exists():
        load_dotenv(CANONICAL_ENV, override=False)
        _loaded = CANONICAL_ENV
        logger.info("loaded env from %s", CANONICAL_ENV)
        return _loaded

    if LEGACY_ENV.exists():
        load_dotenv(LEGACY_ENV, override=False)
        _loaded = LEGACY_ENV
        logger.warning(
            "Loaded configuration from the LEGACY path %s. "
            "Move it to %s — agentic_rag_gemini is the package this service "
            "replaced, and anything still reaching into it will break when it "
            "is deleted.",
            LEGACY_ENV,
            CANONICAL_ENV,
        )
        return _loaded

    logger.warning(
        "No .env found at %s (nor the legacy %s). Continuing with the process "
        "environment only — fine in a container, a mistake on a workstation.",
        CANONICAL_ENV,
        LEGACY_ENV,
    )
    return None


def env_source() -> str:
    """Human-readable description of where configuration came from.

    Used by the health endpoint: "which database am I actually on" is the first
    question during an incident, and "which file did you read" is half the
    answer.
    """
    if _loaded is None:
        return "process environment only"
    if _loaded == LEGACY_ENV:
        return f"{_loaded} (LEGACY — move to {CANONICAL_ENV})"
    return str(_loaded)


def require(name: str) -> str:
    """Read a variable that has no safe default, or fail with a usable message."""
    load_env()
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Add it to {CANONICAL_ENV} or export it in the "
            f"environment. (Configuration source: {env_source()})"
        )
    return value
