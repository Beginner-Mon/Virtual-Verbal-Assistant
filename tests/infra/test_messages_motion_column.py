"""Guards on migration 008_messages_motion_job_id.py.

THIS FILE USED TO PASS FOR THE WRONG REASON. It string-matched
`infra/sql/init_schema.sql` — a file with no runner anywhere in the repo. The
assertion was green while the column did not exist in any database, and while
db/session_store.py had already begun SELECTing and INSERTing it
unconditionally. A green test is exactly what let that ship.

So the checks here are about the migration REACHING a database, not about a
string existing in a file:

  * it is an Alembic revision, in the directory alembic actually reads
  * it chains from the current head, with no sibling claiming the same parent
    (two children of one revision is a branch, and `alembic upgrade head` then
    refuses to run at all)
  * the statements it emits are the ones intended, captured by executing
    upgrade() against a fake `op` — the technique test_rls_policies.py uses,
    and for the same reason: reading the file as text also matches the prose.

Running it against a live Postgres is a separate, integration-marked concern;
there is no Neon connection in unit CI.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from unittest.mock import patch

import pytest

VERSIONS = (
    Path(__file__).resolve().parents[2]
    / "agenticRAG" / "langgraph_agents" / "alembic" / "versions"
)
MIGRATION = VERSIONS / "008_messages_motion_job_id.py"


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(f"mig_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _statements(direction: str = "upgrade") -> list[str]:
    module = _load(MIGRATION)
    captured: list[str] = []
    with patch.object(module.op, "execute", side_effect=captured.append):
        getattr(module, direction)()
    return captured


@pytest.mark.unit
def test_the_migration_lives_where_alembic_reads_from():
    """Not infra/sql/, not db/init_schema.sql. Alembic's versions/ is the only
    schema system in this repo with a runner."""
    assert MIGRATION.is_file()


@pytest.mark.unit
def test_it_chains_from_the_previous_head():
    assert _load(MIGRATION).down_revision == "007_rls"


@pytest.mark.unit
def test_nothing_else_claims_007_rls_as_its_parent():
    """Two children of one revision is a branch, and `alembic upgrade head`
    then fails with 'Multiple head revisions are present' — the migration would
    not run, which is the failure this whole file exists to prevent."""
    children = [
        p.name for p in VERSIONS.glob("*.py")
        if _load(p).down_revision == "007_rls"
    ]
    assert children == ["008_messages_motion_job_id.py"], children


@pytest.mark.unit
def test_revision_id_matches_the_filename():
    """alembic keys on the `revision` string, not the filename; a mismatch
    makes the chain unreadable to a human without fixing anything."""
    assert _load(MIGRATION).revision == MIGRATION.stem


@pytest.mark.unit
def test_upgrade_adds_the_column_the_session_store_selects():
    """db/session_store.py:185 SELECTs motion_job_id and :273 INSERTs it, both
    unconditionally. Without this column GET /sessions/{id} raises
    UndefinedColumn and write_session_turn fails into session_persist_failed —
    where it is swallowed, so chat answers and silently stops persisting."""
    sql = " ".join(_statements()).lower()
    assert re.search(
        r"alter table messages add column if not exists motion_job_id text", sql
    ), sql


@pytest.mark.unit
def test_it_is_additive_and_nullable():
    """Deploy order must not matter: the old Lambda ignores a column it does
    not select, the new one needs it before its first turn."""
    sql = " ".join(_statements()).lower()
    assert "not null" not in sql
    assert "drop column" not in sql


@pytest.mark.unit
def test_downgrade_removes_it():
    sql = " ".join(_statements("downgrade")).lower()
    assert "drop column if exists motion_job_id" in sql


@pytest.mark.unit
def test_one_statement_per_execute():
    """asyncpg rejects a batch with 'cannot insert multiple commands into a
    prepared statement'. Migration 004 never ran at all for this reason."""
    for stmt in _statements() + _statements("downgrade"):
        assert stmt.strip().rstrip(";").count(";") == 0, stmt


@pytest.mark.unit
def test_the_dead_sql_migration_is_gone():
    """infra/sql/migrations/002_messages_motion_job_id.sql had no runner, no
    001, and a number that collided with alembic's own 002_m4_fresh_schema.
    Leaving it there is an invitation to apply the wrong 002."""
    dead = Path(__file__).resolve().parents[2] / "infra" / "sql" / "migrations"
    assert not dead.exists(), f"{dead} should have been deleted"
