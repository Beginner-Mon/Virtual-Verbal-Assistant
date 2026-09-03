"""The two /characters implementations must agree on what they return.

There are two of them and there is no way to make there be one: production runs
infra/lambda/characters/handler.py on pg8000, deployed as a CDK asset directory
that contains nothing else (character_stack.py), while
agenticRAG/langgraph_agents/api/routes_characters.py runs asyncpg inside the
FastAPI app so that a single-port local dev server can serve the catalog. Neither
can import the other.

They drifted the first time nobody was checking. 863458d trimmed the list
response from nine columns to five for the picker grid — and touched only the
FastAPI copy, which its own module docstring calls a local shim. Production kept
returning every column, so the optimisation that was measured and merged had no
effect at all where it was supposed to have one.

Reading the column lists out of the source is enough to catch that: the drift is
always a column added or removed on one side. These tests do not connect to a
database and do not import either module — the Lambda's dependencies (pg8000,
boto3) are built at deploy time and are not installed here.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LAMBDA = _ROOT / "infra" / "lambda" / "characters" / "handler.py"
_FASTAPI = _ROOT / "agenticRAG" / "langgraph_agents" / "api" / "routes_characters.py"


def _string_constants(path: Path) -> dict[str, str]:
    """Every module-level `NAME = "..."` in a file, without importing it.

    Implicit string concatenation across lines parses to one Constant, so the
    parenthesised form one file uses and the triple-quoted form the other uses
    both come back as plain strings.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            value = ast.literal_eval(node.value)
        except ValueError:
            continue
        if isinstance(value, str):
            found[target.id] = value
    return found


def _columns(source: str) -> set[str]:
    return {part.strip() for part in source.split(",") if part.strip()}


@pytest.fixture(scope="module")
def constants() -> tuple[dict[str, str], dict[str, str]]:
    return _string_constants(_LAMBDA), _string_constants(_FASTAPI)


@pytest.mark.unit
@pytest.mark.parametrize("name", ["_PUBLIC_COLUMNS", "_PUBLIC_COLUMNS_LITE"])
def test_both_implementations_declare_the_same_columns(constants, name):
    lambda_consts, fastapi_consts = constants

    assert name in lambda_consts, f"{_LAMBDA.name} no longer defines {name}"
    assert name in fastapi_consts, f"{_FASTAPI.name} no longer defines {name}"

    in_lambda = _columns(lambda_consts[name])
    in_fastapi = _columns(fastapi_consts[name])

    assert in_lambda == in_fastapi, (
        f"{name} has drifted between the two /characters implementations.\n"
        f"  only in the Lambda (production): {sorted(in_lambda - in_fastapi)}\n"
        f"  only in the FastAPI shim (dev):  {sorted(in_fastapi - in_lambda)}\n"
        "Both must be changed together — see this file's docstring."
    )


@pytest.mark.unit
def test_lite_is_a_strict_subset_of_full(constants):
    """The list response may narrow the detail response, never extend it."""
    for consts, path in zip(constants, (_LAMBDA, _FASTAPI)):
        lite = _columns(consts["_PUBLIC_COLUMNS_LITE"])
        full = _columns(consts["_PUBLIC_COLUMNS"])
        assert lite < full, (
            f"{path.name}: the lite column set must be a proper subset of the "
            f"full one; extra columns: {sorted(lite - full)}"
        )


@pytest.mark.unit
def test_lite_omits_the_columns_the_card_grid_does_not_use(constants):
    """What the narrowing was for.

    `vrm_url` points at a 9-17 MB model nobody downloads from a card, and
    `ui_strings` is the whole chat-surface copy for a character. Both are served
    by /characters/{slug} once one has been picked. `vrm_metadata` stays: the
    card greys itself out for models the device cannot run.
    """
    for consts, path in zip(constants, (_LAMBDA, _FASTAPI)):
        lite = _columns(consts["_PUBLIC_COLUMNS_LITE"])
        for column in ("vrm_url", "ui_strings", "voice_language"):
            assert column not in lite, f"{path.name}: {column} is back in the list response"
        assert "vrm_metadata" in lite, f"{path.name}: the card needs vrm_metadata to check compatibility"


@pytest.mark.unit
def test_persona_is_never_public():
    """persona is the LLM system prompt. It must not appear in either list."""
    for path in (_LAMBDA, _FASTAPI):
        consts = _string_constants(path)
        for name in ("_PUBLIC_COLUMNS", "_PUBLIC_COLUMNS_LITE"):
            assert "persona" not in _columns(consts[name]), (
                f"{path.name}: {name} exposes the system prompt"
            )
