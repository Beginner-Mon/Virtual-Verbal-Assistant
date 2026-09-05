"""No node may ship a prompt with a specific human language baked into it.

This is the backend counterpart of the frontend's `i18next/no-literal-string`
rule. Both exist for the same reason: cleaning the language mixing out once is a
morning's work, keeping it out is forever, and nothing else stops the next PR
from adding one more Vietnamese example to a prompt because it happened to be
convenient that day. That is exactly how the current state accumulated — nobody
chose it.

WHAT THIS ENFORCES

Instructions and few-shot examples sent to the model must be English and must
demonstrate STRUCTURE, not language. A model that can answer in Vietnamese can
also read a Vietnamese question against English examples; what it cannot do is
generalise to a third language from examples written in a second one. That was
Owner's point, and it is the whole reason this file exists.

WHAT IT DELIBERATELY ALLOWS

Comments and docstrings. They explain the code to whoever maintains it and never
reach the model, so `# cảnh báo dấu hiệu nguy hiểm` above a tag name is fine.
Only string literals that could travel into a prompt are checked.

WHAT IT CANNOT SEE

Persona text. `personas/*.md` is Vietnamese today and reaches the model through
`_persona_loader.build_voice_card`, placed LAST in the message list on purpose
(see its docstring). That is a real language leak and it is NOT covered here —
it is fixed by the persona overlay work, in a later slice. Do not read a green
run of this file as "the prompt has no Vietnamese in it".
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from langgraph_agents.shared.lang import _VN_EXCLUSIVE

_NODES_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "agenticRAG"
    / "langgraph_agents"
    / "nodes"
)

# The four nodes that build prompts. Listed rather than globbed: a new node
# should be added here deliberately, by someone who has read the rule above.
PROMPT_MODULES = [
    "planner.py",
    "retriever_agent.py",
    "synthesizer.py",
    "summarizer.py",
]

_VN_CHARS = frozenset(_VN_EXCLUSIVE + _VN_EXCLUSIVE.upper())


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Identity of every Constant that is a docstring, so it can be skipped.

    ast gives no direct "is this a docstring" flag — a docstring is just the
    first statement of a module, class or function when that statement is a bare
    string expression. Collecting them by identity is the only way to exclude
    them without also excluding a real string that happens to have the same text.
    """
    found: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            continue
        body = getattr(node, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                found.add(id(first.value))
    return found


def _offending_strings(path: pathlib.Path) -> list[str]:
    """Every non-docstring string literal in `path` carrying Vietnamese."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    skip = _docstring_nodes(tree)

    offenders: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        if id(node) in skip:
            continue
        hits = sorted({c for c in node.value if c in _VN_CHARS})
        if not hits:
            continue
        # Show the offending line, not the whole prompt — a system prompt is
        # 80 lines and dumping it hides the one word that failed.
        line = source.split("\n")[node.lineno - 1].strip()
        offenders.append(f"{path.name}:{node.lineno}: {line[:120]}   [{''.join(hits)}]")
    return offenders


@pytest.mark.unit
@pytest.mark.parametrize("module_name", PROMPT_MODULES)
def test_prompts_carry_no_vietnamese(module_name: str) -> None:
    path = _NODES_DIR / module_name
    assert path.is_file(), f"{module_name} moved — update PROMPT_MODULES"

    offenders = _offending_strings(path)

    assert not offenders, (
        f"\n{module_name} has {len(offenders)} string literal(s) with Vietnamese in them.\n"
        "Instructions and examples sent to the model must be English and must show\n"
        "STRUCTURE, not a language. Move nothing into a 'language pack' — rewrite it.\n\n"
        + "\n".join(offenders)
    )


@pytest.mark.unit
def test_the_detector_would_actually_catch_something() -> None:
    """Guard against the guard: a broken checker passes everything silently.

    If `_VN_EXCLUSIVE` were ever emptied, or the AST walk stopped finding
    Constants, every test above would go green while the codebase filled up with
    Vietnamese again. This proves the mechanism still fires.
    """
    sample = ast.parse('X = "bài tập cho đau lưng"')
    node = next(
        n
        for n in ast.walk(sample)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    )
    assert any(c in _VN_CHARS for c in node.value)
