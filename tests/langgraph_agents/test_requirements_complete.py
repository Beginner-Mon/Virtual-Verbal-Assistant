"""Every third-party import in langgraph_agents must be declared.

WHY THIS EXISTS

`requirements-langgraph.txt` was missing five packages the service imports —
langchain-google-genai, sentence-transformers, numpy, PyYAML, SQLAlchemy. The
backend ran anyway, because `agentic_rag_gemini/requirements.txt` had been
installed into the same environment and supplied them. So the file was wrong for
months with nothing to say so: it only breaks for someone installing from it
alone, which is exactly what a fresh container or a new machine does.

Checking imports at runtime would not have caught it either — the environment
that runs the tests is the same one that has the extra packages. The check has
to be STATIC: parse the source, compare against the declared list. That holds
regardless of which interpreter or virtualenv the suite happens to run under,
and it fails on the machine that adds the import rather than on the machine that
deploys it.
"""

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "agenticRAG" / "langgraph_agents"
REQUIREMENTS = REPO_ROOT / "requirements-langgraph.txt"

# Import name → distribution name, for the cases where they differ. Everything
# else is matched by normalising `_`/`-`/case, which covers the large majority.
IMPORT_TO_DISTRIBUTION = {
    "yaml": "pyyaml",
    "jwt": "pyjwt",
    "dotenv": "python-dotenv",
}

# Modules that are genuinely not ours to declare.
IGNORED = {
    # First-party.
    "langgraph_agents",
    # Also first-party: text-to-motion/kimodo/vva_motion — the DynamoDB job
    # queue shared between kimodo_node (here) and the GPU worker. Reached via
    # the `pythonpath` entry in pytest.ini, not pip-installed, so it has no
    # distribution name to declare. Deliberately dependency-light (boto3 +
    # stdlib only, per its own module docstring) so it can be COPY'd into a
    # GPU worker image that has no langgraph_agents.
    "vva_motion",
    # The legacy package. Any import of it is a coupling bug, and
    # test_no_legacy_imports below is what fails then — not this test, which
    # would report the confusing "add agentic_rag_gemini to requirements".
    "agentic_rag_gemini",
}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_distributions() -> set[str]:
    """Package names in requirements-langgraph.txt, normalised.

    Strips version specifiers and extras: `PyJWT[crypto]>=2.8.0` → `pyjwt`.
    """
    declared = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = re.split(r"[<>=!~\[;]", line, maxsplit=1)[0].strip()
        if name:
            declared.add(_normalise(name))
    return declared


def _imported_modules() -> dict[str, list[str]]:
    """Top-level third-party module → the files that import it."""
    stdlib = set(sys.stdlib_module_names)
    imports: dict[str, list[str]] = {}

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        except (SyntaxError, UnicodeDecodeError):  # pragma: no cover
            pytest.fail(f"Could not parse {path}")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # level > 0 is a relative import — first-party by construction.
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue

            for name in names:
                root = name.split(".")[0]
                if root in stdlib or root in IGNORED:
                    continue
                imports.setdefault(root, []).append(
                    str(path.relative_to(REPO_ROOT)).replace("\\", "/")
                )

    return imports


@pytest.mark.unit
def test_every_import_is_declared():
    declared = _declared_distributions()
    missing: list[str] = []

    for module, files in sorted(_imported_modules().items()):
        distribution = IMPORT_TO_DISTRIBUTION.get(module, module)
        if _normalise(distribution) not in declared:
            missing.append(f"  {module!r} (imported by {files[0]}, +{len(files) - 1} more)")

    assert not missing, (
        "Imported but not declared in requirements-langgraph.txt:\n"
        + "\n".join(missing)
        + "\n\nAdd them, or add an entry to IMPORT_TO_DISTRIBUTION if the "
        "distribution is named differently from the module."
    )


@pytest.mark.unit
def test_no_legacy_imports():
    """langgraph_agents must not import the package it replaced.

    Kept separate from the test above so the failure names the real problem.
    Coupling back to `agentic_rag_gemini` is not something to fix by adding a
    requirement.
    """
    offenders: list[str] = []

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module] if node.level == 0 and node.module else []
            else:
                continue
            if any(n.split(".")[0] == "agentic_rag_gemini" for n in names):
                offenders.append(str(path.relative_to(REPO_ROOT)).replace("\\", "/"))

    assert not offenders, (
        "langgraph_agents imports the legacy agentic_rag_gemini package:\n  "
        + "\n  ".join(sorted(set(offenders)))
    )


@pytest.mark.unit
def test_requirements_file_parses_to_something_sane():
    """A guard on the guard: an empty or unreadable list would pass everything."""
    declared = _declared_distributions()
    assert len(declared) > 15, f"only parsed {len(declared)} requirements — parser broken?"
    for expected in ("langgraph", "asyncpg", "fastapi"):
        assert expected in declared
