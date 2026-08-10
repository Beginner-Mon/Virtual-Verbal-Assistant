"""Start-up import check for lazily-imported packages.

The bug: on this machine `pip` and `python` resolved to different interpreters,
so `langchain_google_genai` was installed for one and missing from the other.
Because its import sits inside the fallback function, the backend started
normally and would only have raised the first time DeepSeek failed — turning one
outage into two, with a traceback pointing at the fallback rather than at the
install.
"""

import ast
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from langgraph_agents.shared import preflight

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "agenticRAG" / "langgraph_agents"


@pytest.mark.unit
def test_every_lazy_import_is_listed():
    """The list must track the code, or the check silently stops covering things.

    Parses every function body for third-party imports and compares against
    LAZY_DEPENDENCIES. This is the test that makes the whole module hold: a
    preflight naming nine of ten lazy imports gives exactly as much false
    confidence as no preflight at all.
    """
    stdlib = set(sys.stdlib_module_names)
    listed = {dep.module for dep in preflight.LAZY_DEPENDENCIES}
    found: dict[str, str] = {}

    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for func in ast.walk(tree):
            if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for node in ast.walk(func):
                if isinstance(node, ast.Import):
                    modules = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    modules = [node.module]
                else:
                    continue
                for module in modules:
                    root = module.split(".")[0]
                    if root in stdlib or root == "langgraph_agents":
                        continue
                    found.setdefault(root, f"{path.relative_to(PACKAGE_ROOT).as_posix()}:{node.lineno}")

    unlisted = {m: where for m, where in found.items() if m not in listed}
    assert not unlisted, (
        "Lazily imported but absent from preflight.LAZY_DEPENDENCIES:\n  "
        + "\n  ".join(f"{m} ({where})" for m, where in sorted(unlisted.items()))
    )


@pytest.mark.unit
def test_listed_dependencies_still_exist_in_code():
    """The reverse drift: an entry left behind after its import was removed."""
    source = "\n".join(
        p.read_text(encoding="utf-8-sig") for p in PACKAGE_ROOT.rglob("*.py")
    )
    stale = [d.module for d in preflight.LAZY_DEPENDENCIES if d.module not in source]
    assert not stale, f"listed in LAZY_DEPENDENCIES but no longer imported anywhere: {stale}"


@pytest.mark.unit
def test_all_present_in_this_environment():
    """Doubles as an environment check: this suite runs on the service's deps."""
    result = preflight.check_lazy_dependencies()
    assert result.ok, (
        f"missing: {[d.module for d in result.missing_critical + result.missing_optional]} "
        f"(interpreter: {sys.executable})"
    )


@pytest.mark.unit
def test_missing_optional_is_reported_but_never_raises():
    def explode(name):
        if name == "langchain_google_genai":
            raise ImportError("No module named 'langchain_google_genai'")
        return object()

    with patch.object(preflight.importlib, "import_module", side_effect=explode):
        result = preflight.run_preflight()

    assert [d.module for d in result.missing_optional] == ["langchain_google_genai"]
    assert result.missing_critical == []
    assert result.ok is False


@pytest.mark.unit
def test_missing_critical_is_separated_from_optional():
    def explode(name):
        if name in {"asyncpg", "youtube_transcript_api"}:
            raise ImportError(name)
        return object()

    with patch.object(preflight.importlib, "import_module", side_effect=explode):
        result = preflight.check_lazy_dependencies()

    assert [d.module for d in result.missing_critical] == ["asyncpg"]
    assert [d.module for d in result.missing_optional] == ["youtube_transcript_api"]


@pytest.mark.unit
def test_log_names_the_interpreter(caplog):
    """The interpreter path is the whole point of the message.

    "No module named X" sends people to `pip install X`, which succeeds and
    changes nothing when pip and python are different environments.
    """
    def explode(name):
        if name == "langchain_google_genai":
            raise ImportError(name)
        return object()

    with patch.object(preflight.importlib, "import_module", side_effect=explode):
        with caplog.at_level("WARNING"):
            preflight.run_preflight()

    assert any(sys.executable in record.getMessage() for record in caplog.records)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_health_reports_missing_dependency_as_degraded_not_down():
    """A missing optional package must not take the instance out of the pool."""
    from langgraph_agents.api import health

    def explode(name):
        if name == "langchain_google_genai":
            raise ImportError(name)
        return object()

    with patch.object(preflight.importlib, "import_module", side_effect=explode):
        result = await health.check_dependencies()

    assert result.ok is False
    assert "langchain_google_genai" in (result.detail or "")
    assert health._is_critical("dependencies") is False
