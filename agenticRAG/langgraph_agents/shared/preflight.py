"""Import-check the packages that are imported lazily, at start-up.

THE BUG THIS CLOSES

Twenty third-party imports in this package sit inside functions rather than at
module level — for real reasons (`sentence_transformers` costs seconds to load,
`youtube_transcript_api` is used by one tool, `langchain_google_genai` only runs
when the primary provider has failed). The cost is that a missing package is
invisible until the moment that function is first called.

The worst case is not hypothetical. On this machine `python` resolves to a
different interpreter than `pip`, so `langchain_google_genai` was installed for
one and absent from the other. The backend started fine, served traffic fine,
and would have raised ImportError the first time the Gemini fallback ran — which
is, by construction, the moment DeepSeek is already broken. Two outages for the
price of one, and the traceback points at the fallback rather than at the
install.

So: import them all once at start-up, where a missing one is cheap to see and
nothing is on fire yet. This does not make any of them non-lazy at runtime; the
modules stay in `sys.modules` afterwards, which is exactly what a later lazy
import wants anyway.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass

from langgraph_agents.shared.logging import get_logger

logger = get_logger("langgraph.preflight")


@dataclass(frozen=True)
class LazyDependency:
    module: str
    #: What stops working when it is missing. Written for whoever reads the log
    #: at 3am, not for whoever wrote the import.
    breaks: str
    #: False when the service is fully usable without it.
    critical: bool


# Every third-party module imported inside a function body. Keep in step with
# the code — `test_preflight.py` fails when a lazy import is not listed here.
LAZY_DEPENDENCIES: tuple[LazyDependency, ...] = (
    LazyDependency("asyncpg", "all database access", critical=True),
    LazyDependency("pgvector", "vector search over the knowledge base", critical=True),
    LazyDependency("yaml", "reading langgraph.yaml / mcp_servers.yaml", critical=True),
    LazyDependency("redis", "short-term memory and Celery results", critical=True),
    LazyDependency("langchain_core", "the graph itself", critical=True),
    LazyDependency("httpx", "calls to TTS and SearXNG", critical=True),
    LazyDependency("sentence_transformers", "embeddings — retrieval returns nothing", critical=True),
    LazyDependency("dotenv", "loading .env (real env vars still work)", critical=False),
    LazyDependency(
        "langchain_google_genai",
        "the Gemini fallback — the service looks healthy until DeepSeek fails, "
        "then fails a second time on the way out",
        critical=False,
    ),
    LazyDependency("youtube_transcript_api", "the YouTube transcript tool", critical=False),
)


@dataclass
class PreflightResult:
    missing_critical: list[LazyDependency]
    missing_optional: list[LazyDependency]

    @property
    def ok(self) -> bool:
        return not self.missing_critical and not self.missing_optional


def check_lazy_dependencies() -> PreflightResult:
    """Import each lazily-used package. Never raises."""
    missing_critical: list[LazyDependency] = []
    missing_optional: list[LazyDependency] = []

    for dep in LAZY_DEPENDENCIES:
        try:
            importlib.import_module(dep.module)
        except ImportError:
            (missing_critical if dep.critical else missing_optional).append(dep)
        except Exception as exc:  # a broken install, not an absent one
            logger.warning("preflight_import_error", extra={"module": dep.module, "error": str(exc)})

    return PreflightResult(missing_critical, missing_optional)


def log_preflight(result: PreflightResult) -> None:
    """Report the result, naming the interpreter.

    The interpreter path is the point. "No module named X" sends people to
    `pip install X`, which succeeds, after which the problem is unchanged
    because pip and python were never the same environment. Printing
    sys.executable turns a twenty-minute confusion into a five-second one.
    """
    if result.ok:
        logger.info(
            "preflight_ok",
            extra={"checked": len(LAZY_DEPENDENCIES), "interpreter": sys.executable},
        )
        return

    for dep in result.missing_critical:
        logger.error(
            "MISSING DEPENDENCY %r — %s. Interpreter: %s. "
            "Install into THIS interpreter: %s -m pip install -r requirements-langgraph.txt",
            dep.module, dep.breaks, sys.executable, sys.executable,
        )

    for dep in result.missing_optional:
        logger.warning(
            "MISSING OPTIONAL DEPENDENCY %r — %s. Interpreter: %s. "
            "Install into THIS interpreter: %s -m pip install -r requirements-langgraph.txt",
            dep.module, dep.breaks, sys.executable, sys.executable,
        )


def run_preflight() -> PreflightResult:
    result = check_lazy_dependencies()
    log_preflight(result)
    return result
