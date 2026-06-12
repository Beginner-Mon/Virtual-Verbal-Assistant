"""Run local LLM smoke tests for individual LangGraph nodes.

This script is intentionally outside the normal pytest suite. It monkeypatches
node-level LLM factories to use an OpenAI-compatible endpoint (Ollama or Gemini)
and mocks DB/tool dependencies so each node can be exercised from a developer
shell.

Example:
    python -m langgraph_agents.local_tests.run_ollama_node_smoke --query "Bai tap cho dau lung"
    python -m langgraph_agents.local_tests.run_ollama_node_smoke --provider gemini
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Awaitable, Callable
from urllib.error import URLError
from urllib.parse import urlparse, urlunparse
from urllib.request import urlopen

# Allow direct execution from the repo root without setting PYTHONPATH:
#   python agenticRAG/langgraph_agents/local_tests/run_ollama_node_smoke.py ...
_THIS_FILE = Path(__file__).resolve()
_AGENTIC_ROOT = _THIS_FILE.parents[2]
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph_agents.state import ErrorSeverity


DEFAULT_QUERY = "Bai tap cho dau lung"
DEFAULT_OLLAMA_MODEL = "llama3.1:8b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_NODES = [
    "memory",
    "planner",
    "retriever_agent",
    "synthesizer",
    "grader",
    "error_handler",
]


def load_dotenv_once() -> None:
    """Best-effort load of repo env files used by the existing app."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    package_root = _THIS_FILE.parents[1]
    agentic_root = _AGENTIC_ROOT
    repo_root = _THIS_FILE.parents[3]
    for env_path in (
        package_root / ".env",
        agentic_root / ".env",
        agentic_root / "agentic_rag_gemini" / ".env",
        repo_root / ".env",
    ):
        if env_path.exists():
            load_dotenv(env_path, override=False)


@tool
async def pgvector_search(query: str, top_k: int = 5, source_type: str = "document") -> list[dict]:
    """Fake pgvector search for local node smoke tests."""
    return [
        {
            "content": (
                "For mild low back pain, physical therapy commonly starts with "
                "gentle mobility, core activation, and safety screening. Stop if "
                "pain radiates, weakness appears, or symptoms worsen."
            ),
            "similarity": 0.91,
            "source_type": source_type,
            "query": query,
            "top_k": top_k,
        }
    ]


@tool
async def generate_motion(description: str) -> dict:
    """Fake Kimodo-style motion synthesis result for local node smoke tests."""
    return {
        "motion_id": "local-smoke-motion",
        "status": "generated",
        "description": description,
        "joint_constraints": ["demo shoulder flexion <= 90 degrees"],
    }


@tool
async def search_medical(query: str) -> list[dict]:
    """Fake web medical search result for local node smoke tests."""
    return [
        {
            "title": "Local smoke medical result",
            "snippet": "Exercise advice should be individualized and stopped for red flags.",
            "url": "https://example.local/smoke",
            "query": query,
        }
    ]


def _ollama_tags_url(openai_base_url: str) -> str:
    parsed = urlparse(openai_base_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1"):
        path = path[:-3]
    return urlunparse((parsed.scheme, parsed.netloc, f"{path}/api/tags", "", "", ""))


def check_ollama(openai_base_url: str, model: str) -> dict[str, Any]:
    tags_url = _ollama_tags_url(openai_base_url)
    try:
        with urlopen(tags_url, timeout=3) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return {
            "ok": False,
            "tags_url": tags_url,
            "error": str(exc),
            "hint": f"Start Ollama and pull the model: ollama serve ; ollama pull {model}",
        }

    available = [m.get("name", "") for m in payload.get("models", [])]
    model_found = any(model == name or model in name or name in model for name in available)
    return {
        "ok": True,
        "tags_url": tags_url,
        "model_found": model_found,
        "available_models": available,
        "hint": None if model_found else f"Model not found. Run: ollama pull {model}",
    }


def resolve_gemini_api_key() -> tuple[str | None, str | None]:
    """Return a Gemini API key from common Google AI Studio env names."""
    import os

    for env_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        value = os.getenv(env_name, "").strip()
        if value:
            return value, env_name

    keys = [k.strip() for k in os.getenv("GEMINI_API_KEYS", "").split(",") if k.strip()]
    if keys:
        return keys[0], "GEMINI_API_KEYS[0]"

    return None, None


def check_gemini(model: str) -> dict[str, Any]:
    api_key, source = resolve_gemini_api_key()
    if not api_key:
        return {
            "ok": False,
            "model": model,
            "api_key_source": None,
            "hint": "Set GEMINI_API_KEY, GOOGLE_API_KEY, or GEMINI_API_KEYS in your .env.",
        }
    return {
        "ok": True,
        "model": model,
        "api_key_source": source,
        "hint": None,
    }


def make_openai_compatible_factory(
    *,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
) -> Callable[..., ChatOpenAI]:
    _ = provider
    cache: dict[tuple[str, float | None], ChatOpenAI] = {}

    def _factory(role: str, *, temperature: float | None = None) -> ChatOpenAI:
        cache_key = (role, temperature)
        if cache_key in cache:
            return cache[cache_key]

        temp = 0.0 if role in {"planner", "retriever", "health_check"} else 0.7
        if temperature is not None:
            temp = temperature
        model_client = ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=api_key,
            base_url=base_url,
            streaming=False,
        )
        cache[cache_key] = model_client
        return model_client

    return _factory


def precreate_node_clients(factory: Callable[..., ChatOpenAI], nodes: list[str]) -> None:
    """Create reusable chat clients before per-node timing starts."""
    roles = {
        "planner": "planner",
        "retriever_agent": "retriever",
        "synthesizer": "synthesizer",
        "conversation": "conversation",
    }
    for node in nodes:
        role = roles.get(node)
        if role:
            factory(role)


@contextmanager
def patched_llms(factory: Callable[..., ChatOpenAI], include_legacy_conversation: bool):
    from langgraph_agents.nodes import planner as planner_mod
    from langgraph_agents.nodes import retriever_agent as retriever_mod
    from langgraph_agents.nodes import synthesizer as synthesizer_mod

    patches: list[tuple[Any, str, Any]] = [
        (planner_mod, "get_chat_model", planner_mod.get_chat_model),
        (retriever_mod, "get_chat_model", retriever_mod.get_chat_model),
        (synthesizer_mod, "get_chat_model", synthesizer_mod.get_chat_model),
    ]
    if include_legacy_conversation:
        from langgraph_agents.nodes import conversation as conversation_mod

        patches.append((conversation_mod, "get_chat_model", conversation_mod.get_chat_model))

    try:
        for module, attr, _old in patches:
            setattr(module, attr, factory)
        yield
    finally:
        for module, attr, old in patches:
            setattr(module, attr, old)


@contextmanager
def patched_local_dependencies():
    from langgraph_agents.nodes import memory as memory_mod
    from langgraph_agents.nodes import retriever_agent as retriever_mod

    old_read_stm = memory_mod._read_stm
    old_lookup_ltm = memory_mod._lookup_ltm
    old_get_profile = memory_mod._get_user_profile
    old_build_tools = retriever_mod._build_tools

    async def _fake_read_stm(_session_id: str) -> list[dict]:
        return [
            {
                "q": "Lan truoc toi hoi bai tap nao?",
                "a": "Ban da hoi ve bai tap nhe cho dau lung.",
            }
        ]

    async def _fake_lookup_ltm(_user_id: str, _query: str) -> dict:
        return {"found": False, "skipped": True, "local_smoke": True}

    async def _fake_get_profile(_user_id: str) -> dict:
        return {"age": 35, "language": "vi", "local_smoke": True}

    async def _fake_build_tools() -> list:
        return [pgvector_search, generate_motion, search_medical]

    try:
        memory_mod._read_stm = _fake_read_stm
        memory_mod._lookup_ltm = _fake_lookup_ltm
        memory_mod._get_user_profile = _fake_get_profile
        retriever_mod._build_tools = _fake_build_tools
        yield
    finally:
        memory_mod._read_stm = old_read_stm
        memory_mod._lookup_ltm = old_lookup_ltm
        memory_mod._get_user_profile = old_get_profile
        retriever_mod._build_tools = old_build_tools


def make_config(args: argparse.Namespace) -> dict:
    return {
        "configurable": {
            "user_id": "00000000-0000-0000-0000-000000000001",
            "session_id": "local-smoke-session",
            "query": args.query,
            "persona_id": args.persona_id,
            "output_mode": "text",
            "request_id": "local-smoke",
            "token_limit": 0,
            "web_search": args.web_search,
        }
    }


def base_state() -> dict:
    return {
        "messages": [],
        "errors": [],
        "retry_count": 0,
        "total_tokens": 0,
        "memory_context": {
            "short_term": [],
            "long_term": {"found": False, "skipped": True},
            "user_profile": {},
        },
    }


def sample_plan(query: str) -> dict:
    return {
        "intent": "exercise_recommendation",
        "confidence": 0.9,
        "expanded_query": f"{query} physical therapy lower back pain exercises",
        "needs_clarification": False,
        "clarification_question": None,
        "required_outputs": ["exercise_name", "description", "sets_reps", "safety_warnings"],
        "search_strategy": ["pgvector_search"],
        "constraints_detected": ["stop if pain worsens"],
        "notes": "Local smoke-test sample plan.",
    }


def tool_message_for_query(query: str) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(
            [
                {
                    "content": (
                        "Local evidence: start with pelvic tilts, knee-to-chest, "
                        "and gentle bridge progressions. Use 2 sets of 8-10 reps "
                        "if symptoms stay mild."
                    ),
                    "similarity": 0.92,
                    "source_type": "document",
                    "query": query,
                }
            ],
            ensure_ascii=False,
        ),
        name="pgvector_search",
        tool_call_id="local-smoke-tool-call",
    )


async def run_memory(args: argparse.Namespace, context: dict) -> dict:
    from langgraph_agents.nodes.memory import memory_node

    out = await memory_node(base_state(), make_config(args))
    context["memory_context"] = out.get("memory_context", {})
    return out


async def run_planner(args: argparse.Namespace, context: dict) -> dict:
    from langgraph_agents.nodes.planner import planner_node

    state = base_state()
    state["memory_context"] = context.get("memory_context") or state["memory_context"]
    out = await planner_node(state, make_config(args))
    context.update(out)
    return out


async def run_retriever_agent(args: argparse.Namespace, context: dict) -> dict:
    from langgraph_agents.nodes.retriever_agent import retriever_agent_node

    plan = context.get("plan") or sample_plan(args.query)
    state = base_state()
    state.update(
        {
            "intent": context.get("intent", plan.get("intent", "exercise_recommendation")),
            "plan": plan,
            "expanded_query": context.get("expanded_query") or plan.get("expanded_query", args.query),
            "memory_context": context.get("memory_context") or base_state()["memory_context"],
            "grader_feedback": context.get("grader_feedback"),
        }
    )
    out = await retriever_agent_node(state, make_config(args))
    context["retriever_output"] = out
    context["messages"] = [*context.get("messages", []), *out.get("messages", [])]
    return out


async def run_synthesizer(args: argparse.Namespace, context: dict) -> dict:
    from langgraph_agents.nodes.synthesizer import synthesizer_node

    plan = context.get("plan") or sample_plan(args.query)
    state = base_state()
    state.update(
        {
            "intent": context.get("intent", plan.get("intent", "exercise_recommendation")),
            "confidence": context.get("confidence", plan.get("confidence", 0.9)),
            "expanded_query": context.get("expanded_query") or plan.get("expanded_query", args.query),
            "plan": plan,
            "needs_clarification": context.get("needs_clarification", False),
            "memory_context": context.get("memory_context") or base_state()["memory_context"],
            "messages": context.get("messages") or [tool_message_for_query(args.query)],
        }
    )
    out = await synthesizer_node(state, make_config(args))
    context.update(out)
    return out


async def run_grader(args: argparse.Namespace, context: dict) -> dict:
    from langgraph_agents.nodes.grader import grader_node

    plan = context.get("plan") or sample_plan(args.query)
    fallback_answer = (
        "Pelvic tilt: do 2 sets of 10 reps. Knee-to-chest stretch: hold 20 seconds, "
        "repeat 3 times. Stop if pain worsens."
    )
    state = base_state()
    state.update(
        {
            "intent": context.get("intent", plan.get("intent", "exercise_recommendation")),
            "reasoning_output": context.get("reasoning_output") or fallback_answer,
            "final_answer": context.get("final_answer") or fallback_answer,
            "messages": context.get("messages") or [tool_message_for_query(args.query)],
            "retry_count": context.get("retry_count", 0),
        }
    )
    out = await grader_node(state)
    context.update(out)
    return out


async def run_error_handler(_args: argparse.Namespace, _context: dict) -> dict:
    from langgraph_agents.nodes.error_handler import error_handler_node

    state = {
        "messages": [],
        "errors": [
            {
                "node": "local_smoke",
                "severity": ErrorSeverity.CRITICAL,
                "message": "Synthetic critical error for smoke test.",
            }
        ],
    }
    return await error_handler_node(state)


async def run_legacy_conversation(args: argparse.Namespace, context: dict) -> dict:
    from langgraph_agents.nodes.conversation import conversation_node

    state = base_state()
    state.update(
        {
            "intent": context.get("intent", "conversation"),
            "needs_clarification": False,
            "reasoning_output": context.get("reasoning_output")
            or "Pelvic tilt: do 2 sets of 10 reps. Stop if pain worsens.",
            "plan": context.get("plan") or sample_plan(args.query),
            "grader_warning": context.get("grader_warning"),
        }
    )
    return await conversation_node(state, make_config(args))


RUNNERS: dict[str, Callable[[argparse.Namespace, dict], Awaitable[dict]]] = {
    "memory": run_memory,
    "planner": run_planner,
    "retriever_agent": run_retriever_agent,
    "synthesizer": run_synthesizer,
    "grader": run_grader,
    "error_handler": run_error_handler,
    "conversation": run_legacy_conversation,
}


def simplify(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        data = {
            "type": value.__class__.__name__,
            "content": value.content,
        }
        if isinstance(value, AIMessage):
            data["tool_calls"] = value.tool_calls
        if isinstance(value, ToolMessage):
            data["name"] = value.name
            data["tool_call_id"] = value.tool_call_id
        return data
    if isinstance(value, dict):
        return {str(k): simplify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [simplify(v) for v in value]
    if hasattr(value, "model_dump"):
        return simplify(value.model_dump())
    if isinstance(value, ErrorSeverity):
        return value.value
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)


def print_result(name: str, result: dict, json_only: bool) -> None:
    if json_only:
        return
    status = result["status"]
    elapsed_ms = result["elapsed_ms"]
    print(f"\n[{name}] elapsed_ms={elapsed_ms} status={status}")
    if result.get("error"):
        print(f"error={result['error']}")
    print(json.dumps(result.get("output"), ensure_ascii=False, indent=2))


async def run_node(name: str, args: argparse.Namespace, context: dict) -> dict:
    start = time.perf_counter()
    try:
        out = await RUNNERS[name](args, context)
        return {
            "node": name,
            "status": "ok",
            "elapsed_ms": round((time.perf_counter() - start) * 1000),
            "output": simplify(out),
        }
    except Exception as exc:  # noqa: BLE001 - smoke runner should keep going.
        return {
            "node": name,
            "status": "error",
            "elapsed_ms": round((time.perf_counter() - start) * 1000),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "output": None,
        }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run standalone OpenAI-compatible smoke tests for langgraph_agents nodes."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--provider", choices=["ollama", "gemini"], default="ollama")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Model name. Defaults to llama3.1:8b for Ollama, or LLM_MODEL/"
            "gemini-2.5-flash for Gemini."
        ),
    )
    parser.add_argument("--ollama-base-url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--gemini-base-url", default=DEFAULT_GEMINI_BASE_URL)
    parser.add_argument("--nodes", nargs="+", default=DEFAULT_NODES, choices=sorted(RUNNERS))
    parser.add_argument("--persona-id", default="eca_default")
    parser.add_argument("--web-search", action="store_true")
    parser.add_argument("--include-legacy-conversation", action="store_true")
    parser.add_argument(
        "--no-precreate-clients",
        action="store_true",
        help="Include ChatOpenAI client construction inside the first node timing.",
    )
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON summary.")
    return parser.parse_args(argv)


async def amain(argv: list[str]) -> int:
    load_dotenv_once()
    args = parse_args(argv)
    import os

    if args.model is None:
        if args.provider == "gemini":
            args.model = os.getenv("LLM_MODEL", DEFAULT_GEMINI_MODEL)
        else:
            args.model = DEFAULT_OLLAMA_MODEL

    nodes = list(args.nodes)
    if args.include_legacy_conversation and "conversation" not in nodes:
        nodes.append("conversation")
    if "conversation" in nodes and not args.include_legacy_conversation:
        print(
            "Refusing to run legacy conversation unless --include-legacy-conversation is set.",
            file=sys.stderr,
        )
        return 2

    if args.provider == "gemini":
        health = check_gemini(args.model)
        base_url = args.gemini_base_url
        api_key, key_source = resolve_gemini_api_key()
        if not args.json:
            print(f"Provider: gemini")
            print(f"Gemini base URL: {base_url}")
            print(f"Gemini model: {args.model}")
            if not health["ok"]:
                print(health["hint"])
                return 1
            print(f"Gemini API key source: {key_source}")
        elif not health["ok"]:
            print(json.dumps({"provider": "gemini", "health": health, "results": []}, ensure_ascii=False, indent=2))
            return 1
    else:
        health = check_ollama(args.ollama_base_url, args.model)
        base_url = args.ollama_base_url
        api_key = "ollama"
        if not args.json:
            print(f"Provider: ollama")
            print(f"Ollama tags URL: {health['tags_url']}")
            if not health["ok"]:
                print(f"Ollama not reachable: {health['error']}")
                print(health["hint"])
                return 1
            if health.get("hint"):
                print(health["hint"])
            else:
                print(f"Ollama reachable. Using model: {args.model}")
        elif not health["ok"]:
            print(json.dumps({"provider": "ollama", "health": health, "results": []}, ensure_ascii=False, indent=2))
            return 1

    factory = make_openai_compatible_factory(
        provider=args.provider,
        model=args.model,
        base_url=base_url,
        api_key=api_key or "",
    )
    if not args.no_precreate_clients:
        precreate_node_clients(factory, nodes)

    context: dict[str, Any] = {}
    results: list[dict] = []

    with patched_llms(factory, args.include_legacy_conversation), patched_local_dependencies():
        for node in nodes:
            result = await run_node(node, args, context)
            results.append(result)
            print_result(node, result, args.json)

    if args.json:
        print(
            json.dumps(
                {
                    "provider": args.provider,
                    "health": health,
                    "query": args.query,
                    "model": args.model,
                    "base_url": base_url,
                    "results": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )

    return 0 if all(r["status"] == "ok" for r in results) else 1


def main() -> None:
    raise SystemExit(asyncio.run(amain(sys.argv[1:])))


if __name__ == "__main__":
    main()
