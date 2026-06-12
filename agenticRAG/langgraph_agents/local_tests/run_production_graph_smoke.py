"""Run the production LangGraph pipeline with local smoke-test dependencies.

This runner is closer to the FastAPI production path than the per-node smoke
runner: it builds the compiled graph from graph.py, streams it with
graph.astream(..., stream_mode=["updates", "custom"]), and reports elapsed time
per graph node based on update events.

It still monkeypatches LLM/provider and external dependencies so it can run from
a developer shell without DeepSeek, Redis, Postgres, or MCP services.

Example:
    python agenticRAG/langgraph_agents/local_tests/run_production_graph_smoke.py --provider gemini --query "Bai tap cho dau lung"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

_THIS_FILE = Path(__file__).resolve()
_AGENTIC_ROOT = _THIS_FILE.parents[2]
if str(_AGENTIC_ROOT) not in sys.path:
    sys.path.insert(0, str(_AGENTIC_ROOT))

from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from langgraph_agents.local_tests.run_ollama_node_smoke import (
    DEFAULT_GEMINI_BASE_URL,
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_QUERY,
    check_gemini,
    check_ollama,
    load_dotenv_once,
    make_openai_compatible_factory,
    make_config,
    patched_llms,
    patched_local_dependencies,
    precreate_node_clients,
    pgvector_search,
    resolve_gemini_api_key,
    simplify,
)


GRAPH_NODE_ORDER = [
    "memory",
    "planner",
    "retriever_agent",
    "tools",
    "synthesizer",
    "grader",
    "error_handler",
]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the production LangGraph pipeline with smoke-test dependencies."
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
    parser.add_argument("--persona-id", default="eca_default")
    parser.add_argument("--web-search", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print a machine-readable JSON summary.")
    parser.add_argument(
        "--no-precreate-clients",
        action="store_true",
        help="Include ChatOpenAI client construction inside graph timing.",
    )
    return parser.parse_args(argv)


def production_initial_state() -> dict:
    """Mirror FastAPI /chat initial state."""
    return {"messages": [], "errors": [], "retry_count": 0, "total_tokens": 0}


async def build_smoke_graph() -> Any:
    """Build production graph while disabling real MCP discovery and DB tools."""
    from langgraph_agents import graph as graph_mod

    old_mcp_module = sys.modules.get("langgraph_agents.mcp.client")
    old_retriever_tools = graph_mod.RETRIEVER_TOOLS

    async def _fake_get_mcp_tools() -> list:
        return []

    fake_mcp_client_mod = types.ModuleType("langgraph_agents.mcp.client")
    fake_mcp_client_mod.get_mcp_tools = _fake_get_mcp_tools
    fake_mcp_client_mod.close_mcp_client = lambda: None

    # build_graph_async imports langgraph_agents.mcp.client inside the function.
    # Installing this fake module avoids requiring langchain_mcp_adapters in a
    # local smoke-test environment.
    sys.modules["langgraph_agents.mcp.client"] = fake_mcp_client_mod
    graph_mod.RETRIEVER_TOOLS = [pgvector_search]
    try:
        return await graph_mod.build_graph_async()
    finally:
        graph_mod.RETRIEVER_TOOLS = old_retriever_tools
        if old_mcp_module is None:
            sys.modules.pop("langgraph_agents.mcp.client", None)
        else:
            sys.modules["langgraph_agents.mcp.client"] = old_mcp_module


def summarize_output(output: dict) -> dict:
    summary = simplify(output)
    if "messages" in summary:
        summary["messages"] = summarize_messages(summary["messages"])
    return summary


def summarize_messages(messages: Any) -> Any:
    if not isinstance(messages, list):
        return messages
    summarized = []
    for message in messages:
        if not isinstance(message, dict):
            summarized.append(message)
            continue
        item = {
            "type": message.get("type"),
            "content_preview": str(message.get("content", ""))[:300],
        }
        if message.get("tool_calls"):
            item["tool_calls"] = message["tool_calls"]
        if message.get("name"):
            item["name"] = message["name"]
        summarized.append(item)
    return summarized


def simplify_final_state(state: dict) -> dict:
    return {
        "intent": state.get("intent", ""),
        "needs_clarification": state.get("needs_clarification", False),
        "grader_result": state.get("grader_result", ""),
        "retry_count": state.get("retry_count", 0),
        "total_tokens": state.get("total_tokens", 0),
        "final_answer": state.get("final_answer", ""),
        "errors": simplify(state.get("errors", [])),
        "messages": summarize_messages(simplify(state.get("messages", []))),
    }


async def run_graph(args: argparse.Namespace) -> dict:
    factory = make_openai_compatible_factory(
        provider=args.provider,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
    )
    if not args.no_precreate_clients:
        precreate_node_clients(factory, GRAPH_NODE_ORDER)

    timings: dict[str, list[int]] = {}
    node_events: list[dict] = []
    custom_tokens: list[str] = []
    final_state: dict = {}
    graph_start = time.perf_counter()
    previous_update_at = graph_start

    with patched_llms(factory, include_legacy_conversation=False), patched_local_dependencies():
        graph = await build_smoke_graph()
        async for mode, payload in graph.astream(
            production_initial_state(),
            make_config(args),
            stream_mode=["updates", "custom"],
        ):
            now = time.perf_counter()
            if mode == "updates" and isinstance(payload, dict):
                for node_name, node_output in payload.items():
                    elapsed_ms = round((now - previous_update_at) * 1000)
                    timings.setdefault(node_name, []).append(elapsed_ms)
                    previous_update_at = now
                    summary = summarize_output(node_output) if isinstance(node_output, dict) else simplify(node_output)
                    node_events.append(
                        {
                            "node": node_name,
                            "elapsed_since_previous_update_ms": elapsed_ms,
                            "output": summary,
                        }
                    )
                    if isinstance(node_output, dict):
                        final_state.update(node_output)
            elif mode == "custom" and isinstance(payload, dict) and "content" in payload:
                custom_tokens.append(payload["content"])

    total_elapsed_ms = round((time.perf_counter() - graph_start) * 1000)
    return {
        "total_elapsed_ms": total_elapsed_ms,
        "node_timings_ms": timings,
        "node_events": node_events,
        "streamed_text": "".join(custom_tokens),
        "final_state": simplify_final_state(final_state),
    }


def print_human(result: dict) -> None:
    print(f"\n[graph] total_elapsed_ms={result['total_elapsed_ms']}")
    print("\n[node timings]")
    for node_name, timings in result["node_timings_ms"].items():
        for idx, elapsed_ms in enumerate(timings, 1):
            suffix = f"#{idx}" if len(timings) > 1 else ""
            print(f"- {node_name}{suffix}: {elapsed_ms}ms")

    print("\n[node outputs]")
    for event in result["node_events"]:
        print(
            f"\n[{event['node']}] elapsed_since_previous_update_ms="
            f"{event['elapsed_since_previous_update_ms']}"
        )
        print(json.dumps(event["output"], ensure_ascii=False, indent=2))

    final_state = result["final_state"]
    print("\n[final]")
    print(json.dumps(final_state, ensure_ascii=False, indent=2))


async def amain(argv: list[str]) -> int:
    load_dotenv_once()
    args = parse_args(argv)

    import os

    if args.model is None:
        args.model = os.getenv("LLM_MODEL", DEFAULT_GEMINI_MODEL) if args.provider == "gemini" else DEFAULT_OLLAMA_MODEL

    if args.provider == "gemini":
        health = check_gemini(args.model)
        args.base_url = args.gemini_base_url
        api_key, key_source = resolve_gemini_api_key()
        args.api_key = api_key or ""
        if not health["ok"]:
            if args.json:
                print(json.dumps({"provider": "gemini", "health": health}, ensure_ascii=False, indent=2))
            else:
                print(health["hint"])
            return 1
        if not args.json:
            print("Provider: gemini")
            print(f"Gemini base URL: {args.base_url}")
            print(f"Gemini model: {args.model}")
            print(f"Gemini API key source: {key_source}")
    else:
        health = check_ollama(args.ollama_base_url, args.model)
        args.base_url = args.ollama_base_url
        args.api_key = "ollama"
        if not health["ok"]:
            if args.json:
                print(json.dumps({"provider": "ollama", "health": health}, ensure_ascii=False, indent=2))
            else:
                print(f"Ollama not reachable: {health['error']}")
                print(health["hint"])
            return 1
        if not args.json:
            print("Provider: ollama")
            print(f"Ollama tags URL: {health['tags_url']}")
            if health.get("hint"):
                print(health["hint"])
            else:
                print(f"Ollama reachable. Using model: {args.model}")

    try:
        result = await run_graph(args)
    except Exception as exc:  # noqa: BLE001 - CLI should show the full failure.
        failure = {
            "provider": args.provider,
            "model": args.model,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        if args.json:
            print(json.dumps(failure, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1

    output = {
        "provider": args.provider,
        "model": args.model,
        "base_url": args.base_url,
        "query": args.query,
        "health": health,
        "result": result,
    }

    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print_human(result)

    return 0


def main() -> None:
    raise SystemExit(asyncio.run(amain(sys.argv[1:])))


if __name__ == "__main__":
    main()
