"""Spike — verify token streaming through astream_events."""

import asyncio
import os
import sys

# Add langgraph_agents to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agenticRAG", "agentic_rag_gemini"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "agenticRAG", "agentic_rag_gemini", ".env"))

from langgraph_agents.graph import build_graph_async


async def main():
    graph = await build_graph_async()
    state = {"messages": [], "errors": [], "retry_count": 0, "total_tokens": 0}
    config = {"configurable": {
        "user_id": "spike", "session_id": "spike-001",
        "query": "Xin chao",
        "persona_id": "eca_default", "output_mode": "text",
        "request_id": "spike", "token_limit": None,
    }}

    seen_events = {}
    async for event in graph.astream_events(state, config=config, version="v2"):
        ev_type = event["event"]
        seen_events[ev_type] = seen_events.get(ev_type, 0) + 1
        if ev_type == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            print(f"TOKEN: {chunk.content!r}")

    print("\nEvent counts:")
    for k, v in sorted(seen_events.items()):
        print(f"  {k}: {v}")

    token_count = seen_events.get("on_chat_model_stream", 0)
    if token_count >= 5:
        print(f"\n✓ PASS — {token_count} on_chat_model_stream events (>= 5)")
    else:
        print(f"\n✗ FAIL — only {token_count} on_chat_model_stream events (< 5)")
        print("  Need alternate conversation streaming approach.")


if __name__ == "__main__":
    asyncio.run(main())
