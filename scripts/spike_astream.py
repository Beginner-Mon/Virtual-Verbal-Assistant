"""Spike — verify graph.astream(stream_mode="updates") produces stage events + final_answer.

Token streaming is handled at the FastAPI layer (word-by-word from final_answer).
"""

import asyncio
import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "agenticRAG", "agentic_rag_gemini"))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "agenticRAG", "agentic_rag_gemini", ".env"))

from langgraph_agents.graph import build_graph_async


async def main():
    graph = await build_graph_async()
    state = {"messages": [], "errors": [], "retry_count": 0, "total_tokens": 0}
    config = {"configurable": {
        "user_id": "spike", "session_id": "spike-003",
        "query": "Xin chao",
        "persona_id": "eca_default", "output_mode": "text",
        "request_id": "spike", "token_limit": None,
    }}

    stage_nodes = []
    final_answer = ""
    final_total_tokens = 0

    async for chunk in graph.astream(state, config, stream_mode="updates"):
        # Single stream_mode → yields raw chunk, not (mode, chunk) tuple
        for node_name, node_output in chunk.items():
            stage_nodes.append(node_name)
            if node_name == "conversation":
                final_answer = node_output.get("final_answer", "")
                final_total_tokens = node_output.get("total_tokens", 0)

    print(f"Stage nodes: {stage_nodes}")
    print(f"Total tokens: {final_total_tokens}")
    # Safe Unicode print
    safe_answer = final_answer[:120].encode("ascii", errors="replace").decode("ascii")
    print(f"Final answer preview: {safe_answer}...")

    has_all_nodes = all(n in stage_nodes for n in ["memory", "planner", "conversation"])
    has_answer = len(final_answer) > 10

    if has_all_nodes and has_answer:
        print(f"\nPASS -- graph runs, all nodes present, answer length={len(final_answer)}")
    else:
        print(f"\nFAIL -- all_nodes={has_all_nodes}, has_answer={has_answer}")


if __name__ == "__main__":
    asyncio.run(main())
