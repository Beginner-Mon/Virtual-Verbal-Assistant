"""In-process @tool wrappers for retriever agent.

pgvector is NOT an MCP server — runs in-process, 0ms network.
Only Kimodo + web_search will be MCP (Phase 3).
"""

import asyncio
from langchain_core.tools import tool

from langgraph_agents.shared import get_pg_client, get_embedding_service
from langgraph_agents.db.vector_backend import VectorBackend


@tool
async def pgvector_search(query: str, top_k: int = 5, source_type: str = "document") -> list[dict]:
    """Search internal medical knowledge base for exercises, treatments, and PT theory.

    Use for knowledge_query and exercise_recommendation intents.
    Returns documents ranked by cosine similarity (highest first).

    Args:
        query: Semantic search query (use expanded_query from planner)
        top_k: Number of results to return (default 5)
        source_type: One of "document", "humanml3d" (default "document")
    """
    pg = get_pg_client()
    svc = get_embedding_service()

    embedding = await asyncio.to_thread(svc.embed_texts, query)
    if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
        embedding = embedding[0]

    vb = VectorBackend(pg)
    results = await vb.search(
        query_embedding=embedding,
        top_k=top_k,
        source_type=source_type,
    )
    return [
        {
            "content": r["content"],
            "similarity": round(r["similarity"], 3),
            "source_type": r.get("metadata", {}).get("source_type", source_type),
        }
        for r in results
    ]
