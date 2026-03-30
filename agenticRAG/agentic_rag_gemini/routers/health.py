from typing import Any, Dict

from fastapi import APIRouter

from core.config.settings import get_main_api_settings
from services.main_api_health import check_services_health

router = APIRouter(tags=["health"])
SETTINGS = get_main_api_settings()


@router.get("/health", summary="Health check")
async def health_check() -> Dict[str, Any]:
    statuses = await check_services_health(SETTINGS.agentic_rag_url, SETTINGS.dart_url)
    overall = "healthy" if all(v == "ok" for v in statuses.values()) else "degraded"
    return {"status": overall, "services": statuses}


@router.get("/info", summary="Get service info")
async def get_info() -> Dict[str, Any]:
    return {
        "service": "Unified Multi-Service Pipeline",
        "version": "2.1.0",
        "async_enrichment": SETTINGS.async_enrichment,
        "include_debug": SETTINGS.include_debug,
        "upstream_services": {
            "agenticrag": f"{SETTINGS.agentic_rag_url}/query",
            "dart": f"{SETTINGS.dart_url}/generate",
            "tts": f"{SETTINGS.tts_url}/synthesize",
        },
        "endpoints": {
            "POST /answer": "Fast text-first response (supports session_id), optional async motion/TTS enrichment",
            "GET /answer/status/{request_id}": "Poll async enrichment status/results including debug diagnostics",
            "POST /query": "Compatibility alias for 8000-style query clients (supports session_id)",
            "POST /process_query": "Compatibility task submission endpoint",
            "GET /tasks/{task_id}": "Compatibility task polling endpoint",
            "POST /sessions": "Create a new chat session",
            "GET /sessions/{user_id}": "List all sessions for a user",
            "GET /sessions/{user_id}/{session_id}": "Get full session with messages",
            "DELETE /sessions/{user_id}/{session_id}": "Delete a chat session",
            "POST /sessions/{user_id}/{session_id}/summarize": "Summarize session to ChromaDB",
            "GET /health": "Ping downstream services",
            "GET /info": "This document",
        },
    }
