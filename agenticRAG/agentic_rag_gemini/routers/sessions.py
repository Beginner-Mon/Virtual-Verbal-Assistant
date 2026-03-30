from typing import Any, Dict

import httpx
from fastapi import APIRouter, HTTPException

from core.config.settings import get_main_api_settings

router = APIRouter(tags=["sessions"])
SETTINGS = get_main_api_settings()


@router.post("/sessions", summary="Create a new chat session")
async def proxy_create_session(body: Dict[str, Any]) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=SETTINGS.downstream_session_timeout) as client:
        resp = await client.post(f"{SETTINGS.agentic_rag_url}/sessions", json=body)
        resp.raise_for_status()
        return resp.json()


@router.get("/sessions/{user_id}", summary="List all sessions for a user")
async def proxy_list_sessions(user_id: str) -> Any:
    async with httpx.AsyncClient(timeout=SETTINGS.downstream_session_timeout) as client:
        resp = await client.get(f"{SETTINGS.agentic_rag_url}/sessions/{user_id}")
        resp.raise_for_status()
        return resp.json()


@router.get("/sessions/{user_id}/{session_id}", summary="Get full session with messages")
async def proxy_get_session(user_id: str, session_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=SETTINGS.downstream_session_timeout) as client:
        resp = await client.get(f"{SETTINGS.agentic_rag_url}/sessions/{user_id}/{session_id}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        resp.raise_for_status()
        return resp.json()


@router.delete("/sessions/{user_id}/{session_id}", summary="Delete a chat session")
async def proxy_delete_session(user_id: str, session_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=SETTINGS.downstream_session_timeout) as client:
        resp = await client.delete(f"{SETTINGS.agentic_rag_url}/sessions/{user_id}/{session_id}")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        resp.raise_for_status()
        return resp.json()


@router.post("/sessions/{user_id}/{session_id}/summarize", summary="Summarize session to ChromaDB")
async def proxy_summarize_session(user_id: str, session_id: str) -> Dict[str, Any]:
    async with httpx.AsyncClient(timeout=SETTINGS.downstream_timeout) as client:
        resp = await client.post(f"{SETTINGS.agentic_rag_url}/sessions/{user_id}/{session_id}/summarize")
        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        resp.raise_for_status()
        return resp.json()
