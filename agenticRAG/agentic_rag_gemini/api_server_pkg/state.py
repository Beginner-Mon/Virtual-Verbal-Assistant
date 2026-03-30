"""State and persistence helpers for the dedicated api_server package."""

import asyncio
import json
import os
import re
from typing import Any, Dict, List

from utils.logger import get_logger

logger = get_logger(__name__)

# In-memory task store for unified /process_query -> /tasks/{id} polling flow.
TASK_STORE: Dict[str, Dict[str, Any]] = {}
TASK_STORE_LOCK = asyncio.Lock()
CHAT_HISTORY_DIR = os.getenv("CHAT_HISTORY_DIR", "./memory/chat_history")
CHAT_HISTORY_LOCK = asyncio.Lock()


def _history_file_path(user_id: str) -> str:
    safe_user = re.sub(r"[^a-zA-Z0-9._-]", "_", (user_id or "guest")).strip("._-") or "guest"
    os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
    return os.path.join(CHAT_HISTORY_DIR, f"{safe_user}.json")


async def _read_chat_history(user_id: str) -> List[Dict[str, Any]]:
    file_path = _history_file_path(user_id)
    if not os.path.exists(file_path):
        return []

    async with CHAT_HISTORY_LOCK:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if isinstance(payload, list):
                return payload
        except Exception as exc:
            logger.warning("Failed to read chat history %s: %s", file_path, exc)
    return []


async def _append_chat_history(user_id: str, entries: List[Dict[str, Any]]) -> None:
    if not entries:
        return

    file_path = _history_file_path(user_id)
    async with CHAT_HISTORY_LOCK:
        history: List[Dict[str, Any]] = []
        if os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                if isinstance(payload, list):
                    history = payload
            except Exception:
                history = []

        history.extend(entries)
        # Keep file size bounded for UI load performance.
        history = history[-200:]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
