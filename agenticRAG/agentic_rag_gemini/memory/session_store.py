"""Session Store — JSON-on-disk persistence for chat sessions.

Manages chat sessions as individual JSON files under ./data/sessions/{user_id}/.
Each session contains a full message transcript plus metadata.
Summaries are generated lazily (on session switch) by the Summarize Agent.
"""

import json
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Default base directory for session files
DEFAULT_SESSIONS_DIR = Path(__file__).parent.parent / "data" / "sessions"


class SessionMeta:
    """Lightweight metadata about a session (for listing, no full messages)."""

    def __init__(
        self,
        session_id: str,
        title: str,
        created_at: str,
        updated_at: str,
        message_count: int,
        is_summarized: bool = False,
    ):
        self.session_id = session_id
        self.title = title
        self.created_at = created_at
        self.updated_at = updated_at
        self.message_count = message_count
        self.is_summarized = is_summarized

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "message_count": self.message_count,
            "is_summarized": self.is_summarized,
        }


class SessionStore:
    """Manages chat sessions as JSON files on disk.

    Directory layout::

        data/sessions/<user_id>/
            <session_id>.json
            <session_id>.json
    """

    def __init__(self, user_id: str, base_dir: Optional[Path] = None):
        self.user_id = user_id
        self.base_dir = base_dir or DEFAULT_SESSIONS_DIR
        self.user_dir = self.base_dir / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core CRUD
    # ------------------------------------------------------------------

    def create_session(self, first_message: Optional[str] = None) -> str:
        """Create a new empty session and return its ID."""
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat()

        title = self._generate_title(first_message) if first_message else "New conversation"

        session_data = {
            "session_id": session_id,
            "user_id": self.user_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
            "summary": None,
            "is_summarized": False,
        }

        self._write_session(session_id, session_data)
        logger.info(f"Created session {session_id} for user {self.user_id}")
        return session_id

    def save_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append a single turn (user or assistant) to the session."""
        data = self._read_session(session_id)
        if data is None:
            raise FileNotFoundError(f"Session {session_id} not found")

        turn = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
        }
        if metadata:
            turn["metadata"] = metadata

        data["messages"].append(turn)
        data["updated_at"] = datetime.now().isoformat()

        # Auto-update title from first user message if still default
        if data["title"] == "New conversation" and role == "user":
            data["title"] = self._generate_title(content)

        self._write_session(session_id, data)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load a full session (messages + metadata). Returns None if not found."""
        return self._read_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session file. Returns True if deleted."""
        path = self._session_path(session_id)
        if path.exists():
            path.unlink()
            logger.info(f"Deleted session {session_id}")
            return True
        return False

    # ------------------------------------------------------------------
    # Listing & pruning
    # ------------------------------------------------------------------

    def list_sessions(self, limit: Optional[int] = None) -> List[SessionMeta]:
        """Return sessions ordered by most-recently-updated first.

        Args:
            limit: Max number of sessions to return. None = all.

        Returns:
            List of SessionMeta ordered newest-first.
        """
        metas: List[SessionMeta] = []
        for path in self.user_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                metas.append(
                    SessionMeta(
                        session_id=data["session_id"],
                        title=data.get("title", "Untitled"),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        message_count=len(data.get("messages", [])),
                        is_summarized=data.get("is_summarized", False),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Skipping corrupt session file {path}: {exc}")

        # Sort newest first
        metas.sort(key=lambda m: m.updated_at, reverse=True)
        if limit is not None:
            return metas[:limit]
        return metas

    def delete_oldest_sessions(self, keep: int) -> int:
        """Delete sessions beyond *keep* most recent. Returns count deleted."""
        all_sessions = self.list_sessions()
        if len(all_sessions) <= keep:
            return 0

        to_delete = all_sessions[keep:]
        deleted = 0
        for meta in to_delete:
            if self.delete_session(meta.session_id):
                deleted += 1
        logger.info(f"Pruned {deleted} old sessions for user {self.user_id}")
        return deleted

    def clear_all_sessions(self) -> int:
        """Delete every session for this user. Returns count deleted."""
        count = 0
        for path in self.user_dir.glob("*.json"):
            path.unlink()
            count += 1
        logger.info(f"Cleared {count} sessions for user {self.user_id}")
        return count

    # ------------------------------------------------------------------
    # Summary helpers
    # ------------------------------------------------------------------

    def mark_summarized(self, session_id: str, summary: str) -> None:
        """Store the summary text and flip the is_summarized flag."""
        data = self._read_session(session_id)
        if data is None:
            return
        data["summary"] = summary
        data["is_summarized"] = True
        self._write_session(session_id, data)
        logger.info(f"Session {session_id} marked as summarized")

    def get_unsummarized_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return session data only if it has ≥2 messages and is not yet summarized."""
        data = self._read_session(session_id)
        if data is None:
            return None
        if data.get("is_summarized", False):
            return None
        if len(data.get("messages", [])) < 2:
            return None
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _session_path(self, session_id: str) -> Path:
        return self.user_dir / f"{session_id}.json"

    def _read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self._session_path(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read session {session_id}: {exc}")
            return None

    def _write_session(self, session_id: str, data: Dict[str, Any]) -> None:
        path = self._session_path(session_id)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @staticmethod
    def _generate_title(message: Optional[str]) -> str:
        """Derive a short title from the first user message (no LLM call)."""
        if not message:
            return "New conversation"
        # Take first 50 chars, strip, add ellipsis if truncated
        title = message.strip().replace("\n", " ")[:50]
        if len(message.strip()) > 50:
            title += "…"
        return title
