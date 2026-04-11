"""Session Store — Firebase Firestore persistence for chat sessions.

Manages chat sessions as Firestore documents under
``users/{user_id}/sessions/{session_id}``.

Each session document contains a full message transcript plus metadata.
Summaries are generated lazily (on session switch) by the Summarize Agent.

Environment variables
---------------------
FIRESTORE_PROJECT_ID : str, optional
    GCP project ID.  If unset the SDK infers from credentials.
GOOGLE_APPLICATION_CREDENTIALS : str, optional
    Path to the Firebase/GCP service-account JSON key file.
    Not required if running on GCP with default credentials.

Falls back to local JSON-on-disk storage if Firestore is unavailable,
so existing local dev workflows are unaffected.
"""

import json
import uuid
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

from utils.logger import get_logger

logger = get_logger(__name__)

# Package root: agenticRAG/agentic_rag_gemini/
_PKG_ROOT = Path(__file__).parent.parent

# Explicitly load .env from the package root so Firebase env vars are available
# even if another module's load_dotenv() didn't find this .env file.
from dotenv import load_dotenv
load_dotenv(_PKG_ROOT / ".env")

# Default base directory for local fallback session files
DEFAULT_SESSIONS_DIR = _PKG_ROOT / "data" / "sessions"

# ------------------------------------------------------------------
# Firestore initialisation (lazy singleton)
# ------------------------------------------------------------------
_firestore_db = None
_firestore_init_attempted = False


def _get_firestore_db():
    """Return the Firestore client singleton, or None if unavailable."""
    global _firestore_db, _firestore_init_attempted
    if _firestore_init_attempted:
        return _firestore_db

    _firestore_init_attempted = True
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        # Avoid reinitialising if another module already did
        if not firebase_admin._apps:
            cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            project_id = os.getenv("FIRESTORE_PROJECT_ID")

            # Resolve cred_path: try as-is, then relative to this package's root
            resolved_cred = None
            if cred_path:
                if os.path.isfile(cred_path):
                    resolved_cred = cred_path
                else:
                    # Try relative to agenticRAG/agentic_rag_gemini/
                    alt = _PKG_ROOT / cred_path.lstrip("./")
                    if alt.is_file():
                        resolved_cred = str(alt)
                    else:
                        # Try just the filename in the package root
                        alt2 = _PKG_ROOT / Path(cred_path).name
                        if alt2.is_file():
                            resolved_cred = str(alt2)

            if resolved_cred:
                logger.info("Using Firebase credentials from: %s", resolved_cred)
                cred = credentials.Certificate(resolved_cred)
                firebase_admin.initialize_app(cred, {"projectId": project_id} if project_id else None)
            else:
                logger.info("No credential file found, using Application Default Credentials")
                # Application Default Credentials (e.g. on GCP)
                firebase_admin.initialize_app(options={"projectId": project_id} if project_id else None)

        _firestore_db = firestore.client()
        logger.info("Firestore client initialised successfully")
    except Exception as exc:
        logger.warning("Firestore unavailable (%s) — falling back to local JSON storage", exc)
        _firestore_db = None

    return _firestore_db


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
    """Manages chat sessions via Firestore with local JSON fallback.

    Firestore layout::

        users/{user_id}/sessions/{session_id}
            → { session_id, user_id, title, created_at, updated_at,
                messages[], summary, is_summarized }

    Local fallback layout::

        data/sessions/{user_id}/{session_id}.json
    """

    def __init__(self, user_id: str, base_dir: Optional[Path] = None):
        self.user_id = user_id
        self._db = _get_firestore_db()

        # Local fallback
        self.base_dir = base_dir or DEFAULT_SESSIONS_DIR
        self.user_dir = self.base_dir / user_id

        if self._db is None:
            # Ensure local directory exists for fallback
            self.user_dir.mkdir(parents=True, exist_ok=True)

    @property
    def _use_firestore(self) -> bool:
        return self._db is not None

    # ------------------------------------------------------------------
    # Firestore helpers
    # ------------------------------------------------------------------

    def _sessions_ref(self):
        """Return the Firestore collection reference for this user's sessions."""
        return self._db.collection("users").document(self.user_id).collection("sessions")

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

        if self._use_firestore:
            self._sessions_ref().document(session_id).set(session_data)
        else:
            self._write_session_local(session_id, session_data)

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

        turn: Dict[str, Any] = {
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
        """Delete a session. Returns True if deleted."""
        if self._use_firestore:
            doc_ref = self._sessions_ref().document(session_id)
            doc = doc_ref.get()
            if doc.exists:
                doc_ref.delete()
                logger.info(f"Deleted session {session_id} from Firestore")
                return True
            return False
        else:
            path = self._session_path_local(session_id)
            if path.exists():
                path.unlink()
                logger.info(f"Deleted session {session_id}")
                return True
            return False

    # ------------------------------------------------------------------
    # Listing & pruning
    # ------------------------------------------------------------------

    def list_sessions(self, limit: Optional[int] = None) -> List[SessionMeta]:
        """Return non-empty sessions ordered by most-recently-updated first."""
        if self._use_firestore:
            return self._list_sessions_firestore(limit)
        return self._list_sessions_local(limit)

    def _list_sessions_firestore(self, limit: Optional[int] = None) -> List[SessionMeta]:
        query = self._sessions_ref().order_by("updated_at", direction="DESCENDING")
        if limit:
            query = query.limit(limit * 2)  # fetch extra to filter empties

        metas: List[SessionMeta] = []
        for doc in query.stream():
            data = doc.to_dict()
            msg_count = len(data.get("messages", []))
            if msg_count == 0:
                continue
            metas.append(
                SessionMeta(
                    session_id=data["session_id"],
                    title=data.get("title", "Untitled"),
                    created_at=data.get("created_at", ""),
                    updated_at=data.get("updated_at", ""),
                    message_count=msg_count,
                    is_summarized=data.get("is_summarized", False),
                )
            )

        if limit is not None:
            return metas[:limit]
        return metas

    def _list_sessions_local(self, limit: Optional[int] = None) -> List[SessionMeta]:
        metas: List[SessionMeta] = []
        for path in self.user_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                msg_count = len(data.get("messages", []))
                if msg_count == 0:
                    continue
                metas.append(
                    SessionMeta(
                        session_id=data["session_id"],
                        title=data.get("title", "Untitled"),
                        created_at=data.get("created_at", ""),
                        updated_at=data.get("updated_at", ""),
                        message_count=msg_count,
                        is_summarized=data.get("is_summarized", False),
                    )
                )
            except (json.JSONDecodeError, KeyError) as exc:
                logger.warning(f"Skipping corrupt session file {path}: {exc}")

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
        if self._use_firestore:
            count = 0
            for doc in self._sessions_ref().stream():
                doc.reference.delete()
                count += 1
            logger.info(f"Cleared {count} sessions for user {self.user_id} from Firestore")
            return count
        else:
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
    # Internal read/write (dispatch to Firestore or local)
    # ------------------------------------------------------------------

    def _read_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        if self._use_firestore:
            return self._read_session_firestore(session_id)
        return self._read_session_local(session_id)

    def _write_session(self, session_id: str, data: Dict[str, Any]) -> None:
        if self._use_firestore:
            self._write_session_firestore(session_id, data)
        else:
            self._write_session_local(session_id, data)

    # -- Firestore -------------------------------------------------

    def _read_session_firestore(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            doc = self._sessions_ref().document(session_id).get()
            if doc.exists:
                return doc.to_dict()
            return None
        except Exception as exc:
            logger.error(f"Failed to read session {session_id} from Firestore: {exc}")
            return None

    def _write_session_firestore(self, session_id: str, data: Dict[str, Any]) -> None:
        try:
            self._sessions_ref().document(session_id).set(data)
        except Exception as exc:
            logger.error(f"Failed to write session {session_id} to Firestore: {exc}")

    # -- Local JSON fallback ----------------------------------------

    def _session_path_local(self, session_id: str) -> Path:
        return self.user_dir / f"{session_id}.json"

    def _read_session_local(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = self._session_path_local(session_id)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.error(f"Failed to read session {session_id}: {exc}")
            return None

    def _write_session_local(self, session_id: str, data: Dict[str, Any]) -> None:
        path = self._session_path_local(session_id)
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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
