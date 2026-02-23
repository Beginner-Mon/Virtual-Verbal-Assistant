"""Tests for SessionStore — JSON-on-disk session persistence.

Run:  python -m pytest tests/test_session_store.py -v
"""

import tempfile
import shutil
from pathlib import Path

import pytest

# Adjust path so we can import from the main package
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from memory.session_store import SessionStore, SessionMeta


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for session files."""
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def store(tmp_dir):
    """Create a SessionStore backed by a temp directory."""
    return SessionStore(user_id="test_user", base_dir=tmp_dir)


class TestSessionCreation:
    def test_create_session_returns_uuid(self, store):
        sid = store.create_session()
        assert isinstance(sid, str)
        assert len(sid) == 36  # UUID format

    def test_create_session_creates_file(self, store):
        sid = store.create_session()
        assert (store.user_dir / f"{sid}.json").exists()

    def test_new_session_has_default_title(self, store):
        sid = store.create_session()
        data = store.load_session(sid)
        assert data["title"] == "New conversation"

    def test_new_session_with_first_message(self, store):
        sid = store.create_session(first_message="Tell me about neck pain")
        data = store.load_session(sid)
        assert data["title"] == "Tell me about neck pain"


class TestTurnPersistence:
    def test_save_and_load_turns(self, store):
        sid = store.create_session()
        store.save_turn(sid, "user", "Hello!")
        store.save_turn(sid, "assistant", "Hi there!")

        data = store.load_session(sid)
        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][1]["content"] == "Hi there!"

    def test_save_turn_updates_title_from_first_user_msg(self, store):
        sid = store.create_session()
        store.save_turn(sid, "user", "What exercises help with lower back pain?")
        data = store.load_session(sid)
        assert "lower back" in data["title"].lower()

    def test_save_turn_nonexistent_session_raises(self, store):
        with pytest.raises(FileNotFoundError):
            store.save_turn("nonexistent-id", "user", "test")


class TestListing:
    def test_list_empty(self, store):
        assert store.list_sessions() == []

    def test_list_sessions_ordered_newest_first(self, store):
        sid1 = store.create_session(first_message="First")
        store.save_turn(sid1, "user", "one")

        sid2 = store.create_session(first_message="Second")
        store.save_turn(sid2, "user", "two")

        sessions = store.list_sessions()
        assert len(sessions) == 2
        assert sessions[0].session_id == sid2  # newer

    def test_list_sessions_with_limit(self, store):
        for i in range(5):
            store.create_session(first_message=f"Session {i}")
        sessions = store.list_sessions(limit=3)
        assert len(sessions) == 3


class TestPruning:
    def test_delete_oldest_keeps_n(self, store):
        ids = [store.create_session(first_message=f"S{i}") for i in range(5)]
        deleted = store.delete_oldest_sessions(keep=2)
        assert deleted == 3
        remaining = store.list_sessions()
        assert len(remaining) == 2

    def test_delete_oldest_noop_when_under_limit(self, store):
        store.create_session()
        deleted = store.delete_oldest_sessions(keep=5)
        assert deleted == 0


class TestSummarization:
    def test_mark_summarized(self, store):
        sid = store.create_session()
        store.save_turn(sid, "user", "Q1")
        store.save_turn(sid, "assistant", "A1")
        store.mark_summarized(sid, "Summary of Q1 and A1")

        data = store.load_session(sid)
        assert data["is_summarized"] is True
        assert data["summary"] == "Summary of Q1 and A1"

    def test_get_unsummarized_returns_none_if_too_few_messages(self, store):
        sid = store.create_session()
        store.save_turn(sid, "user", "Only one message")
        assert store.get_unsummarized_session(sid) is None

    def test_get_unsummarized_returns_data(self, store):
        sid = store.create_session()
        store.save_turn(sid, "user", "Q")
        store.save_turn(sid, "assistant", "A")
        data = store.get_unsummarized_session(sid)
        assert data is not None
        assert len(data["messages"]) == 2

    def test_get_unsummarized_returns_none_after_summary(self, store):
        sid = store.create_session()
        store.save_turn(sid, "user", "Q")
        store.save_turn(sid, "assistant", "A")
        store.mark_summarized(sid, "Done")
        assert store.get_unsummarized_session(sid) is None


class TestClearAll:
    def test_clear_all_sessions(self, store):
        for i in range(3):
            store.create_session()
        count = store.clear_all_sessions()
        assert count == 3
        assert store.list_sessions() == []


class TestTitleGeneration:
    def test_short_message_no_ellipsis(self, store):
        title = SessionStore._generate_title("Hello")
        assert title == "Hello"
        assert "…" not in title

    def test_long_message_truncated(self, store):
        long = "A" * 100
        title = SessionStore._generate_title(long)
        assert len(title) <= 51  # 50 + ellipsis
        assert title.endswith("…")

    def test_none_returns_default(self, store):
        assert SessionStore._generate_title(None) == "New conversation"
