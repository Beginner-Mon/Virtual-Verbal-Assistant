"""Embedding service — intfloat/multilingual-e5-small (D10).

Replaces all-MiniLM-L6-v2. Same dimension (384) but stronger for Vietnamese.
CRITICAL: e5 models REQUIRE prefix — query: for queries, passage: for documents.
Without prefix, embedding quality degrades significantly.

Usage:
    embed = get_embedding()
    query_vec = embed.embed_query("bài tập cho đau lưng")     # auto-prefixed query:
    doc_vec   = embed.embed_passage("Hướng dẫn bài tập...")   # auto-prefixed passage:
"""

from __future__ import annotations

import asyncio
import os
import threading
from functools import lru_cache
from typing import List, Union

import numpy as np


E5_MODEL_NAME = "intfloat/multilingual-e5-small"
E5_DIM = 384
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class E5EmbeddingService:
    """Thin wrapper around sentence-transformers with mandatory e5 prefixes.

    Singleton via get_embedding(). Thread-safe for encode(); async wrappers
    use asyncio.to_thread for non-blocking embedding in async contexts.
    """

    def __init__(self, model_name: str = E5_MODEL_NAME):
        self.model_name = model_name
        self.dim = E5_DIM
        self._model = None
        self._load_lock = threading.Lock()

    @property
    def model(self):
        """Lazy-load the SentenceTransformer on first use.

        Loads from local cache by default (no HF-Hub round-trips on restart).
        Set EMBEDDING_ALLOW_DOWNLOAD=1 for a one-time download on a clean machine.

        Double-checked locking: aembed_query/aembed_passage run this via
        asyncio.to_thread, so concurrent tool calls (e.g. a retriever round that
        fires several kb_search calls in parallel) can all reach this property
        before any of them finishes loading. Without a lock, each thread sees
        self._model is None and independently constructs its own
        SentenceTransformer — loading the model N times simultaneously, spiking
        peak memory (crashed the process under memory pressure — reproduced
        live: 3 parallel kb_search calls → 3 concurrent loads → OOM). The lock
        only serializes the ONE-TIME load; every call after that takes the fast
        path (no lock) since self._model is no longer None. This does not change
        the lazy-loading behavior itself — still nothing loads until first use.
        """
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                _offline = os.getenv("EMBEDDING_ALLOW_DOWNLOAD") != "1"
                self._model = SentenceTransformer(self.model_name, local_files_only=_offline)
        return self._model

    # ── Public API ──────────────────────────────────────────────────────

    def embed_query(self, query: str) -> list[float]:
        """Embed a search query. Auto-prepends 'query:' prefix."""
        return self._encode(f"{QUERY_PREFIX}{query}")

    def embed_queries(self, queries: list[str]) -> list[list[float]]:
        """Batch-embed search queries. Each gets 'query:' prefix."""
        prefixed = [f"{QUERY_PREFIX}{q}" for q in queries]
        return self._encode_batch(prefixed)

    def embed_passage(self, passage: str) -> list[float]:
        """Embed a document passage. Auto-prepends 'passage:' prefix."""
        return self._encode(f"{PASSAGE_PREFIX}{passage}")

    def embed_passages(self, passages: list[str]) -> list[list[float]]:
        """Batch-embed document passages. Each gets 'passage:' prefix."""
        prefixed = [f"{PASSAGE_PREFIX}{p}" for p in passages]
        return self._encode_batch(prefixed)

    def embed(self, text: str, is_query: bool = True) -> list[float]:
        """Convenience: embed with explicit prefix choice."""
        return self.embed_query(text) if is_query else self.embed_passage(text)

    # ── Async wrappers (non-blocking) ───────────────────────────────────

    async def aembed_query(self, query: str) -> list[float]:
        return await asyncio.to_thread(self.embed_query, query)

    async def aembed_queries(self, queries: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_queries, queries)

    async def aembed_passage(self, passage: str) -> list[float]:
        return await asyncio.to_thread(self.embed_passage, passage)

    async def aembed_passages(self, passages: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_passages, passages)

    # ── Internal ────────────────────────────────────────────────────────

    def _encode(self, text: str) -> list[float]:
        vec = self.model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        return vec.tolist()

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False, batch_size=32)
        return vecs.tolist()

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Cosine similarity between two vectors."""
        va, vb = np.array(a), np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    def get_dimension(self) -> int:
        return self.dim

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Rough token estimate: ~4 chars/token for Vietnamese/English."""
        return len(text) // 4


# ── Singleton ─────────────────────────────────────────────────────────────

_embedding_instance: E5EmbeddingService | None = None


def get_embedding() -> E5EmbeddingService:
    """Get or create the global E5EmbeddingService singleton."""
    global _embedding_instance
    if _embedding_instance is None:
        _embedding_instance = E5EmbeddingService()
    return _embedding_instance
