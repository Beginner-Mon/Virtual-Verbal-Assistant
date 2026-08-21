"""Prove the ONNX embedding path matches the torch one. Blocks the switch.

    python scripts/verify_onnx_parity.py            # KB rows from Neon
    python scripts/verify_onnx_parity.py --offline  # built-in texts, no DB

Why this is not optional
------------------------
`shared/onnx_backend.py` reimplements two stages that sentence-transformers
performs after the model: masked mean pooling and L2 normalization. Both produce
output of the correct shape whether or not they are correct. Get the mean's
denominator wrong and you still get 384 floats, pgvector still accepts them,
every existing test still passes — the vectors are simply a little wrong, more
so for longer inputs, and recall falls without a single error anywhere.

The 2918 rows in `kb_embeddings` were embedded with the torch path. Switching
runtimes without proving agreement means the query vector and the stored vectors
come from different functions.

What is checked
---------------
1. Cosine similarity per sample, against the torch path, on REAL KB text.
2. Long inputs specifically (>512 tokens, where truncation and pooling errors
   show up) and a mixed-length batch (where a padding-ratio bug shows up but a
   one-text-at-a-time test cannot).
3. Retrieval order: the top-5 of a vector search must not be reshuffled, which
   is the thing users would actually notice.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agenticRAG"))

# Thresholds. 0.9999 rather than "close enough": fp32 ONNX of the same weights
# should agree to ~1e-6, so anything looser would pass a genuinely broken
# pooling implementation.
COSINE_FLOOR = 0.9999

_OFFLINE_TEXTS = [
    "bài tập vật lý trị liệu cho thoát vị đĩa đệm L4-L5",
    "đau lưng dưới khi ngồi lâu nên làm gì",
    "cat-cow stretch hướng dẫn từng bước",
    "chống chỉ định của bài tập squat với người đau gối",
    "How many repetitions for a beginner core routine?",
]


async def _load_kb_texts(limit: int) -> list[str]:
    from langgraph_agents.shared import get_pg_client

    pg = get_pg_client()
    await pg.connect()
    rows = await pg.fetch(
        """
        SELECT content FROM kb_embeddings
        WHERE content IS NOT NULL AND length(content) > 40
        ORDER BY random()
        LIMIT $1
        """,
        limit,
    )
    return [r["content"] for r in rows]


def _build(backend: str):
    from langgraph_agents.shared.embedding import E5EmbeddingService

    return E5EmbeddingService(backend=backend)


def _cosines(a: list[list[float]], b: list[list[float]]) -> np.ndarray:
    va, vb = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    num = (va * vb).sum(axis=1)
    den = np.linalg.norm(va, axis=1) * np.linalg.norm(vb, axis=1)
    return num / np.clip(den, 1e-12, None)


def _report(label: str, cos: np.ndarray) -> bool:
    ok = bool((cos >= COSINE_FLOOR).all())
    worst = float(cos.min())
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {label}: n={len(cos)} min={worst:.8f} mean={cos.mean():.8f}")
    if not ok:
        bad = int(np.argmin(cos))
        print(f"         worst sample index {bad}, cosine {worst:.8f} < {COSINE_FLOOR}")
    return ok


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200)
    parser.add_argument("--offline", action="store_true",
                        help="skip the database, use built-in texts")
    args = parser.parse_args()

    os.environ.setdefault("EMBEDDING_ALLOW_DOWNLOAD", "0")

    if args.offline:
        texts = list(_OFFLINE_TEXTS)
        print(f"[parity] offline mode — {len(texts)} built-in texts")
    else:
        try:
            texts = await _load_kb_texts(args.sample)
        except Exception as exc:                                # noqa: BLE001
            print(f"[parity] could not read kb_embeddings: {exc}")
            print("[parity] rerun with --offline to check the maths without the DB")
            return 2
        print(f"[parity] {len(texts)} rows from kb_embeddings")

    if not texts:
        print("[parity] no texts to compare")
        return 2

    torch_svc, onnx_svc = _build("torch"), _build("onnx")
    all_ok = True

    # 1. One at a time — the ordinary path, and the only one a naive
    #    implementation gets right.
    print("[parity] per-text (passage)")
    t_single = [torch_svc.embed_passage(t) for t in texts]
    o_single = [onnx_svc.embed_passage(t) for t in texts]
    all_ok &= _report("single", _cosines(t_single, o_single))

    # 2. Batched, mixed lengths. THIS is where a padding-ratio bug appears: in a
    #    batch, short texts get padded to the longest member, so dividing by the
    #    padded length instead of the token count scales them wrongly — and the
    #    per-text run above would never show it.
    print("[parity] batched, mixed lengths")
    mixed = sorted(texts, key=len)
    all_ok &= _report(
        "batch",
        _cosines(torch_svc.embed_passages(mixed), onnx_svc.embed_passages(mixed)),
    )

    # 3. Genuinely longer than max_seq_length, so truncation is actually
    #    exercised. Built by repetition until the token count is comfortably past
    #    512 — an earlier version just joined the sample texts and produced ~60
    #    tokens while printing "> 512 tokens", which is worse than not testing it:
    #    it reported a pass for something it never ran.
    long_text = " ".join(texts)
    while len(long_text) < 4000:            # ~4 chars/token, so ~1000 tokens
        long_text += " " + " ".join(texts)
    approx_tokens = len(long_text) // 4
    print(f"[parity] long input ({len(long_text)} chars, ~{approx_tokens} tokens, "
          f"truncated at {512})")
    all_ok &= _report(
        "long",
        _cosines([torch_svc.embed_passage(long_text)],
                 [onnx_svc.embed_passage(long_text)]),
    )

    # 4. Queries carry a different prefix, so they exercise a different first
    #    token — cheap to check, and the prefix is the thing embedding.py's
    #    docstring warns about most loudly.
    print("[parity] queries")
    queries = texts[: min(20, len(texts))]
    all_ok &= _report(
        "query",
        _cosines([torch_svc.embed_query(q) for q in queries],
                 [onnx_svc.embed_query(q) for q in queries]),
    )

    # 5. Retrieval order — what a user would actually notice. Vectors can agree
    #    to 0.9999 and still swap two near-tied neighbours; this checks the thing
    #    that matters rather than the proxy for it.
    print("[parity] top-5 retrieval order")
    corpus = [torch_svc.embed_passage(t) for t in texts]
    reshuffled = 0
    for q in queries:
        tq = np.asarray(torch_svc.embed_query(q))
        oq = np.asarray(onnx_svc.embed_query(q))
        mat = np.asarray(corpus)
        top_t = np.argsort(-(mat @ tq))[:5]
        top_o = np.argsort(-(mat @ oq))[:5]
        if list(top_t) != list(top_o):
            reshuffled += 1
    order_ok = reshuffled == 0
    print(f"  [{'PASS' if order_ok else 'FAIL'}] order: {reshuffled}/{len(queries)} "
          f"queries changed their top-5")
    all_ok &= order_ok

    print()
    if all_ok:
        print("[parity] ALL PASS — safe to set EMBEDDING_BACKEND=onnx")
        return 0
    print("[parity] FAILED — do NOT switch. Check _masked_mean/_l2_normalize in "
          "shared/onnx_backend.py against modules.json (mean pooling + Normalize).")
    return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
