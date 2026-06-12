"""YouTube transcript ingestion → pgvector.

Usage:
    python -m langgraph_agents.tools.youtube_ingest \
        --url "https://www.youtube.com/watch?v=VIDEO_ID" \
        --title "Bài tập đau lưng"
"""
import argparse
import asyncio
from urllib.parse import urlparse, parse_qs

from youtube_transcript_api import YouTubeTranscriptApi

from langgraph_agents.shared import get_pg_client, get_embedding_service
from langgraph_agents.db.vector_backend import VectorBackend

_CHUNK_SIZE   = 500
_CHUNK_OVERLAP = 50


def _extract_video_id(url: str) -> str:
    """Extract video ID from youtube.com/watch?v= or youtu.be/ URLs."""
    parsed = urlparse(url)
    if parsed.netloc in ("youtu.be",):
        return parsed.path.lstrip("/")
    qs = parse_qs(parsed.query)
    ids = qs.get("v", [])
    if not ids:
        raise ValueError(f"Cannot extract video ID from: {url}")
    return ids[0]


def _chunk_text(text: str, size: int = _CHUNK_SIZE, overlap: int = _CHUNK_OVERLAP) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


async def ingest_youtube(url: str, title: str = "", channel_url: str = "") -> int:
    """Ingest YouTube video transcript into pgvector. Returns number of chunks inserted."""
    video_id = _extract_video_id(url)

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["vi", "en"])
    except Exception as exc:
        raise RuntimeError(f"Cannot fetch transcript for {video_id}: {exc}") from exc

    full_text = " ".join(entry["text"] for entry in transcript)
    chunks    = _chunk_text(full_text)

    pg  = get_pg_client()
    svc = get_embedding_service()
    vb  = VectorBackend(pg)

    inserted = 0
    for idx, chunk in enumerate(chunks):
        embedding = await asyncio.to_thread(svc.embed_texts, chunk)
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]

        metadata = {
            "youtube_id":   video_id,
            "title":        title or video_id,
            "channel_url":  channel_url,
            "chunk_index":  idx,
            "total_chunks": len(chunks),
        }
        await vb.insert(
            content=chunk,
            embedding=embedding,
            source_type="youtube",
            source_id=video_id,
            metadata=metadata,
        )
        inserted += 1

    return inserted


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url",     required=True)
    parser.add_argument("--title",   default="")
    parser.add_argument("--channel", default="")
    args = parser.parse_args()

    n = await ingest_youtube(args.url, title=args.title, channel_url=args.channel)
    print(f"Inserted {n} chunks for {args.url}")


if __name__ == "__main__":
    asyncio.run(main())
