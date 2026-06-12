"""Cleanup VieNeu TTS audio files older than TTL.

Usage:
    python scripts/cleanup_tts_audio.py               # delete files older than 1 hour (default)
    python scripts/cleanup_tts_audio.py --ttl 3600    # explicit TTL in seconds
    python scripts/cleanup_tts_audio.py --dry-run     # preview only

Run periodically via Task Scheduler (Windows) or cron (Linux):
    # Linux cron — every 30 minutes
    */30 * * * * /path/to/venv/bin/python /path/to/scripts/cleanup_tts_audio.py

    # Windows Task Scheduler: trigger = every 30 min, action = python cleanup_tts_audio.py
"""
import argparse
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl",     type=int, default=3600, help="Max age in seconds (default 3600)")
    parser.add_argument("--dir",     type=str, default=None, help="Audio dir (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dir:
        audio_dir = Path(args.dir)
    else:
        script_dir = Path(__file__).resolve().parent
        audio_dir  = script_dir.parent / "agenticRAG" / "agentic_rag_gemini" / \
                     "langgraph_agents" / "services" / "vieneu_tts" / "outputs"

    if not audio_dir.exists():
        print(f"Audio dir not found: {audio_dir}")
        return

    cutoff = time.time() - args.ttl
    deleted = 0
    freed   = 0

    for f in audio_dir.glob("*.wav"):
        if f.stat().st_mtime < cutoff:
            size = f.stat().st_size
            if args.dry_run:
                print(f"[dry-run] Would delete: {f.name} ({size // 1024} KB)")
            else:
                f.unlink()
                deleted += 1
                freed   += size

    if args.dry_run:
        print("Dry run complete.")
    else:
        print(f"Deleted {deleted} file(s), freed {freed // 1024 // 1024:.1f} MB")


if __name__ == "__main__":
    main()
