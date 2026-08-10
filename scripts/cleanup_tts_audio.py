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
import sys
import time
from pathlib import Path

# Where the WAV files actually are.
#
# This script used to default to
# `agenticRAG/agentic_rag_gemini/langgraph_agents/services/vieneu_tts/outputs`,
# a path that has never existed — `langgraph_agents` is a sibling of
# `agentic_rag_gemini`, not a child of it, and the TTS files are not written by
# that package at all. The script found no directory, printed one line, and
# exited 0. It has therefore never deleted anything, and 86 WAVs had accumulated
# by the time anyone checked.
#
# The files come from SpeechLLm's TTS clients, whose `output_dir` config
# defaults to `data/temp_audio` relative to the SpeechLLm root.
DEFAULT_AUDIO_DIR = Path(__file__).resolve().parents[1] / "SpeechLLm" / "data" / "temp_audio"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ttl",     type=int, default=3600, help="Max age in seconds (default 3600)")
    parser.add_argument("--dir",     type=str, default=None, help="Audio dir (default: auto-detect)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    audio_dir = Path(args.dir) if args.dir else DEFAULT_AUDIO_DIR

    if not audio_dir.exists():
        # Exit non-zero. Returning 0 here is what let a scheduled job report
        # success every 30 minutes for months while deleting nothing.
        print(f"ERROR: audio dir not found: {audio_dir}", file=sys.stderr)
        print("       Pass --dir if the TTS service writes somewhere else.", file=sys.stderr)
        return 1

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
        # Was `freed // 1024 // 1024:.1f` — integer division first, so anything
        # under a megabyte reported "0.0 MB".
        print(f"Deleted {deleted} file(s), freed {freed / 1024 / 1024:.1f} MB")
    return 0


if __name__ == "__main__":
    # `main()` returned a status that nobody propagated, so the process always
    # exited 0 — including on the missing-directory path above.
    raise SystemExit(main())
