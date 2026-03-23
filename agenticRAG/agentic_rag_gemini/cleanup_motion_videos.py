"""Manual cleanup utility for old generated motion artifacts.

Usage:
  python cleanup_motion_videos.py
  python cleanup_motion_videos.py --root ./static/videos --retention-hours 12
"""

import argparse
import time
from pathlib import Path


def cleanup(root: Path, retention_hours: int) -> int:
    root.mkdir(parents=True, exist_ok=True)

    threshold = time.time() - (retention_hours * 3600)
    deleted = 0
    patterns = ["job_*.mp4", "motion_*.mp4", "*.npz"]

    for pattern in patterns:
        for path in root.glob(pattern):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                if path.stat().st_mtime < threshold:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except OSError:
                pass

    return deleted


def main() -> int:
    parser = argparse.ArgumentParser(description="Cleanup old motion artifacts")
    parser.add_argument("--root", default="./static/videos", help="Motion artifact directory")
    parser.add_argument("--retention-hours", type=int, default=6, help="Retention window in hours")
    args = parser.parse_args()

    deleted = cleanup(Path(args.root), args.retention_hours)
    print(f"deleted={deleted} retention_hours={args.retention_hours} root={args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
