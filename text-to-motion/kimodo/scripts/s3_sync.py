#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Download an S3 prefix into a local directory — a stand-in for `aws s3 sync`.

The production image no longer ships the AWS CLI v2 (~220 MB unpacked) just to run the
three sync calls in `inference-entrypoint.sh`. boto3 is already a kimodo dependency, so
this covers the one direction we need: S3 -> local, skip what is already there.

Usage:
    python s3_sync.py s3://bucket/some/prefix/ /local/dir

Differences from `aws s3 sync` (deliberate, and sufficient for our use):
  - one direction only (download); it never uploads or deletes
  - "already there" means same size AND local mtime >= S3 LastModified, which is the
    same test the CLI applies by default

Startup latency matters here (this runs before the MCP server binds), so files are
fetched concurrently and boto3's own multipart threads handle the large checkpoints.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

# Fetch this many objects at once. The text-encoder prefix is a handful of multi-GB
# safetensors shards; the skeleton prefix is a few KB-sized files. 8 keeps both busy
# without thrashing a 4-vCPU task.
MAX_CONCURRENT_FILES = 8

# Per-file multipart settings. boto3 spawns its own threads per download, so keep this
# modest to avoid MAX_CONCURRENT_FILES * max_concurrency runaway.
TRANSFER_CONFIG = TransferConfig(
    multipart_threshold=16 * 1024 * 1024,
    multipart_chunksize=16 * 1024 * 1024,
    max_concurrency=4,
)

# botocore's default pool is 10; with 8 files x 4 parts we would starve it and stall.
BOTO_CONFIG = Config(max_pool_connections=MAX_CONCURRENT_FILES * 4 + 8, retries={"max_attempts": 5, "mode": "standard"})


def parse_s3_uri(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise SystemExit(f"ERROR: not an s3:// URI: {uri}")
    return parsed.netloc, parsed.path.lstrip("/")


def needs_download(local_path: Path, size: int, last_modified) -> bool:
    if not local_path.exists():
        return True
    stat = local_path.stat()
    if stat.st_size != size:
        return True
    # S3 timestamps are tz-aware; compare as epoch seconds.
    return stat.st_mtime < last_modified.timestamp()


def main() -> int:
    ap = argparse.ArgumentParser(description="Download an S3 prefix into a local directory.")
    ap.add_argument("source", help="s3://bucket/prefix/")
    ap.add_argument("dest", help="local directory")
    args = ap.parse_args()

    bucket, prefix = parse_s3_uri(args.source)
    dest_root = Path(args.dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    client = boto3.client("s3", config=BOTO_CONFIG)

    pending: list[tuple[str, Path, int]] = []
    skipped = 0
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/"):  # directory marker
                continue
            relative = key[len(prefix):].lstrip("/") if prefix else key
            if not relative:  # the prefix itself names a single object
                relative = Path(key).name
            local_path = dest_root / relative
            if needs_download(local_path, obj["Size"], obj["LastModified"]):
                pending.append((key, local_path, obj["Size"]))
            else:
                skipped += 1

    if not pending:
        print(f"[s3_sync] {args.source} -> {dest_root}: up to date ({skipped} files)")
        return 0

    total_bytes = sum(size for _, _, size in pending)
    print(
        f"[s3_sync] {args.source} -> {dest_root}: "
        f"{len(pending)} file(s), {total_bytes / 1e6:.1f} MB to download ({skipped} already current)"
    )

    started = time.monotonic()
    failures: list[tuple[str, BaseException]] = []

    def download(key: str, local_path: Path) -> str:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        # Download to a temp name so an interrupted task never leaves a truncated file
        # that the size check would later accept as complete.
        tmp_path = local_path.with_name(local_path.name + ".part")
        client.download_file(bucket, key, str(tmp_path), Config=TRANSFER_CONFIG)
        os.replace(tmp_path, local_path)
        return key

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_FILES) as pool:
        futures = {pool.submit(download, key, path): key for key, path, _ in pending}
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except BaseException as exc:  # noqa: BLE001 - report every failure, not just the first
                failures.append((key, exc))
                print(f"[s3_sync] FAILED {key}: {exc}", file=sys.stderr)

    elapsed = time.monotonic() - started
    if failures:
        print(f"[s3_sync] {len(failures)} of {len(pending)} download(s) failed", file=sys.stderr)
        return 1

    rate = (total_bytes / 1e6 / elapsed) if elapsed > 0 else 0.0
    print(f"[s3_sync] done in {elapsed:.1f}s ({rate:.1f} MB/s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
