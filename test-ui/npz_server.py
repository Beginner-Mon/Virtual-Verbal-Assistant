#!/usr/bin/env python3
"""
Serve the test-ui folder and expose lightweight NPZ APIs for motion playback.

Usage:
  python npz_server.py --npz "..\\text-to-motion\\DART\\data\\outputs\\motion_xxx.npz" --port 8090
"""

from __future__ import annotations

import argparse
import json
import os
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_npz = here.parent / "text-to-motion" / "DART" / "data" / "outputs" / (
        "motion_b06121e2-a44e-4aeb-ba09-ad030d9f1d9f.npz"
    )
    parser = argparse.ArgumentParser(description="NPZ test UI server")
    parser.add_argument("--npz", default=str(default_npz), help="Path to .npz motion file")
    parser.add_argument("--port", type=int, default=8090, help="Server port")
    return parser.parse_args()


def build_motion_model(npz_path: Path) -> dict:
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=False)
    required = {"motion", "poses_6d", "transl", "betas"}
    missing = required - set(data.files)
    if missing:
        raise ValueError(f"Missing required arrays: {sorted(missing)}")

    motion = data["motion"].astype(np.float32, copy=False)
    poses_6d = data["poses_6d"].astype(np.float32, copy=False)
    transl = data["transl"].astype(np.float32, copy=False)
    betas = data["betas"].astype(np.float32, copy=False)

    return {
        "path": str(npz_path),
        "motion": motion,
        "poses_6d": poses_6d,
        "transl": transl,
        "betas": betas,
        "num_frames": int(motion.shape[0]),
        "fps": 30,
    }


class NpzHandler(SimpleHTTPRequestHandler):
    model: dict = {}

    def _send_json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _handle_summary(self) -> None:
        model = self.model
        summary = {
            "file": model["path"],
            "num_frames": model["num_frames"],
            "fps": model["fps"],
            "duration_seconds": round(model["num_frames"] / model["fps"], 3),
            "shapes": {
                "motion": list(model["motion"].shape),
                "poses_6d": list(model["poses_6d"].shape),
                "transl": list(model["transl"].shape),
                "betas": list(model["betas"].shape),
            },
            "betas": [round(float(x), 6) for x in model["betas"].tolist()],
        }
        self._send_json(summary)

    def _handle_frame(self, query: dict) -> None:
        model = self.model
        frame_idx = int(query.get("i", ["0"])[0])
        frame_idx = max(0, min(frame_idx, model["num_frames"] - 1))

        transl = model["transl"][frame_idx]
        poses = model["poses_6d"][frame_idx]
        motion = model["motion"][frame_idx]

        payload = {
            "frame": frame_idx,
            "time_seconds": round(frame_idx / model["fps"], 3),
            "transl": [round(float(v), 6) for v in transl.tolist()],
            "poses_6d_head": [round(float(v), 6) for v in poses[:12].tolist()],
            "motion_head": [round(float(v), 6) for v in motion[:12].tolist()],
        }
        self._send_json(payload)

    def _handle_curve(self) -> None:
        transl = self.model["transl"]
        payload = {
            "x": [round(float(v), 6) for v in transl[:, 0].tolist()],
            "y": [round(float(v), 6) for v in transl[:, 1].tolist()],
            "z": [round(float(v), 6) for v in transl[:, 2].tolist()],
        }
        self._send_json(payload)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/npz/summary":
            return self._handle_summary()
        if parsed.path == "/api/npz/frame":
            return self._handle_frame(parse_qs(parsed.query))
        if parsed.path == "/api/npz/curve":
            return self._handle_curve()
        return super().do_GET()


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    npz_path = Path(args.npz).resolve()
    model = build_motion_model(npz_path)
    NpzHandler.model = model

    server = ThreadingHTTPServer(("127.0.0.1", args.port), NpzHandler)
    print(f"Serving test-ui at http://127.0.0.1:{args.port}")
    print(f"NPZ file: {npz_path}")
    print("Endpoints: /api/npz/summary, /api/npz/frame?i=0, /api/npz/curve")
    server.serve_forever()


if __name__ == "__main__":
    main()
