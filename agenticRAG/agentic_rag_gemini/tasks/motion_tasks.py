"""Celery consumer tasks for async motion rendering."""

import os
import re
import shutil
import subprocess
import tempfile
import uuid
import time
from functools import lru_cache
from importlib import import_module
from typing import Any, Dict, Optional
from urllib.parse import urljoin

import numpy as np
import requests

from celery_app import celery_app
from agents.tools.motion_candidate_retriever import MotionCandidate, MotionCandidateRetriever
from agents.tools.motion_reranker import MotionReranker
from utils.logger import get_logger
from utils.cache_service import CacheService

logger = get_logger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}

@lru_cache(maxsize=1)
def _get_cache() -> CacheService:
    return CacheService()

DART_GENERATE_ENDPOINT = os.getenv("DART_GENERATE_ENDPOINT", "http://localhost:5001/generate")
MOTION_DEFAULT_DURATION_SECONDS = float(os.getenv("MOTION_DEFAULT_DURATION_SECONDS", "12"))
RERANK_TIMEOUT_SECONDS = int(os.getenv("MOTION_RERANK_TIMEOUT_SECONDS", "8"))
MOTION_VIDEO_ROOT = os.getenv("MOTION_VIDEO_ROOT", "./static/videos")
MOTION_GLB_ROOT = os.getenv("MOTION_GLB_ROOT", "./static/motions")
API_PUBLIC_BASE_URL = os.getenv("API_PUBLIC_BASE_URL", "http://localhost:8000")
DART_REQUEST_TIMEOUT_SECONDS = int(os.getenv("MOTION_DART_TIMEOUT_SECONDS", "180"))
DIRECT_HIT_THRESHOLD = float(os.getenv("MOTION_DIRECT_HIT_THRESHOLD", "0.95"))
MOTION_GENERATE_MP4_FALLBACK = os.getenv("MOTION_GENERATE_MP4_FALLBACK", "false").strip().lower() in {
    "1", "true", "yes", "on"
}
MOTION_WORKER_WARMUP = _env_bool("MOTION_WORKER_WARMUP", True)


def _celery_task(*args, **kwargs):
    """Use Celery decorator when available, else no-op for import safety."""
    if celery_app is None:
        def _wrap(fn):
            return fn
        return _wrap
    return celery_app.task(*args, **kwargs)


def _ensure_video_root() -> str:
    os.makedirs(MOTION_VIDEO_ROOT, exist_ok=True)
    return MOTION_VIDEO_ROOT


def _ensure_glb_root() -> str:
    os.makedirs(MOTION_GLB_ROOT, exist_ok=True)
    return MOTION_GLB_ROOT


def _normalize_job_id(raw_job_id: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", (raw_job_id or "").strip())
    return cleaned or f"job_{uuid.uuid4().hex}"


@lru_cache(maxsize=1)
def _get_retriever() -> MotionCandidateRetriever:
    # Per-worker lazy singleton avoids import-time global initialization races.
    return MotionCandidateRetriever()


@lru_cache(maxsize=1)
def _get_reranker() -> MotionReranker:
    # Per-worker lazy singleton keeps model clients process-local and deterministic.
    return MotionReranker()


def _warm_worker_components() -> None:
    """Pre-load retrieval/rerank components to avoid first-request cold spikes."""
    if not MOTION_WORKER_WARMUP:
        return
    try:
        _get_retriever()
        _get_reranker()
        logger.info("Motion worker warmup completed")
    except Exception as exc:
        # Warmup failure should not block worker boot; runtime call can still retry.
        logger.warning("Motion worker warmup failed: %s", exc)


_warm_worker_components()


def _extract_npz_filename(motion_file_url: str) -> str:
    return motion_file_url.rstrip("/").split("/")[-1]


def _download_artifact(motion_file_url: str, output_path: str) -> str:
    if not motion_file_url:
        raise RuntimeError("DART response missing motion_file_url")

    if motion_file_url.startswith("/"):
        dart_base = DART_GENERATE_ENDPOINT.rsplit("/generate", 1)[0]
        download_url = f"{dart_base}{motion_file_url}"
    else:
        download_url = motion_file_url

    resp = requests.get(download_url, timeout=DART_REQUEST_TIMEOUT_SECONDS)
    resp.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(resp.content)

    return download_url


def _download_npz(dart_payload: Dict[str, Any], tmp_npz_path: str) -> str:
    motion_file_url = dart_payload.get("motion_file_url")
    return _download_artifact(motion_file_url, tmp_npz_path)


def _render_npz_to_mp4(npz_path: str, output_mp4_path: str, fps: int = 30) -> None:
    """Render an MP4 from DART npz using trajectory visualization + ffmpeg."""
    try:
        matplotlib = import_module("matplotlib")
        matplotlib.use("Agg")
        plt = import_module("matplotlib.pyplot")
    except Exception as exc:
        raise RuntimeError(f"matplotlib is required for mp4 rendering: {exc}") from exc

    data = np.load(npz_path, allow_pickle=False)

    transl = None
    for key in ("trans", "transl"):
        if key in data:
            transl = data[key]
            break
    if transl is None:
        raise RuntimeError("NPZ does not contain trans/transl for rendering")

    transl = np.asarray(transl)
    if transl.ndim != 2 or transl.shape[0] == 0:
        raise RuntimeError("Invalid translation shape in NPZ")

    if transl.shape[1] >= 3:
        x = transl[:, 0]
        z = transl[:, 2]
    elif transl.shape[1] == 2:
        x = transl[:, 0]
        z = transl[:, 1]
    else:
        x = transl[:, 0]
        z = np.zeros_like(x)

    frames_dir = tempfile.mkdtemp(prefix="motion_frames_")
    try:
        xmin, xmax = float(np.min(x)), float(np.max(x))
        zmin, zmax = float(np.min(z)), float(np.max(z))
        pad_x = max(0.25, (xmax - xmin) * 0.1)
        pad_z = max(0.25, (zmax - zmin) * 0.1)

        for i in range(len(x)):
            fig, ax = plt.subplots(figsize=(8, 6), dpi=100)
            ax.plot(x[: i + 1], z[: i + 1], color="#1f77b4", linewidth=2.0, label="trajectory")
            ax.scatter([x[i]], [z[i]], color="#d62728", s=60, label="current")
            ax.set_xlim(xmin - pad_x, xmax + pad_x)
            ax.set_ylim(zmin - pad_z, zmax + pad_z)
            ax.set_title("DART Motion Preview")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper right")
            ax.text(0.02, 0.95, f"Frame {i + 1}/{len(x)}", transform=ax.transAxes, va="top")
            frame_path = os.path.join(frames_dir, f"frame_{i:06d}.png")
            fig.savefig(frame_path)
            plt.close(fig)

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg not found in PATH")

        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(max(1, int(fps))),
            "-i",
            os.path.join(frames_dir, "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            output_mp4_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")
    finally:
        shutil.rmtree(frames_dir, ignore_errors=True)


def _public_video_url(relative_path: str) -> str:
    return urljoin(API_PUBLIC_BASE_URL.rstrip("/") + "/", relative_path.lstrip("/"))


@_celery_task(name="tasks.motion_tasks.render_motion_job", bind=True)
def render_motion_job(
    self,
    user_query: str,
    motion_prompt: str,
    user_id: str,
    duration_seconds: float,
    enqueue_epoch_ms: Optional[int] = None,
    timeline_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Render motion in worker node with Top-K retrieval + rerank + fallback."""
    raw_job_id = getattr(getattr(self, "request", None), "id", None) or ""
    job_id = _normalize_job_id(raw_job_id)
    worker_started_epoch_ms = int(time.time() * 1000)
    timeline = timeline_id or f"tl_{uuid.uuid4().hex[:12]}"
    timings_ms: Dict[str, float] = {}
    worker_start_perf = time.perf_counter()

    if isinstance(enqueue_epoch_ms, int):
        timings_ms["queue_wait_ms"] = round(max(0.0, float(worker_started_epoch_ms - enqueue_epoch_ms)), 3)

    effective_duration = duration_seconds or MOTION_DEFAULT_DURATION_SECONDS
    video_root = _ensure_video_root()
    glb_root = _ensure_glb_root()
    output_filename = f"job_{job_id}.mp4"
    output_glb_filename = f"job_{job_id}.glb"
    output_mp4_path = os.path.join(video_root, output_filename)
    output_glb_path = os.path.join(glb_root, output_glb_filename)
    relative_video_path = f"/static/videos/{output_filename}"
    relative_glb_path = f"/static/motions/{output_glb_filename}"
    generate_mp4_fallback = bool(MOTION_GENERATE_MP4_FALLBACK)
    timings_ms["mp4_enabled"] = 1.0 if generate_mp4_fallback else 0.0

    def _build_meta(stage_name: str, error: Optional[str] = None) -> Dict[str, Any]:
        meta: Dict[str, Any] = {
            "stage": stage_name,
            "timeline_id": timeline,
            "enqueue_epoch_ms": enqueue_epoch_ms,
            "worker_started_epoch_ms": worker_started_epoch_ms,
            "timings_ms": dict(timings_ms),
        }
        if error:
            meta["error"] = error
        return meta

    stage = "candidate_retrieval"
    self.update_state(state="STARTED", meta=_build_meta(stage))

    query_for_selection = (user_query or motion_prompt or "").strip()
    retrieval_start_perf = time.perf_counter()
    try:
        candidates = _get_retriever().retrieve_top_k(query_for_selection, k=5)
    except Exception as exc:
        logger.warning("Candidate retrieval failed, fallback to raw query: %s", exc)
        candidates = [MotionCandidate("fallback", query_for_selection, query_for_selection, 0.0)]
    timings_ms["candidate_retrieval_ms"] = round((time.perf_counter() - retrieval_start_perf) * 1000, 3)

    top1 = candidates[0]

    # ── Stage 2: Decision Gate — Direct Hit vs. Semantic Bridge ──
    rewritten_prompt = None
    selected_strategy = "unknown"

    if top1.score > DIRECT_HIT_THRESHOLD:
        # ── FAST PATH: Direct Hit ──────────────────────────────
        # High similarity means the matched HumanML3D description is already
        # in native T2M vocabulary — no need for Gemini rewrite.
        rewritten_prompt = top1.motion_prompt
        selected_strategy = "direct_hit"
        logger.info(
            "DIRECT HIT (score=%.3f > %.2f): skipping Semantic Bridge, "
            "using matched description directly: '%s'",
            top1.score, DIRECT_HIT_THRESHOLD, rewritten_prompt[:80],
        )
        timings_ms["semantic_bridge_ms"] = 0.0
        self.update_state(state="STARTED", meta=_build_meta("direct_hit"))
    else:
        # ── SLOW PATH: Semantic Bridge — rewrite via Gemini ────
        stage = "semantic_bridge"
        self.update_state(state="STARTED", meta=_build_meta(stage))

        reference_captions = [c.text_description for c in candidates]
        bridge_start_perf = time.perf_counter()
        try:
            rewritten_prompt = _get_reranker().rewrite_prompt(
                user_query=query_for_selection,
                reference_captions=reference_captions,
                timeout_seconds=RERANK_TIMEOUT_SECONDS,
            )
            selected_strategy = "semantic_bridge"
        except Exception as exc:
            logger.warning("Semantic Bridge failed, falling back to top-1 candidate: %s", exc)
        timings_ms["semantic_bridge_ms"] = round((time.perf_counter() - bridge_start_perf) * 1000, 3)

        # Fallback: if rewrite failed, use the best candidate's motion_prompt
        if not rewritten_prompt:
            rewritten_prompt = top1.motion_prompt
            selected_strategy = "embedding_fallback"

    logger.info(
        "Motion path: strategy=%s, prompt='%s'",
        selected_strategy, rewritten_prompt[:80],
    )

    body = {
        "text_prompt": rewritten_prompt,
        "duration_seconds": effective_duration,
        "guidance_scale": 5.0,
        "num_steps": 25,
    }

    seed_value = int.from_bytes(os.urandom(4), "big")

    cache = _get_cache()
    cached_url = cache.get_motion_result(rewritten_prompt, effective_duration, body["num_steps"])
    if cached_url:
        cached_filename = cached_url.rstrip("/").split("/")[-1]
        cached_local_path = os.path.join(video_root, cached_filename)
        if os.path.exists(cached_local_path):
            logger.info("Motion cache HIT for mp4 fallback: '%s'", rewritten_prompt[:40])

    tmp_npz_path: Optional[str] = None
    if generate_mp4_fallback:
        tmp_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".npz",
            prefix=f"motion_{job_id}_",
            delete=False,
        )
        tmp_npz_path = tmp_handle.name
        tmp_handle.close()

    try:
        stage = "dart_generate_glb"
        self.update_state(state="STARTED", meta=_build_meta(stage))
        glb_body = {**body, "seed": seed_value, "output_format": "glb"}
        dart_glb_start_perf = time.perf_counter()
        glb_resp = requests.post(DART_GENERATE_ENDPOINT, json=glb_body, timeout=DART_REQUEST_TIMEOUT_SECONDS)
        glb_resp.raise_for_status()
        timings_ms["dart_generate_glb_ms"] = round((time.perf_counter() - dart_glb_start_perf) * 1000, 3)
        glb_payload = glb_resp.json()

        motion_fps = int(glb_payload.get("fps", 30) or 30)
        motion_frames = int(glb_payload.get("num_frames", max(1, round(effective_duration * motion_fps))) or max(1, round(effective_duration * motion_fps)))
        motion_duration_seconds = float(glb_payload.get("duration_seconds", effective_duration) or effective_duration)

        stage = "glb_download"
        self.update_state(state="STARTED", meta=_build_meta(stage))
        glb_download_start_perf = time.perf_counter()
        _download_artifact(glb_payload.get("motion_file_url"), output_glb_path)
        timings_ms["glb_download_ms"] = round((time.perf_counter() - glb_download_start_perf) * 1000, 3)

        if not os.path.exists(output_glb_path):
            raise RuntimeError("GLB download completed but output glb was not created")

        # Return relative path so API gateway can project correct host (localhost/ngrok).
        motion_file_url = relative_glb_path

        video_url = None
        if generate_mp4_fallback:
            try:
                stage = "dart_generate_npz"
                self.update_state(state="STARTED", meta=_build_meta(stage))
                npz_body = {**body, "seed": seed_value, "output_format": "npz"}
                dart_npz_start_perf = time.perf_counter()
                resp = requests.post(DART_GENERATE_ENDPOINT, json=npz_body, timeout=DART_REQUEST_TIMEOUT_SECONDS)
                resp.raise_for_status()
                timings_ms["dart_generate_npz_ms"] = round((time.perf_counter() - dart_npz_start_perf) * 1000, 3)
                dart_payload = resp.json()

                motion_fps = int(dart_payload.get("fps", motion_fps) or motion_fps)
                motion_frames = int(dart_payload.get("num_frames", motion_frames) or motion_frames)
                motion_duration_seconds = float(dart_payload.get("duration_seconds", motion_duration_seconds) or motion_duration_seconds)

                stage = "npz_download"
                self.update_state(state="STARTED", meta=_build_meta(stage))
                npz_download_start_perf = time.perf_counter()
                _download_npz(dart_payload, tmp_npz_path)
                timings_ms["npz_download_ms"] = round((time.perf_counter() - npz_download_start_perf) * 1000, 3)

                stage = "mp4_render"
                self.update_state(state="STARTED", meta=_build_meta(stage))
                fps = int(dart_payload.get("fps", 30) or 30)
                mp4_render_start_perf = time.perf_counter()
                _render_npz_to_mp4(tmp_npz_path, output_mp4_path, fps=fps)
                timings_ms["mp4_render_ms"] = round((time.perf_counter() - mp4_render_start_perf) * 1000, 3)
                if os.path.exists(output_mp4_path):
                    video_url = relative_video_path
            except Exception as mp4_exc:
                logger.warning("MP4 fallback generation failed for job %s: %s", job_id, mp4_exc)
                timings_ms["mp4_render_ms"] = round(timings_ms.get("mp4_render_ms", 0.0), 3)
            finally:
                if tmp_npz_path and os.path.exists(tmp_npz_path):
                    os.remove(tmp_npz_path)
        else:
            timings_ms["dart_generate_npz_ms"] = 0.0
            timings_ms["npz_download_ms"] = 0.0
            timings_ms["mp4_render_ms"] = 0.0
            self.update_state(state="STARTED", meta=_build_meta("mp4_skipped"))

        if video_url:
            cache.set_motion_result(rewritten_prompt, effective_duration, body["num_steps"], video_url)

        worker_finished_epoch_ms = int(time.time() * 1000)
        timings_ms["worker_total_ms"] = round((time.perf_counter() - worker_start_perf) * 1000, 3)

        return {
            "status": "completed",
            "motion_file_url": motion_file_url,
            "video_url": video_url,
            "frames": motion_frames,
            "fps": motion_fps,
            "duration_seconds": motion_duration_seconds,
            "error": None,
            "job_id": job_id,
            "stage": "completed",
            "relative_motion_path": relative_glb_path,
            "relative_video_path": relative_video_path,
            "selected_strategy": selected_strategy,
            "timeline_id": timeline,
            "enqueue_epoch_ms": enqueue_epoch_ms,
            "worker_started_epoch_ms": worker_started_epoch_ms,
            "worker_finished_epoch_ms": worker_finished_epoch_ms,
            "timings_ms": timings_ms,
            "selected_candidate": {
                "candidate_id": top1.candidate_id,
                "text_description": top1.text_description,
                "rewritten_prompt": rewritten_prompt,
                "score": top1.score,
            },
        }
    except Exception as exc:
        if tmp_npz_path and os.path.exists(tmp_npz_path):
            try:
                os.remove(tmp_npz_path)
            except Exception:
                pass
        if os.path.exists(output_mp4_path):
            try:
                os.remove(output_mp4_path)
            except Exception:
                pass
        if os.path.exists(output_glb_path):
            try:
                os.remove(output_glb_path)
            except Exception:
                pass
        logger.error("Motion render job failed: %s", exc)
        timings_ms["worker_total_ms"] = round((time.perf_counter() - worker_start_perf) * 1000, 3)
        worker_finished_epoch_ms = int(time.time() * 1000)
        error_message = f"motion_render_failed stage={stage}: {exc}"
        try:
            failure_meta = _build_meta(stage, str(exc))
            failure_meta["worker_finished_epoch_ms"] = worker_finished_epoch_ms
            self.update_state(state="FAILURE", meta=failure_meta)
        except Exception:
            pass
        raise RuntimeError(error_message) from exc
