"""Upload storage and a single background video-processing worker.

One job runs at a time so CPU inference cannot starve the Flask dashboard.  The
worker thread only writes short-lived SQLite connections, and progress is
persisted so the browser can poll without blocking a request handler.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage

import storage
import runtime_config
from . import sample_library
from .video_engine import VideoProcessingError, VideoSettings, process_video

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_UPLOAD_DIR = runtime_config.VIDEO_UPLOAD_DIR
DEFAULT_OUTPUT_DIR = runtime_config.VIDEO_OUTPUT_DIR
UPLOAD_ID_RE = re.compile(r"^[a-f0-9]{24}$")
JOB_ID_RE = re.compile(r"^[a-f0-9]{32}$")
MAX_VIDEO_BYTES = 256 * 1024 * 1024
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ALLOWED_VIDEO_MIME_PREFIXES = ("video/",)
ALLOWED_VIDEO_MIME_EXACT = {"application/octet-stream", "application/mp4", ""}

logger = logging.getLogger(__name__)


class UploadError(ValueError):
    """User-facing upload validation failure."""


# ---------------------------------------------------------------------------
# Configurable directories
# ---------------------------------------------------------------------------


def upload_root() -> Path:
    override = os.environ.get("SITEGUARD_VIDEO_UPLOAD_DIR")
    return Path(override).expanduser() if override else DEFAULT_UPLOAD_DIR


def output_root() -> Path:
    override = os.environ.get("SITEGUARD_VIDEO_OUTPUT_DIR")
    return Path(override).expanduser() if override else DEFAULT_OUTPUT_DIR


def _ensure_inside(base: Path, candidate: Path) -> Path:
    base = base.resolve()
    resolved = candidate.resolve()
    if not resolved.is_relative_to(base):
        raise UploadError("That path is outside the video workspace.")
    return resolved


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Uploads
# ---------------------------------------------------------------------------


def validate_video_mime(mimetype: str | None) -> bool:
    value = (mimetype or "").strip().lower()
    if value in ALLOWED_VIDEO_MIME_EXACT:
        return True
    return any(value.startswith(prefix) for prefix in ALLOWED_VIDEO_MIME_PREFIXES)


def save_upload(uploaded: FileStorage, original_filename: str) -> dict[str, Any]:
    """Persist an uploaded video in an isolated directory and probe it.

    Extension, MIME, and actual decodability are all checked; a failed upload
    is removed before the error reaches the client.
    """
    suffix = Path(original_filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_EXTENSIONS:
        raise UploadError(
            "Unsupported video type. Choose an MP4, AVI, MOV, MKV, or WEBM file."
        )

    upload_id = uuid.uuid4().hex[:24]
    directory = upload_root() / upload_id
    directory.mkdir(parents=True, exist_ok=False)

    try:
        target = directory / f"original{suffix}"
        uploaded.save(target)
        size = target.stat().st_size
        if size <= 0:
            raise UploadError("The uploaded video is empty.")
        if size > MAX_VIDEO_BYTES:
            raise UploadError("That video is larger than 256 MB.")

        probe = sample_library.probe_video(target)
        if not probe.get("ok"):
            raise UploadError(
                probe.get("error")
                or "That file is not a readable video. It may be corrupt or use an unsupported codec."
            )

        meta = {
            "upload_id": upload_id,
            "filename": original_filename,
            "stored_name": target.name,
            "size_bytes": size,
            "created_at": _now(),
            "probe": probe,
        }
        (directory / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {**meta, "path": str(target)}
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise


def resolve_upload(upload_id: str) -> dict[str, Any]:
    if not isinstance(upload_id, str) or not UPLOAD_ID_RE.match(upload_id):
        raise UploadError("That uploaded video reference is not valid.")
    base = upload_root()
    directory = _ensure_inside(base, base / upload_id)
    meta_path = directory / "meta.json"
    if not meta_path.is_file():
        raise UploadError("That uploaded video was not found. Upload it again.")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UploadError("The uploaded video metadata could not be read.") from exc
    if not isinstance(meta, dict) or not meta.get("stored_name"):
        raise UploadError("The uploaded video metadata is incomplete.")
    target = _ensure_inside(directory, directory / str(meta["stored_name"]))
    if not target.is_file():
        raise UploadError("The uploaded video file is missing. Upload it again.")
    return {**meta, "upload_id": upload_id, "path": str(target)}


def delete_upload(upload_id: str) -> None:
    try:
        directory = resolve_upload(upload_id).get("path")
        if directory:
            shutil.rmtree(Path(directory).parent, ignore_errors=True)
    except UploadError:
        return


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------

_queue: "queue.Queue[str]" = queue.Queue()
_worker_lock = threading.Lock()
_worker_started = False


def ensure_worker() -> None:
    """Start the single worker thread once, failing abandoned jobs first."""
    global _worker_started
    with _worker_lock:
        if _worker_started:
            return
        try:
            storage.fail_stale_video_jobs()
        except Exception:  # noqa: BLE001 - never block job creation
            logger.warning("Could not clear stale video jobs", exc_info=True)
        thread = threading.Thread(
            target=_worker_loop, name="siteguard-video-worker", daemon=True
        )
        thread.start()
        _worker_started = True


def enqueue_new_job(
    *,
    source_type: str,
    original_filename: str,
    source_id: str | None = None,
    stored_path: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ensure_worker()
    job_id = uuid.uuid4().hex
    job = storage.create_video_job(
        job_id,
        source_type=source_type,
        original_filename=original_filename,
        source_id=source_id,
        stored_path=stored_path,
        settings=settings or {},
    )
    _queue.put(job_id)
    return job


def _worker_loop() -> None:
    while True:
        job_id = _queue.get()
        try:
            run_job(job_id)
        except Exception:  # noqa: BLE001 - a failed job must not kill the worker
            logger.exception("Unhandled video job failure for %s", job_id)
        finally:
            _queue.task_done()


def _resolve_source(job: dict[str, Any]) -> Path:
    source_type = job.get("source_type")
    if source_type == "sample":
        sample = sample_library.get_sample(str(job.get("source_id") or ""))
        if sample is None:
            raise VideoProcessingError("The selected sample video is no longer available.")
        return sample_library.resolve_sample_path(sample["id"])
    if source_type == "upload":
        return Path(resolve_upload(str(job.get("source_id") or ""))["path"])
    raise VideoProcessingError("Unknown video source type.")


def run_job(job_id: str) -> None:
    job = storage.get_video_job(job_id)
    if job is None or job.get("status") != "queued":
        return

    storage.update_video_job(
        job_id,
        status="running",
        started_at=_now(),
        progress=0,
        message="Starting video analysis…",
    )

    progress_lock = threading.Lock()
    last = {"percent": -1.0, "processed": 0}

    def progress(percent: float, processed: int, total: int, message: str) -> None:
        with progress_lock:
            # Persist at most every 2% so the database is not hammered.
            if percent < 100 and (percent - last["percent"]) < 2 and processed == last["processed"]:
                return
            last["percent"] = percent
            last["processed"] = processed
        storage.update_video_job(
            job_id,
            progress=round(min(99.0, max(0.0, percent)), 1),
            processed_frames=int(processed),
            sampled_frames=int(total),
            message=message,
        )

    try:
        source = _resolve_source(job)
        settings = VideoSettings.from_payload(job.get("settings") or {})
        output = output_root() / f"{job_id}.mp4"
        result = process_video(source, output, settings, progress)

        summary = result.get("summary") or {}
        video = result.get("video") or {}
        processing = result.get("processing") or {}
        storage.finish_video_job(
            job_id,
            result=result,
            events=result.get("events") or [],
            output_rel=output.name,
            summary=summary,
            duration_s=video.get("duration_s"),
            width=video.get("width"),
            height=video.get("height"),
            fps=video.get("fps"),
            total_frames=video.get("frame_count"),
            sampled_frames=processing.get("estimated_frames"),
            processed_frames=processing.get("processed_frames"),
        )
    except (UploadError, VideoProcessingError) as exc:
        logger.warning("Video job %s failed: %s", job_id, exc)
        storage.update_video_job(
            job_id,
            status="failed",
            completed_at=_now(),
            error=str(exc),
            message="Video analysis failed.",
        )
    except Exception as exc:  # noqa: BLE001 - persisted for the UI
        logger.exception("Video job %s crashed", job_id)
        storage.update_video_job(
            job_id,
            status="failed",
            completed_at=_now(),
            error=f"{type(exc).__name__}: {exc}",
            message="Video analysis failed.",
        )


def job_output_path(job: dict[str, Any]) -> Path | None:
    """Resolve a job's annotated output inside the allowlisted output folder."""
    name = job.get("output_rel")
    if not name or Path(str(name)).name != str(name):
        return None
    base = output_root()
    candidate = _ensure_inside(base, base / str(name))
    return candidate if candidate.is_file() else None
