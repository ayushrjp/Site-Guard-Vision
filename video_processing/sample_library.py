"""Safe discovery of preloaded sample videos.

Sample videos live in an allowlisted directory (``test_videos`` by default).
Clients never send file paths: every sample is addressed by a stable hash ID,
and every resolved path is checked to stay inside the allowlisted directory.
Metadata and thumbnails come from OpenCV, not from filenames.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import runtime_config

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_SAMPLE_DIR = runtime_config.DEFAULT_SAMPLE_VIDEO_DIR
DEFAULT_THUMB_DIR = runtime_config.VIDEO_THUMB_DIR
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "video_yolo_test" / "video_detection_report.json"

SAMPLE_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
THUMBNAIL_WIDTH = 480

_cache_lock = threading.Lock()
_cache: dict[str, Any] = {"signature": None, "samples": []}


class SampleVideoError(RuntimeError):
    """A sample video could not be resolved or read."""


# ---------------------------------------------------------------------------
# Configurable locations (read at call time so tests can isolate them)
# ---------------------------------------------------------------------------


def sample_dir() -> Path:
    override = os.environ.get("SITEGUARD_SAMPLE_VIDEO_DIR")
    return Path(override).expanduser() if override else DEFAULT_SAMPLE_DIR


def thumbnail_dir() -> Path:
    override = os.environ.get("SITEGUARD_VIDEO_THUMB_DIR")
    return Path(override).expanduser() if override else DEFAULT_THUMB_DIR


def report_path() -> Path:
    override = os.environ.get("SITEGUARD_VIDEO_REPORT")
    return Path(override).expanduser() if override else DEFAULT_REPORT_PATH


def is_supported_video(filename: str) -> bool:
    return Path(filename).suffix.lower() in SAMPLE_VIDEO_EXTENSIONS


def video_id_for(filename: str) -> str:
    """Stable, path-free identifier derived from the bare filename."""
    return hashlib.sha1(filename.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Probing with OpenCV
# ---------------------------------------------------------------------------


def probe_video(path: str | Path) -> dict[str, Any]:
    """Read real container metadata; return ``ok=False`` for unreadable files."""
    path = Path(path)
    info: dict[str, Any] = {
        "ok": False,
        "width": 0,
        "height": 0,
        "fps": 0.0,
        "frame_count": 0,
        "duration_s": 0.0,
        "size_bytes": 0,
        "error": None,
    }
    if not path.is_file():
        info["error"] = "The video file does not exist."
        return info
    try:
        info["size_bytes"] = int(path.stat().st_size)
    except OSError:
        pass

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            info["error"] = "OpenCV could not open this video. It may be corrupt or use an unsupported codec."
            return info

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        ok, frame = capture.read()
        if not ok or frame is None:
            info["error"] = "The video contains no readable frames."
            return info
        if not width or not height:
            height, width = frame.shape[:2]
        if fps <= 0 or fps > 240:
            # Some containers omit FPS.  Do not guess a plausible-looking value
            # silently: report the file as unusable instead.
            info["error"] = "The video does not report a usable frame rate."
            return info
        if frame_count <= 0:
            frame_count = int(round(fps * 0.0))
        duration = frame_count / fps if frame_count > 0 else 0.0

        info.update(
            {
                "ok": True,
                "width": width,
                "height": height,
                "fps": round(fps, 3),
                "frame_count": frame_count,
                "duration_s": round(duration, 3),
            }
        )
        return info
    except Exception as exc:  # noqa: BLE001 - surfaced as a readable error
        info["error"] = f"{type(exc).__name__}: {exc}"
        return info
    finally:
        capture.release()


# ---------------------------------------------------------------------------
# Verified categories from the existing detection report
# ---------------------------------------------------------------------------


def _load_report() -> dict[str, dict[str, Any]]:
    path = report_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(payload, list):
        return {}
    report: dict[str, dict[str, Any]] = {}
    for item in payload:
        if isinstance(item, dict) and isinstance(item.get("video"), str):
            report[item["video"]] = item
    return report


def _category_for(filename: str, report: dict[str, dict[str, Any]]) -> dict[str, str]:
    entry = report.get(filename)
    if not entry:
        return {
            "kind": "general",
            "label": "General footage",
            "detail": "No sampled-frame report available.",
        }

    classes = [
        str(item.get("class"))
        for item in (entry.get("ppe_detections") or [])
        if isinstance(item, dict) and item.get("class")
    ]
    negatives = sorted({name for name in classes if name.upper().startswith("NO-")})
    positives = sorted(
        {
            name
            for name in classes
            if not name.upper().startswith("NO-")
            and name.lower() not in {"person", "ladder", "safety cone", "fall-detected"}
        }
    )
    if negatives:
        return {
            "kind": "violation_evidence",
            "label": "Missing PPE observed",
            "detail": "Sampled frame reported: " + ", ".join(negatives) + ".",
        }
    if positives:
        return {
            "kind": "compliant_evidence",
            "label": "PPE observed",
            "detail": "Sampled frame reported: " + ", ".join(positives) + ".",
        }
    return {
        "kind": "general",
        "label": "General footage",
        "detail": "No PPE evidence in the existing sampled-frame report.",
    }


# ---------------------------------------------------------------------------
# Thumbnails
# ---------------------------------------------------------------------------


def thumbnail_path(video_id: str) -> Path:
    return thumbnail_dir() / f"{video_id}.jpg"


def placeholder_thumbnail(video_id: str, filename: str) -> Path:
    """Create a cached neutral preview when a frame cannot be decoded."""
    target = thumbnail_dir() / f"{video_id}_placeholder.jpg"
    try:
        if target.is_file():
            return target
    except OSError:
        pass

    target.parent.mkdir(parents=True, exist_ok=True)
    image = np.zeros((270, THUMBNAIL_WIDTH, 3), dtype=np.uint8)
    image[:] = (44, 27, 18)  # navy (BGR), close to the dashboard panel
    cv2.putText(
        image,
        "Video preview unavailable",
        (18, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (245, 247, 251),
        1,
        cv2.LINE_AA,
    )
    label = filename if len(filename) <= 44 else filename[:41] + "..."
    cv2.putText(
        image,
        label,
        (18, 162),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (216, 201, 32),
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(target), image, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    return target


def ensure_thumbnail(sample: dict[str, Any]) -> Path | None:
    """Create a cached JPEG thumbnail; return None when the frame is unreadable."""
    source = Path(sample["path"])
    target = thumbnail_path(sample["id"])
    try:
        if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
            return target
    except OSError:
        pass

    capture = cv2.VideoCapture(str(source))
    try:
        if not capture.isOpened():
            return None
        duration = float(sample.get("duration_s") or 0.0)
        fps = float(sample.get("fps") or 0.0)
        if duration > 1.0 and fps > 0:
            capture.set(cv2.CAP_PROP_POS_MSEC, min(duration * 0.1, 3.0) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = capture.read()
        if not ok or frame is None:
            return None

        height, width = frame.shape[:2]
        if width > THUMBNAIL_WIDTH:
            scale = THUMBNAIL_WIDTH / float(width)
            frame = cv2.resize(
                frame,
                (THUMBNAIL_WIDTH, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(".tmp.jpg")
        if not cv2.imwrite(str(temp), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82]):
            return None
        temp.replace(target)
        return target
    except Exception:  # noqa: BLE001 - thumbnails are optional
        return None
    finally:
        capture.release()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _signature(directory: Path) -> tuple[tuple[str, int, int], ...]:
    if not directory.is_dir():
        return ()
    entries: list[tuple[str, int, int]] = []
    for path in directory.iterdir():
        if not path.is_file() or not is_supported_video(path.name):
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(entries))


def _resolve_inside(directory: Path, name: str) -> Path | None:
    """Resolve a bare filename inside ``directory`` or return None."""
    if Path(name).name != name:
        return None
    try:
        base = directory.resolve()
        candidate = (directory / name).resolve()
    except OSError:
        return None
    try:
        if not candidate.is_relative_to(base):
            return None
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def discover_samples(force: bool = False) -> list[dict[str, Any]]:
    """Return metadata for every readable sample video, cached by directory state."""
    directory = sample_dir()
    signature = _signature(directory)
    with _cache_lock:
        if not force and _cache["signature"] == signature:
            return [dict(item) for item in _cache["samples"]]

    report = _load_report()
    samples: list[dict[str, Any]] = []
    for name, _mtime, size in signature:
        path = _resolve_inside(directory, name)
        if path is None:
            continue
        info = probe_video(path)
        if not info["ok"]:
            continue
        category = _category_for(name, report)
        video_id = video_id_for(name)
        sample = {
            "id": video_id,
            "filename": name,
            "path": str(path),
            "width": info["width"],
            "height": info["height"],
            "fps": info["fps"],
            "frame_count": info["frame_count"],
            "duration_s": info["duration_s"],
            "size_bytes": int(size),
            "category": category["label"],
            "category_kind": category["kind"],
            "category_detail": category["detail"],
            "orientation": (
                "portrait" if info["height"] > info["width"] else "landscape"
            ),
            "thumbnail_url": f"/api/videos/samples/{video_id}/thumbnail",
            "stream_url": f"/api/videos/samples/{video_id}/stream",
            "metadata_url": f"/api/videos/samples/{video_id}/metadata",
        }
        samples.append(sample)

    samples.sort(key=lambda item: item["filename"].lower())
    with _cache_lock:
        _cache["signature"] = signature
        _cache["samples"] = samples
    return [dict(item) for item in samples]


def get_sample(video_id: str) -> dict[str, Any] | None:
    if not video_id or len(video_id) > 64:
        return None
    for sample in discover_samples():
        if sample["id"] == video_id:
            return sample
    return None


def resolve_sample_path(video_id: str) -> Path:
    sample = get_sample(video_id)
    if sample is None:
        raise SampleVideoError("That sample video was not found.")
    path = _resolve_inside(sample_dir(), sample["filename"])
    if path is None:
        raise SampleVideoError("That sample video is no longer available.")
    return path


def clear_cache() -> None:
    with _cache_lock:
        _cache["signature"] = None
        _cache["samples"] = []
