"""Environment-driven filesystem layout for SiteGuardVision.

Local development keeps every path at its historical location inside the
project directory, so existing databases, annotated images and video outputs
continue to work untouched.

On a container host such as Render the project directory is read-only/
ephemeral.  Setting a single variable, ``SITEGUARD_DATA_DIR``, relocates every
writable path (SQLite database, image uploads, annotated images, video
uploads/outputs/thumbnails) onto the mounted persistent disk.  Individual
paths can still be overridden with their historical variables.

Bundled, read-only assets (models, sample photos, sample videos, templates,
static CSS/JS) always stay in the repository.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent


def env_path(name: str, default: Path) -> Path:
    """Return ``$name`` as an expanded Path, or ``default`` when unset/blank."""
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return Path(raw.strip()).expanduser()


# ---------------------------------------------------------------------------
# Bundled read-only assets (shipped in the Git checkout / container image)
# ---------------------------------------------------------------------------
MODELS_DIR = ROOT_DIR / "models"
SAMPLES_DIR = ROOT_DIR / "samples"
TEMPLATES_DIR = ROOT_DIR / "templates"
STATIC_DIR = ROOT_DIR / "static"
DEFAULT_SAMPLE_VIDEO_DIR = ROOT_DIR / "test_videos"

# ---------------------------------------------------------------------------
# Writable runtime state
# ---------------------------------------------------------------------------
DATA_DIR = env_path("SITEGUARD_DATA_DIR", ROOT_DIR / "data")

DB_PATH = env_path("SITEGUARD_DB_PATH", DATA_DIR / "siteguard.db")

# Image workspace. Defaults deliberately match the historical locations so
# existing stored analyses ("rendered/<name>") keep resolving locally.
INCOMING_DIR = env_path("SITEGUARD_IMAGE_UPLOAD_DIR", ROOT_DIR / "static" / "incoming")
RENDERED_DIR = env_path("SITEGUARD_RENDERED_DIR", ROOT_DIR / "static" / "rendered")

# Video workspace (video_processing.jobs / sample_library honour the same
# variables; these constants are their shared defaults).
VIDEO_UPLOAD_DIR = env_path(
    "SITEGUARD_VIDEO_UPLOAD_DIR", DATA_DIR / "video_uploads"
)
VIDEO_OUTPUT_DIR = env_path(
    "SITEGUARD_VIDEO_OUTPUT_DIR", DATA_DIR / "video_outputs"
)
VIDEO_THUMB_DIR = env_path(
    "SITEGUARD_VIDEO_THUMB_DIR", DATA_DIR / "video_thumbnails"
)

# Assistant / RAG state.
VECTOR_DB_DIR = env_path("SITEGUARD_VECTOR_DB", ROOT_DIR / "vector_db")
KNOWLEDGE_BASE_DIR = env_path(
    "SITEGUARD_KNOWLEDGE_BASE", ROOT_DIR / "knowledge_base" / "documents"
)

# Read-only sample videos can also be relocated, but the default is the
# bundled test_videos/ directory.
SAMPLE_VIDEO_DIR = env_path("SITEGUARD_SAMPLE_VIDEO_DIR", DEFAULT_SAMPLE_VIDEO_DIR)


def writable_dirs() -> tuple[Path, ...]:
    """Every directory the running application may create/write at runtime."""
    return (
        INCOMING_DIR,
        RENDERED_DIR,
        DB_PATH.parent,
        VIDEO_UPLOAD_DIR,
        VIDEO_OUTPUT_DIR,
        VIDEO_THUMB_DIR,
        VECTOR_DB_DIR,
        KNOWLEDGE_BASE_DIR,
    )


def ensure_writable_dirs() -> None:
    """Create runtime directories; safe to call at import time."""
    for directory in writable_dirs():
        directory.mkdir(parents=True, exist_ok=True)


def describe() -> dict[str, str]:
    """Compact, secret-free description used by /health and diagnostics."""
    return {
        "root": str(ROOT_DIR),
        "data_dir": str(DATA_DIR),
        "db_path": str(DB_PATH),
        "incoming_dir": str(INCOMING_DIR),
        "rendered_dir": str(RENDERED_DIR),
        "video_upload_dir": str(VIDEO_UPLOAD_DIR),
        "video_output_dir": str(VIDEO_OUTPUT_DIR),
        "video_thumb_dir": str(VIDEO_THUMB_DIR),
        "vector_db_dir": str(VECTOR_DB_DIR),
        "knowledge_base_dir": str(KNOWLEDGE_BASE_DIR),
        "sample_video_dir": str(SAMPLE_VIDEO_DIR),
    }
