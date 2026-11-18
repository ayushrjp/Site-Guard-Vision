"""Lightweight SQLite persistence for analysis history and application settings.

SiteGuardVision previously kept no history: every result existed only for the
duration of one HTTP response.  This module adds a small, dependency-free store
so the Reports and Settings pages can operate on real data.

The database location can be overridden with ``SITEGUARD_DB_PATH``.  Schema
creation is idempotent, and every call opens a short-lived connection so the
module is safe to use from Flask's threaded development server.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import runtime_config

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_DB_PATH = runtime_config.DB_PATH

DEFAULT_SETTINGS: dict[str, str] = {
    "conf": "0.25",
    "imgsz": "960",
    "ollama_model": "qwen2.5:3b",
}

ALLOWED_IMGSZ = {640, 960, 1280}
MIN_CONF = 0.05
MAX_CONF = 0.95
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]{0,79}$")

_init_lock = threading.Lock()
_schema_ready: set[str] = set()


class _ClosingConnection(sqlite3.Connection):
    """SQLite connection that also closes when used with a ``with`` block."""

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            return super().__exit__(exc_type, exc_value, traceback)
        finally:
            self.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    source_label TEXT,
    source_name TEXT,
    annotated_rel TEXT,
    assessment_status TEXT NOT NULL,
    assessment_title TEXT,
    assessment_message TEXT,
    person_count INTEGER NOT NULL DEFAULT 0,
    hardhat_count INTEGER NOT NULL DEFAULT 0,
    no_hardhat_count INTEGER NOT NULL DEFAULT 0,
    vest_count INTEGER NOT NULL DEFAULT 0,
    no_vest_count INTEGER NOT NULL DEFAULT 0,
    confidence REAL,
    person_confidence REAL,
    imgsz INTEGER,
    inference_ms REAL,
    ppe_inference_ms REAL,
    person_inference_ms REAL,
    total_ms REAL,
    image_width INTEGER,
    image_height INTEGER,
    ppe_model TEXT,
    person_source TEXT,
    payload_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(assessment_status);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Video analysis is additive.  The image-analysis table above is unchanged.
CREATE TABLE IF NOT EXISTS video_jobs (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    source_type TEXT NOT NULL,
    source_id TEXT,
    original_filename TEXT NOT NULL,
    stored_path TEXT,
    status TEXT NOT NULL,
    progress REAL NOT NULL DEFAULT 0,
    message TEXT,
    error TEXT,
    duration_s REAL,
    width INTEGER,
    height INTEGER,
    fps REAL,
    total_frames INTEGER,
    sampled_frames INTEGER,
    processed_frames INTEGER,
    output_rel TEXT,
    worker_count INTEGER NOT NULL DEFAULT 0,
    compliant_workers INTEGER NOT NULL DEFAULT 0,
    violation_workers INTEGER NOT NULL DEFAULT 0,
    unconfirmed_workers INTEGER NOT NULL DEFAULT 0,
    event_count INTEGER NOT NULL DEFAULT 0,
    settings_json TEXT,
    summary_json TEXT,
    result_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_video_jobs_created ON video_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_video_jobs_status ON video_jobs(status);

CREATE TABLE IF NOT EXISTS video_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    worker_id INTEGER,
    category TEXT NOT NULL,
    label TEXT,
    status TEXT NOT NULL,
    confidence REAL,
    first_seen_s REAL,
    last_seen_s REAL,
    first_frame INTEGER,
    last_frame INTEGER,
    observations INTEGER,
    evidence TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(job_id) REFERENCES video_jobs(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_video_events_job ON video_events(job_id, first_seen_s);
"""


def db_path() -> Path:
    """Return the active database path (environment override wins)."""
    override = os.environ.get("SITEGUARD_DB_PATH")
    return Path(override).expanduser() if override else DEFAULT_DB_PATH


def _connect() -> sqlite3.Connection:
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=15, factory=_ClosingConnection)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=15000")
    key = str(path.resolve())
    with _init_lock:
        if key not in _schema_ready:
            connection.executescript(SCHEMA)
            connection.commit()
            _schema_ready.add(key)
    return connection


def init_db() -> None:
    """Create the database and schema if they do not exist yet."""
    with _connect():
        pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------------


def save_analysis(
    result: dict[str, Any],
    *,
    source_label: str,
    source_name: str,
    imgsz: int,
) -> int | None:
    """Persist one successful analysis and return its row id.

    Failure to write history must never break image analysis, so callers treat
    any exception as a warning.  The full engine payload is stored as JSON so a
    historical result page can be reconstructed exactly.
    """
    if not result.get("success"):
        return None

    image_size = result.get("image_size") or [0, 0]
    payload = json.dumps(result, ensure_ascii=False, default=str)
    model_info = result.get("model_info") or {}

    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                created_at, source_label, source_name, annotated_rel,
                assessment_status, assessment_title, assessment_message,
                person_count, hardhat_count, no_hardhat_count,
                vest_count, no_vest_count, confidence, person_confidence,
                imgsz, inference_ms, ppe_inference_ms, person_inference_ms,
                total_ms, image_width, image_height, ppe_model, person_source,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_now(),
                source_label,
                source_name,
                result.get("annotated_rel"),
                result.get("assessment_status", "insufficient_evidence"),
                result.get("assessment_title"),
                result.get("assessment_message"),
                int(result.get("person_count") or 0),
                int(result.get("hardhat_count") or 0),
                int(result.get("no_hardhat_count") or 0),
                int(result.get("vest_count") or 0),
                int(result.get("no_vest_count") or 0),
                result.get("confidence_used"),
                result.get("person_confidence_used"),
                int(imgsz),
                result.get("inference_ms"),
                result.get("ppe_inference_ms"),
                result.get("person_inference_ms"),
                result.get("total_ms"),
                int(image_size[0]) if len(image_size) > 0 else 0,
                int(image_size[1]) if len(image_size) > 1 else 0,
                model_info.get("ppe_model"),
                model_info.get("person_source"),
                payload,
            ),
        )
        connection.commit()
        return int(cursor.lastrowid)


def _row_to_summary(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "created_at": row["created_at"],
        "source_label": row["source_label"],
        "annotated_rel": row["annotated_rel"],
        "assessment_status": row["assessment_status"],
        "assessment_title": row["assessment_title"],
        "person_count": row["person_count"],
        "hardhat_count": row["hardhat_count"],
        "no_hardhat_count": row["no_hardhat_count"],
        "vest_count": row["vest_count"],
        "no_vest_count": row["no_vest_count"],
        "inference_ms": row["inference_ms"],
        "image_width": row["image_width"],
        "image_height": row["image_height"],
    }


def list_analyses(limit: int = 50, offset: int = 0) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT * FROM analyses
            ORDER BY datetime(created_at) DESC, id DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        ).fetchall()
    return [_row_to_summary(row) for row in rows]


def get_analysis(analysis_id: int) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM analyses WHERE id = ?", (int(analysis_id),)
        ).fetchone()
    if row is None:
        return None
    try:
        payload = json.loads(row["payload_json"])
    except (TypeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    payload.setdefault("success", True)
    payload.setdefault("assessment_status", row["assessment_status"])
    payload.setdefault("assessment_title", row["assessment_title"])
    payload.setdefault("assessment_message", row["assessment_message"])
    payload.setdefault("annotated_rel", row["annotated_rel"])
    summary = _row_to_summary(row)
    summary["result"] = payload
    return summary


def stats() -> dict[str, Any]:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT
                COUNT(*) AS total,
                COALESCE(SUM(person_count), 0) AS workers,
                COALESCE(SUM(no_hardhat_count), 0) AS missing_hardhats,
                COALESCE(SUM(no_vest_count), 0) AS missing_vests,
                COALESCE(AVG(inference_ms), 0) AS avg_inference_ms,
                SUM(CASE WHEN assessment_status = 'potential_violation' THEN 1 ELSE 0 END) AS violations,
                SUM(CASE WHEN assessment_status = 'insufficient_evidence' THEN 1 ELSE 0 END) AS insufficient,
                SUM(CASE WHEN assessment_status = 'no_violation_detected' THEN 1 ELSE 0 END) AS clear
            FROM analyses
            """
        ).fetchone()
    return {
        "total": int(row["total"] or 0),
        "workers": int(row["workers"] or 0),
        "missing_hardhats": int(row["missing_hardhats"] or 0),
        "missing_vests": int(row["missing_vests"] or 0),
        "violations": int(row["violations"] or 0),
        "insufficient": int(row["insufficient"] or 0),
        "clear": int(row["clear"] or 0),
        "avg_inference_ms": round(float(row["avg_inference_ms"] or 0), 1),
    }


def count_analyses() -> int:
    with _connect() as connection:
        row = connection.execute("SELECT COUNT(*) AS n FROM analyses").fetchone()
    return int(row["n"] or 0)


def export_rows() -> Iterable[dict[str, Any]]:
    """Yield every stored analysis for CSV export (newest first)."""
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM analyses ORDER BY datetime(created_at) DESC, id DESC"
        ).fetchall()
    for row in rows:
        summary = _row_to_summary(row)
        summary.update(
            {
                "confidence": row["confidence"],
                "person_confidence": row["person_confidence"],
                "imgsz": row["imgsz"],
                "assessment_message": row["assessment_message"],
                "ppe_model": row["ppe_model"],
                "person_source": row["person_source"],
            }
        )
        yield summary


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


def _coerce_setting(key: str, raw: Any) -> str:
    value = "" if raw is None else str(raw).strip()
    if key == "conf":
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("PPE confidence must be a number between 0.05 and 0.95.") from exc
        return f"{min(MAX_CONF, max(MIN_CONF, round(number, 2))):.2f}"
    if key == "imgsz":
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("Image detail must be 640, 960, or 1280 pixels.") from exc
        if number not in ALLOWED_IMGSZ:
            raise ValueError("Image detail must be 640, 960, or 1280 pixels.")
        return str(number)
    if key == "ollama_model":
        if not _MODEL_RE.match(value):
            raise ValueError("Ollama model name contains unsupported characters.")
        return value
    raise ValueError(f"Unknown setting: {key}")


def get_settings() -> dict[str, str]:
    settings = dict(DEFAULT_SETTINGS)
    try:
        with _connect() as connection:
            rows = connection.execute("SELECT key, value FROM settings").fetchall()
    except sqlite3.Error:
        return settings
    for row in rows:
        if row["key"] in settings:
            settings[row["key"]] = row["value"]
    return settings


def save_settings(values: dict[str, Any]) -> tuple[dict[str, str], dict[str, str]]:
    """Validate and store settings; returns (settings, errors).

    Unknown keys are ignored.  Nothing is written when any supplied value is
    invalid, so a partially applied settings form cannot occur.
    """
    errors: dict[str, str] = {}
    cleaned: dict[str, str] = {}
    for key, raw in values.items():
        if key not in DEFAULT_SETTINGS:
            continue
        try:
            cleaned[key] = _coerce_setting(key, raw)
        except ValueError as exc:
            errors[key] = str(exc)

    if errors:
        return get_settings(), errors

    now = _utc_now()
    with _connect() as connection:
        for key, value in cleaned.items():
            connection.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value,
                                              updated_at = excluded.updated_at
                """,
                (key, value, now),
            )
        connection.commit()
    return get_settings(), {}


# ---------------------------------------------------------------------------
# Video analysis jobs (additive; never touches the image-analysis tables)
# ---------------------------------------------------------------------------

_VIDEO_JOB_FIELDS = {
    "updated_at",
    "started_at",
    "completed_at",
    "status",
    "progress",
    "message",
    "error",
    "duration_s",
    "width",
    "height",
    "fps",
    "total_frames",
    "sampled_frames",
    "processed_frames",
    "output_rel",
    "worker_count",
    "compliant_workers",
    "violation_workers",
    "unconfirmed_workers",
    "event_count",
    "settings_json",
    "summary_json",
    "result_json",
}

_VIDEO_JOB_COLUMNS = (
    "id",
    "created_at",
    "updated_at",
    "started_at",
    "completed_at",
    "source_type",
    "source_id",
    "original_filename",
    "stored_path",
    "status",
    "progress",
    "message",
    "error",
    "duration_s",
    "width",
    "height",
    "fps",
    "total_frames",
    "sampled_frames",
    "processed_frames",
    "output_rel",
    "worker_count",
    "compliant_workers",
    "violation_workers",
    "unconfirmed_workers",
    "event_count",
    "settings_json",
    "summary_json",
    "result_json",
)


def _json_object(raw: Any) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _video_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    item = {column: row[column] for column in _VIDEO_JOB_COLUMNS}
    item["settings"] = _json_object(item.pop("settings_json", None))
    item["summary"] = _json_object(item.pop("summary_json", None))
    item["result"] = _json_object(item.pop("result_json", None))
    return item


def create_video_job(
    job_id: str,
    *,
    source_type: str,
    original_filename: str,
    source_id: str | None = None,
    stored_path: str | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    now = _utc_now()
    with _connect() as connection:
        connection.execute(
            """
            INSERT INTO video_jobs (
                id, created_at, updated_at, source_type, source_id,
                original_filename, stored_path, status, progress,
                settings_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?)
            """,
            (
                job_id,
                now,
                now,
                source_type,
                source_id,
                original_filename,
                stored_path,
                json.dumps(settings or {}, ensure_ascii=False, default=str),
            ),
        )
        connection.commit()
    job = get_video_job(job_id)
    if job is None:  # pragma: no cover - the insert above just succeeded
        raise RuntimeError("Could not create the video job.")
    return job


def update_video_job(job_id: str, **fields: Any) -> None:
    updates = {key: value for key, value in fields.items() if key in _VIDEO_JOB_FIELDS}
    if not updates:
        return
    updates.setdefault("updated_at", _utc_now())
    assignments = ", ".join(f"{key} = ?" for key in updates)
    values = list(updates.values()) + [job_id]
    with _connect() as connection:
        connection.execute(
            f"UPDATE video_jobs SET {assignments} WHERE id = ?",  # noqa: S608 - fixed whitelist
            values,
        )
        connection.commit()


def get_video_job(job_id: str) -> dict[str, Any] | None:
    with _connect() as connection:
        row = connection.execute(
            "SELECT * FROM video_jobs WHERE id = ?", (str(job_id),)
        ).fetchone()
    return _video_row_to_dict(row) if row is not None else None


def list_video_jobs(limit: int = 20) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 200))
    with _connect() as connection:
        rows = connection.execute(
            "SELECT * FROM video_jobs ORDER BY datetime(created_at) DESC, id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_video_row_to_dict(row) for row in rows]


def finish_video_job(
    job_id: str,
    *,
    result: dict[str, Any],
    events: list[dict[str, Any]],
    output_rel: str | None,
    summary: dict[str, Any],
    duration_s: float | None,
    width: int | None,
    height: int | None,
    fps: float | None,
    total_frames: int | None,
    sampled_frames: int | None,
    processed_frames: int | None,
) -> None:
    """Persist a completed job and replace its event rows atomically."""
    now = _utc_now()
    with _connect() as connection:
        connection.execute(
            """
            UPDATE video_jobs SET
                updated_at = ?, completed_at = ?, status = 'completed',
                progress = 100, message = ?, error = NULL,
                duration_s = ?, width = ?, height = ?, fps = ?,
                total_frames = ?, sampled_frames = ?, processed_frames = ?,
                output_rel = ?, worker_count = ?, compliant_workers = ?,
                violation_workers = ?, unconfirmed_workers = ?, event_count = ?,
                summary_json = ?, result_json = ?
            WHERE id = ?
            """,
            (
                now,
                now,
                "Video analysis complete.",
                duration_s,
                width,
                height,
                fps,
                total_frames,
                sampled_frames,
                processed_frames,
                output_rel,
                int(summary.get("worker_count") or 0),
                int(summary.get("compliant_workers") or 0),
                int(summary.get("violation_workers") or 0),
                int(summary.get("unconfirmed_workers") or 0),
                int(summary.get("event_count") or 0),
                json.dumps(summary, ensure_ascii=False, default=str),
                json.dumps(result, ensure_ascii=False, default=str),
                job_id,
            ),
        )
        connection.execute("DELETE FROM video_events WHERE job_id = ?", (job_id,))
        for event in events:
            connection.execute(
                """
                INSERT INTO video_events (
                    job_id, worker_id, category, label, status, confidence,
                    first_seen_s, last_seen_s, first_frame, last_frame,
                    observations, evidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    event.get("worker_id"),
                    event.get("category"),
                    event.get("label"),
                    event.get("status", "potential_violation"),
                    event.get("confidence"),
                    event.get("first_seen_s"),
                    event.get("last_seen_s"),
                    event.get("first_frame"),
                    event.get("last_frame"),
                    event.get("observations"),
                    event.get("evidence"),
                    now,
                ),
            )
        connection.commit()


def insert_video_events(job_id: str, events: list[dict[str, Any]]) -> None:
    """Replace event rows for a running job (used for incremental progress)."""
    now = _utc_now()
    with _connect() as connection:
        connection.execute("DELETE FROM video_events WHERE job_id = ?", (job_id,))
        for event in events:
            connection.execute(
                """
                INSERT INTO video_events (
                    job_id, worker_id, category, label, status, confidence,
                    first_seen_s, last_seen_s, first_frame, last_frame,
                    observations, evidence, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job_id,
                    event.get("worker_id"),
                    event.get("category"),
                    event.get("label"),
                    event.get("status", "potential_violation"),
                    event.get("confidence"),
                    event.get("first_seen_s"),
                    event.get("last_seen_s"),
                    event.get("first_frame"),
                    event.get("last_frame"),
                    event.get("observations"),
                    event.get("evidence"),
                    now,
                ),
            )
        connection.commit()


def list_video_events(job_id: str) -> list[dict[str, Any]]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT worker_id, category, label, status, confidence,
                   first_seen_s, last_seen_s, first_frame, last_frame,
                   observations, evidence
            FROM video_events WHERE job_id = ?
            ORDER BY first_seen_s ASC, worker_id ASC, category ASC
            """,
            (job_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def fail_stale_video_jobs(message: str = "Server restarted before this job finished.") -> int:
    """Mark abandoned queued/running jobs as failed (called once at startup)."""
    now = _utc_now()
    with _connect() as connection:
        cursor = connection.execute(
            """
            UPDATE video_jobs
               SET status = 'failed', error = ?, message = ?, updated_at = ?
             WHERE status IN ('queued', 'running')
            """,
            (message, message, now),
        )
        connection.commit()
        return int(cursor.rowcount or 0)
