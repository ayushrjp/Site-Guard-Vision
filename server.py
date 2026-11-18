"""
SiteGuardVision - Flask application.

Routes
------
GET  /                          Image-analysis dashboard (upload + samples)
POST /analyze                   Run PPE + person inference and render the result
GET  /analysis/<id>             Re-open a stored analysis
GET  /videos                    Video Safety Analysis workspace
GET  /videos/jobs/<job_id>      Video analysis result dashboard
GET  /assistant                 Safety Assistant workspace
GET  /reports                   Stored analysis history
GET  /reports/export.csv        Export stored analyses as CSV
GET  /evaluation                Dataset & Evaluation reports
GET  /settings                  Application settings
GET  /samples/<filename>        Serve a bundled sample photo
GET  /health                    Health check
GET  /api/status                Combined application/Ollama/RAG health
POST /api/assistant/chat        Grounded assistant answer
GET  /api/assistant/status      Ollama + RAG status
GET  /api/assistant/documents   Knowledge-base document list
POST /api/assistant/documents   Upload and index a PDF/TXT/MD document
DEL  /api/assistant/documents/<name>
POST /api/assistant/reindex     Rebuild the vector index
GET  /api/settings              Read settings
POST /api/settings              Update settings
GET  /api/videos/samples        Discover sample videos
GET  /api/videos/samples/<id>/metadata
GET  /api/videos/samples/<id>/stream
GET  /api/videos/samples/<id>/thumbnail
POST /api/videos/upload         Upload and validate a video
GET  /api/videos/uploads/<id>/stream
POST /api/videos/analyze        Queue a sample or upload for analysis
GET  /api/videos/jobs/<id>      Job status and progress
GET  /api/videos/jobs/<id>/events
GET  /api/videos/jobs/<id>/result
GET  /api/videos/jobs/<id>/output
GET  /api/videos/jobs/<id>/source

Image analysis remains independent of video processing and Ollama.
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import threading
import uuid
from pathlib import Path

from flask import (
    Flask,
    Response,
    abort,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    send_from_directory,
    url_for,
)
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import HTTPException
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

import runtime_config
import storage
from evaluation_loader import load_evaluation
from infer_engine import (
    ALLOWED_IMGSZ,
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    MAX_CONF,
    MIN_CONF,
    analyze_image,
)
from rag import config as rag_config
from rag.embeddings import OllamaError
from rag.ingest import ingest_files, reindex_all, remove_document
from rag.safety_assistant import get_assistant
from rag.vector_store import (
    RagError,
    list_documents,
    pending_documents,
    read_manifest,
)
from video_processing import jobs as video_jobs
from video_processing import sample_library
from video_processing.video_engine import VideoSettings

ROOT_DIR = Path(__file__).resolve().parent
INCOMING_DIR = runtime_config.INCOMING_DIR
RENDERED_DIR = runtime_config.RENDERED_DIR
SAMPLES_DIR = runtime_config.SAMPLES_DIR
MODELS_DIR = runtime_config.MODELS_DIR
runtime_config.ensure_writable_dirs()
rag_config.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_DOCUMENT_BYTES = min(rag_config.MAX_DOCUMENT_BYTES, MAX_UPLOAD_BYTES)
MAX_VIDEO_BYTES = video_jobs.MAX_VIDEO_BYTES

PRETTY_LABELS = {
    "fall-detected": "Fall detected",
    "gloves": "Gloves",
    "goggles": "Goggles",
    "hardhat": "Hard hat",
    "ladder": "Ladder",
    "mask": "Mask",
    "no-gloves": "No gloves",
    "no-goggles": "No goggles",
    "no-hardhat": "No hard hat",
    "no-mask": "No mask",
    "no-safety-vest": "No safety vest",
    "person": "Person",
    "safety-cone": "Safety cone",
    "safety-vest": "Safety vest",
}

app = Flask(__name__)
# The WSGI limit must accommodate videos; image and document routes enforce
# their own 15 MB limits before reading the body.
app.config["MAX_CONTENT_LENGTH"] = MAX_VIDEO_BYTES
# Never run with the interactive debugger in a deployed environment. Local
# development can still opt in with FLASK_DEBUG=1 (see __main__ below).
app.config["DEBUG"] = os.environ.get("FLASK_DEBUG") == "1"
app.config["TEMPLATES_AUTO_RELOAD"] = app.config["DEBUG"]

logging.basicConfig(
    level=os.environ.get("SITEGUARD_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
app.logger.setLevel(os.environ.get("SITEGUARD_LOG_LEVEL", "INFO").upper())

# Render terminates TLS at its proxy; trust exactly one forwarded hop so
# request.url / redirects use https while remaining unspoofable off-proxy.
if os.environ.get("SITEGUARD_BEHIND_PROXY", "").strip() == "1":
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)

_sample_cache_lock = threading.Lock()
_sample_cache: tuple[tuple[tuple[str, int, int], ...], list[str]] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _content_length() -> int:
    try:
        return int(request.content_length or 0)
    except (TypeError, ValueError):
        return 0


def _public_sample(sample: dict) -> dict:
    """Strip server-local paths before a sample reaches a client."""
    return {key: value for key, value in sample.items() if key != "path"}


def _public_job(job: dict, *, include_result: bool = False) -> dict:
    hidden = {"stored_path"}
    if not include_result:
        hidden.add("result")
    return {key: value for key, value in job.items() if key not in hidden}


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def _sample_signature() -> tuple[tuple[str, int, int], ...]:
    """Cheap directory fingerprint so the readable-sample scan is cached."""
    if not SAMPLES_DIR.exists():
        return ()
    entries: list[tuple[str, int, int]] = []
    for path in SAMPLES_DIR.iterdir():
        if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        entries.append((path.name, stat.st_mtime_ns, stat.st_size))
    return tuple(sorted(entries))


def _list_samples() -> list[str]:
    """Readable sample filenames only; broken files must not reach the UI."""
    global _sample_cache

    signature = _sample_signature()
    with _sample_cache_lock:
        cached = _sample_cache
    if cached is not None and cached[0] == signature:
        return list(cached[1])

    readable = [
        name for name, _mtime, _size in signature
        if _is_readable_image(SAMPLES_DIR / name)
    ]
    with _sample_cache_lock:
        _sample_cache = (signature, readable)
    return list(readable)


def _clamp_conf(raw: str | None, default: float | None = None) -> float:
    fallback = DEFAULT_CONF if default is None else default
    try:
        value = float(raw) if raw not in (None, "") else fallback
    except (TypeError, ValueError):
        value = fallback
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = DEFAULT_CONF
    return min(MAX_CONF, max(MIN_CONF, round(value, 2)))


def _parse_imgsz(raw: str | None, default: int | None = None) -> int:
    fallback = DEFAULT_IMGSZ if default is None else default
    try:
        value = int(raw) if raw not in (None, "") else fallback
    except (TypeError, ValueError):
        value = fallback
    return value if value in ALLOWED_IMGSZ else DEFAULT_IMGSZ


def _is_readable_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        return False


def _safe_recent(limit: int = 4) -> list[dict]:
    try:
        return storage.list_analyses(limit=limit)
    except Exception:  # noqa: BLE001 - history is optional
        app.logger.warning("Could not read analysis history", exc_info=True)
        return []


def _safe_stats() -> dict:
    try:
        return storage.stats()
    except Exception:  # noqa: BLE001
        app.logger.warning("Could not read analysis statistics", exc_info=True)
        return {
            "total": 0, "workers": 0, "missing_hardhats": 0, "missing_vests": 0,
            "violations": 0, "insufficient": 0, "clear": 0, "avg_inference_ms": 0.0,
        }


def _home(error: str | None = None, status: int = 200):
    settings = storage.get_settings()
    recent = _safe_recent()
    return (
        render_template(
            "home.html",
            samples=_list_samples(),
            error=error,
            settings=settings,
            recent=recent,
            latest_analysis_id=recent[0]["id"] if recent else None,
        ),
        status,
    )


def _document_listing() -> dict:
    try:
        documents = list_documents()
        pending = pending_documents()
        manifest = read_manifest()
        return {
            "documents": documents,
            "pending": pending,
            "embedding_model": manifest.get("embedding_model"),
            "indexed_at": manifest.get("updated_at"),
        }
    except Exception:  # noqa: BLE001
        app.logger.warning("Could not list knowledge-base documents", exc_info=True)
        return {"documents": [], "pending": [], "embedding_model": None, "indexed_at": None}


def _settings_page(
    *,
    message: str | None = None,
    error: str | None = None,
    status: int = 200,
):
    return (
        render_template(
            "settings.html",
            settings=storage.get_settings(),
            documents=_document_listing(),
            status=get_assistant().status(),
            message=message,
            error=error,
        ),
        status,
    )


# ---------------------------------------------------------------------------
# Template helpers
# ---------------------------------------------------------------------------


@app.template_filter("pretty_label")
def pretty_label(label: str) -> str:
    key = " ".join(label.strip().lower().replace("_", "-").split())
    return PRETTY_LABELS.get(key, key.replace("-", " ").capitalize())


@app.template_filter("worker_state")
def worker_state(state: str) -> str:
    return {
        "present": "Present",
        "missing": "Missing",
        "unknown": "Unconfirmed",
    }.get(state, state)


@app.template_filter("video_state")
def video_state(state: str) -> str:
    """Video slots add conflict/unconfirmed states to the image vocabulary."""
    return {
        "present": "Present",
        "missing": "Missing",
        "conflict": "Conflict",
        "unknown": "Unconfirmed",
        "unconfirmed": "Unconfirmed",
    }.get(state, state)


def video_time(seconds) -> str:
    """Render a video timestamp such as 01:23.4 for templates and events."""
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return "—"
    if value < 0 or value == float("inf"):
        return "—"
    minutes = int(value // 60)
    remainder = value - minutes * 60
    return f"{minutes:02d}:{remainder:04.1f}"


app.jinja_env.globals["video_time"] = video_time


@app.template_global("annotated_url")
def annotated_url(rel: str | None) -> str:
    """Resolve a stored ``annotated_rel`` value to a served URL.

    Stored rows keep the historical ``rendered/<file>`` form; the file is
    served from the configured rendered directory, which may live on a
    persistent disk outside ``static/`` in production.
    """
    if not rel:
        return ""
    return url_for("rendered_file", filename=Path(str(rel)).name)


@app.template_filter("model_label")
def model_label(value: str | None) -> str:
    """Render an Ollama model id as a compact human label (qwen2.5:3b -> Qwen2.5 3B)."""
    if not value:
        return "Not configured"
    base, _, tag = str(value).partition(":")
    base = base.replace("_", " ").replace("-", " ").strip()
    base = base[:1].upper() + base[1:] if base else "Model"
    return f"{base} {tag.upper()}" if tag else base


@app.template_filter("datetime_short")
def datetime_short(value: str | None) -> str:
    if not value:
        return "unknown"
    try:
        from datetime import datetime

        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed.strftime("%d %b %Y, %H:%M")
    except (TypeError, ValueError):
        return str(value)


# ---------------------------------------------------------------------------
# Response headers and errors
# ---------------------------------------------------------------------------


@app.after_request
def _security_headers(response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "same-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if request.endpoint == "analyze" or request.path.startswith("/api/assistant"):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def _too_large(_error):
    if request.path.startswith("/api/videos"):
        return jsonify({"ok": False, "error": "That video is larger than 256 MB."}), 413
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "That file is larger than 15 MB."}), 413
    return _home(
        "That photo is larger than 15 MB. Choose a smaller file and try again.",
        413,
    )


@app.errorhandler(404)
def _not_found(_error):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "Not found."}), 404
    return _home(
        "That image could not be found. Choose a sample or upload a photo.",
        404,
    )


@app.errorhandler(500)
def _server_error(error):
    """Production-safe failure page; tracebacks go to the logs, not the client."""
    app.logger.error("Unhandled server error: %s", error)
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": "Internal server error."}), 500
    try:
        return _home("Something went wrong on the server. Please try again.", 500)
    except Exception:  # noqa: BLE001 - never recurse into the error page
        return "Internal server error.", 500


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    models_present = all(
        path.is_file()
        for path in (
            MODELS_DIR / "yolo_ppe_core.pt",
            MODELS_DIR / "person_yolov8n.pt",
        )
    )
    return {
        "status": "ok",
        "application": "SiteGuardVision",
        "mode": "image + video PPE detection",
        "storage": "sqlite",
        "database": "present" if Path(storage.db_path()).is_file() else "not-created",
        "models": "present" if models_present else "missing",
        "sample_videos": (
            "present" if runtime_config.SAMPLE_VIDEO_DIR.is_dir() else "missing"
        ),
        "video_analysis": "on-demand",
    }


@app.get("/rendered/<path:filename>")
def rendered_file(filename: str):
    """Serve annotated images from the configured (possibly persistent) dir."""
    return send_from_directory(RENDERED_DIR, filename, max_age=3600)


@app.get("/samples/<path:filename>")
def sample_file(filename: str):
    # Only bare file names from samples/ are served.
    if Path(filename).name != filename or not _is_allowed(filename):
        abort(404)
    candidate = SAMPLES_DIR / filename
    if not candidate.is_file():
        abort(404)
    return send_from_directory(SAMPLES_DIR, filename, max_age=86400)


@app.get("/")
def home():
    return _home()


@app.post("/analyze")
def analyze():
    if _content_length() > MAX_UPLOAD_BYTES:
        return _home(
            "That photo is larger than 15 MB. Choose a smaller file and try again.",
            413,
        )
    uploaded = request.files.get("image")
    sample_name = request.form.get("sample")
    image_path: Path | None = None
    source_label = ""
    retain_upload = False
    settings = storage.get_settings()

    try:
        if uploaded and uploaded.filename:
            if not _is_allowed(uploaded.filename):
                return _home(
                    "Unsupported file type. Choose a JPG, PNG, WEBP, or BMP image."
                )

            original_name = Path(uploaded.filename).name
            safe_name = secure_filename(original_name)
            if not safe_name:
                return _home("That filename could not be used. Rename the photo and try again.")

            stored_name = f"{uuid.uuid4().hex[:10]}_{safe_name}"
            image_path = INCOMING_DIR / stored_name
            retain_upload = True
            uploaded.save(image_path)
            source_label = original_name

            if not _is_readable_image(image_path):
                image_path.unlink(missing_ok=True)
                retain_upload = False
                return _home(
                    "That file isn't a readable image. Choose a JPG, PNG, WEBP, or BMP photo."
                )

        elif sample_name:
            if sample_name not in _list_samples():
                abort(404)
            image_path = SAMPLES_DIR / sample_name
            source_label = sample_name

        else:
            return _home("Choose a sample photo or upload an image first.")

        conf = _clamp_conf(request.form.get("conf"), default=float(settings["conf"]))
        imgsz = _parse_imgsz(request.form.get("imgsz"), default=int(settings["imgsz"]))

        result = analyze_image(image_path, conf=conf, imgsz=imgsz)

        if result.get("success"):
            app.logger.info(
                "Analyzed %s | people=%s hardhats=%s no_hardhats=%s vests=%s no_vests=%s status=%s",
                source_label,
                result.get("person_count"),
                result.get("hardhat_count"),
                result.get("no_hardhat_count"),
                result.get("vest_count"),
                result.get("no_vest_count"),
                result.get("assessment_status"),
            )
        else:
            app.logger.warning("Inference failed for %s: %s", source_label, result.get("error"))

        analysis_id = None
        if result.get("success") and not app.config.get("TESTING"):
            try:
                analysis_id = storage.save_analysis(
                    result,
                    source_label=source_label,
                    source_name=image_path.name,
                    imgsz=imgsz,
                )
            except Exception:  # noqa: BLE001 - history must not break analysis
                app.logger.warning("Could not save analysis history", exc_info=True)

        return render_template(
            "analysis.html",
            result=result,
            source_label=source_label,
            source_name=image_path.name,
            conf=conf,
            imgsz=imgsz,
            analysis_id=analysis_id,
            settings=settings,
        )

    except HTTPException:
        raise
    except Exception:  # noqa: BLE001 - user-facing fallback
        app.logger.exception("Analysis failed for %s", source_label or image_path)
        return _home(
            "The analysis couldn't finish. Try another photo or a smaller image detail."
        )
    finally:
        # Uploads are temporary: the annotated preview is the only artifact the
        # dashboard needs, so do not retain the original file on disk.
        if retain_upload and image_path is not None:
            image_path.unlink(missing_ok=True)


@app.get("/analysis/<int:analysis_id>")
def analysis_detail(analysis_id: int):
    stored = storage.get_analysis(analysis_id)
    if stored is None:
        abort(404)
    result = stored["result"]
    return render_template(
        "analysis.html",
        result=result,
        source_label=stored.get("source_label") or "stored analysis",
        source_name=stored.get("source_label") or "",
        conf=result.get("confidence_used") or 0.25,
        imgsz=stored.get("imgsz") or DEFAULT_IMGSZ,
        analysis_id=stored["id"],
        settings=storage.get_settings(),
        is_history=True,
        created_at=stored.get("created_at"),
    )


# ---------------------------------------------------------------------------
# Video pages
# ---------------------------------------------------------------------------


@app.get("/videos")
def videos_page():
    samples = [_public_sample(item) for item in sample_library.discover_samples()]
    return render_template(
        "videos.html",
        samples=samples,
        jobs=storage.list_video_jobs(limit=20),
        settings=storage.get_settings(),
        video_defaults=VideoSettings().as_dict(),
        max_video_mb=MAX_VIDEO_BYTES // (1024 * 1024),
    )


@app.get("/videos/jobs/<job_id>")
def video_result_page(job_id: str):
    if not video_jobs.JOB_ID_RE.match(str(job_id)):
        abort(404)
    job = storage.get_video_job(job_id)
    if job is None:
        abort(404)
    return render_template(
        "video_result.html",
        job=job,
        events=storage.list_video_events(job_id),
        summary=job.get("summary") or {},
        settings=storage.get_settings(),
    )


@app.get("/assistant")
def assistant_page():
    assistant = get_assistant()
    listing = _document_listing()
    recent = storage.list_analyses(limit=1)
    return render_template(
        "assistant.html",
        status=assistant.status(),
        documents=listing["documents"],
        pending=listing["pending"],
        latest_analysis_id=recent[0]["id"] if recent else None,
        settings=storage.get_settings(),
    )


@app.get("/reports")
def reports_page():
    return render_template(
        "reports.html",
        stats=_safe_stats(),
        analyses=_safe_recent(limit=100),
        settings=storage.get_settings(),
    )


@app.get("/reports/export.csv")
def reports_export():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "id", "created_at", "source", "assessment_status", "assessment_title",
            "workers", "hardhats", "missing_hardhats", "vests", "missing_vests",
            "inference_ms", "confidence", "imgsz", "image_width", "image_height",
            "ppe_model", "person_source",
        ]
    )
    for row in storage.export_rows():
        writer.writerow(
            [
                row.get("id"), row.get("created_at"), row.get("source_label"),
                row.get("assessment_status"), row.get("assessment_title"),
                row.get("person_count"), row.get("hardhat_count"),
                row.get("no_hardhat_count"), row.get("vest_count"),
                row.get("no_vest_count"), row.get("inference_ms"),
                row.get("confidence"), row.get("imgsz"),
                row.get("image_width"), row.get("image_height"),
                row.get("ppe_model"), row.get("person_source"),
            ]
        )
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=siteguard-analyses.csv"},
    )


@app.get("/evaluation")
def evaluation_page():
    return render_template(
        "evaluation.html",
        evaluation=load_evaluation(),
        settings=storage.get_settings(),
    )


@app.get("/settings")
def settings_page():
    return _settings_page()


@app.post("/settings")
def settings_save_form():
    _, errors = storage.save_settings(request.form)
    if errors:
        return _settings_page(
            error=" ".join(errors.values()),
            status=400,
        )
    return redirect(url_for("settings_page", saved=1))


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------


@app.get("/api/status")
def api_status():
    assistant_status = get_assistant().status()
    try:
        analysis_total = storage.count_analyses()
        db_label = "SQLite"
        try:
            db_name = storage.db_path().resolve().relative_to(ROOT_DIR).as_posix()
        except ValueError:
            db_name = storage.db_path().name
    except Exception:  # noqa: BLE001
        analysis_total = 0
        db_label = "unavailable"
        db_name = None

    return jsonify(
        {
            "application": {
                "status": "ok",
                "name": "SiteGuardVision",
                "mode": "image-based PPE detection",
            },
            "models": {
                "ppe": {
                    "file": "yolo_ppe_core.pt",
                    "present": (MODELS_DIR / "yolo_ppe_core.pt").is_file(),
                },
                "person": {
                    "file": "person_yolov8n.pt",
                    "present": (MODELS_DIR / "person_yolov8n.pt").is_file(),
                },
            },
            "storage": {
                "engine": db_label,
                "database": db_name,
                "analyses": analysis_total,
            },
            "assistant": assistant_status,
            "settings": storage.get_settings(),
        }
    )


@app.get("/api/assistant/status")
def api_assistant_status():
    assistant = get_assistant()
    assistant_status = assistant.status(force=True)
    listing = _document_listing()
    return jsonify(
        {
            "ok": True,
            "assistant": assistant_status,
            "documents": listing["documents"],
            "pending": listing["pending"],
            "indexed_at": listing["indexed_at"],
        }
    )


@app.post("/api/assistant/chat")
def api_assistant_chat():
    payload = request.get_json(silent=True) or {}
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"ok": False, "error": "Please enter a question."}), 400

    analysis_id = payload.get("analysis_id")
    analysis = None
    if analysis_id not in (None, "", 0):
        try:
            stored = storage.get_analysis(int(analysis_id))
        except (TypeError, ValueError):
            stored = None
        if stored is not None:
            analysis = stored["result"]
        else:
            return jsonify({"ok": False, "error": "That analysis result was not found."}), 404

    session_id = payload.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        session_id = None
    else:
        session_id = session_id.strip()[:80]
    history = payload.get("history")
    if not isinstance(history, list):
        history = None

    try:
        answer = get_assistant().answer(
            message,
            analysis,
            analysis_id=analysis_id,
            history=history,
            session_id=session_id,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001 - must stay JSON for the chat UI
        app.logger.exception("Assistant request failed")
        return (
            jsonify(
                {
                    "ok": False,
                    "mode": "error",
                    "answer": f"The assistant could not complete the request: {type(exc).__name__}.",
                    "sources": [],
                }
            ),
            500,
        )
    return jsonify(answer)


@app.post("/api/assistant/clear")
def api_assistant_clear():
    payload = request.get_json(silent=True) or request.form.to_dict()
    session_id = payload.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        get_assistant().clear_session(session_id.strip()[:80])
    return jsonify({"ok": True, "message": "Conversation cleared."})


@app.get("/api/assistant/documents")
def api_assistant_documents():
    listing = _document_listing()
    return jsonify({"ok": True, **listing})


@app.post("/api/assistant/documents")
def api_assistant_upload_document():
    if _content_length() > MAX_DOCUMENT_BYTES:
        return jsonify({"ok": False, "error": "That document is larger than 15 MB."}), 413
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "Choose a PDF, TXT or Markdown document."}), 400

    original = Path(uploaded.filename).name
    suffix = Path(original).suffix.lower()
    if suffix not in rag_config.ALLOWED_DOCUMENT_EXTENSIONS:
        return jsonify({"ok": False, "error": "Only PDF, TXT and Markdown documents are accepted."}), 400

    safe_name = secure_filename(original)
    if not safe_name or Path(safe_name).stem == "":
        return jsonify({"ok": False, "error": "That filename could not be used."}), 400

    target = rag_config.KNOWLEDGE_BASE_DIR / safe_name
    uploaded.save(target)
    try:
        size = target.stat().st_size
        if size == 0:
            target.unlink(missing_ok=True)
            return jsonify({"ok": False, "error": "The uploaded document is empty."}), 400
        if size > MAX_DOCUMENT_BYTES:
            target.unlink(missing_ok=True)
            return (
                jsonify({"ok": False, "error": "That document is larger than 15 MB."}),
                413,
            )

        assistant = get_assistant()
        outcome = ingest_files(
            [target],
            embeddings=assistant.embeddings,
            store=assistant.store,
        )
    except (OllamaError, RagError) as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "saved": True,
                    "indexed": False,
                    "filename": safe_name,
                    "error": str(exc),
                    "message": "The document was saved but could not be indexed yet. Re-index when Ollama is available.",
                }
            ),
            202,
        )
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Document upload failed")
        return (
            jsonify(
                {
                    "ok": False,
                    "saved": target.is_file(),
                    "indexed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            ),
            500,
        )

    failed = outcome.get("failed") or []
    if failed:
        return (
            jsonify(
                {
                    "ok": False,
                    "saved": True,
                    "indexed": False,
                    "filename": safe_name,
                    "error": failed[0].get("error"),
                    "message": "The document was saved but indexing failed. Fix the issue and re-index.",
                }
            ),
            202,
        )

    indexed = outcome.get("indexed") or []
    chunks = indexed[0].get("chunks") if indexed else 0
    return (
        jsonify(
            {
                "ok": True,
                "saved": True,
                "indexed": bool(indexed),
                "filename": safe_name,
                "chunks": chunks,
                "message": (
                    f"Indexed {chunks} chunks from {safe_name}."
                    if indexed
                    else f"{safe_name} is unchanged and already indexed."
                ),
            }
        ),
        201,
    )


@app.delete("/api/assistant/documents/<path:filename>")
def api_assistant_delete_document(filename: str):
    try:
        remove_document(filename, store=get_assistant().store)
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Could not delete knowledge-base document")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"ok": True, "message": f"Removed {filename}."})


@app.post("/api/assistant/reindex")
def api_assistant_reindex():
    assistant = get_assistant()
    try:
        outcome = reindex_all(embeddings=assistant.embeddings, store=assistant.store)
    except (OllamaError, RagError) as exc:
        return jsonify({"ok": False, "error": str(exc)}), 503
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Re-index failed")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500
    return jsonify({"ok": True, **outcome})


# ---------------------------------------------------------------------------
# Video API
# ---------------------------------------------------------------------------


def _video_job_or_404(job_id: str) -> dict:
    if not video_jobs.JOB_ID_RE.match(str(job_id)):
        abort(404)
    job = storage.get_video_job(job_id)
    if job is None:
        abort(404)
    return job


@app.get("/api/videos/samples")
def api_video_samples():
    samples = [_public_sample(item) for item in sample_library.discover_samples()]
    return jsonify({"ok": True, "samples": samples})


@app.get("/api/videos/samples/<video_id>/metadata")
def api_video_sample_metadata(video_id: str):
    sample = sample_library.get_sample(video_id)
    if sample is None:
        return jsonify({"ok": False, "error": "That sample video was not found."}), 404
    return jsonify({"ok": True, "sample": _public_sample(sample)})


@app.get("/api/videos/samples/<video_id>/thumbnail")
def api_video_sample_thumbnail(video_id: str):
    sample = sample_library.get_sample(video_id)
    if sample is None:
        abort(404)
    thumbnail = sample_library.ensure_thumbnail(sample)
    if thumbnail is None:
        thumbnail = sample_library.placeholder_thumbnail(sample["id"], sample["filename"])
    return send_file(
        thumbnail,
        mimetype="image/jpeg",
        conditional=True,
        max_age=3600,
        download_name=f"{sample['id']}.jpg",
    )


@app.get("/api/videos/samples/<video_id>/stream")
def api_video_sample_stream(video_id: str):
    try:
        path = sample_library.resolve_sample_path(video_id)
    except sample_library.SampleVideoError:
        abort(404)
    sample = sample_library.get_sample(video_id)
    return send_file(
        path,
        conditional=True,
        max_age=0,
        download_name=(sample or {}).get("filename") or path.name,
    )


@app.get("/api/videos/uploads/<upload_id>/stream")
def api_video_upload_stream(upload_id: str):
    try:
        upload = video_jobs.resolve_upload(upload_id)
    except video_jobs.UploadError:
        abort(404)
    return send_file(
        Path(upload["path"]),
        conditional=True,
        max_age=0,
        download_name=upload.get("filename") or Path(upload["path"]).name,
    )


@app.post("/api/videos/upload")
def api_video_upload():
    if _content_length() > MAX_VIDEO_BYTES:
        return jsonify({"ok": False, "error": "That video is larger than 256 MB."}), 413

    uploaded = request.files.get("video") or request.files.get("file")
    if uploaded is None or not uploaded.filename:
        return jsonify({"ok": False, "error": "Choose a video file to upload."}), 400

    original = Path(uploaded.filename).name
    if not original:
        return jsonify({"ok": False, "error": "That filename could not be used."}), 400
    if not video_jobs.validate_video_mime(uploaded.mimetype):
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "That file is not recognised as a video (MP4, AVI, MOV, MKV, WEBM).",
                }
            ),
            400,
        )

    try:
        saved = video_jobs.save_upload(uploaded, original)
    except video_jobs.UploadError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400
    except Exception as exc:  # noqa: BLE001
        app.logger.exception("Video upload failed")
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"}), 500

    probe = saved.get("probe") or {}
    return (
        jsonify(
            {
                "ok": True,
                "upload_id": saved["upload_id"],
                "filename": saved["filename"],
                "size_bytes": saved["size_bytes"],
                "width": probe.get("width"),
                "height": probe.get("height"),
                "fps": probe.get("fps"),
                "duration_s": probe.get("duration_s"),
                "frame_count": probe.get("frame_count"),
                "preview_url": f"/api/videos/uploads/{saved['upload_id']}/stream",
                "message": "Video uploaded and validated. Start analysis when ready.",
            }
        ),
        201,
    )


@app.post("/api/videos/analyze")
def api_video_analyze():
    payload = request.get_json(silent=True) or request.form.to_dict()
    source_type = str(payload.get("source_type") or "").strip().lower()

    settings = VideoSettings.from_payload(
        {
            "sample_fps": payload.get("sample_fps"),
            "max_sampled_frames": payload.get("max_sampled_frames"),
            "imgsz": payload.get("imgsz"),
            "conf": payload.get("conf"),
            "min_observations": payload.get("min_observations"),
            "min_ratio": payload.get("min_ratio"),
            "cooldown_s": payload.get("cooldown_s"),
        }
    )

    if source_type == "sample":
        video_id = str(payload.get("video_id") or "").strip()
        sample = sample_library.get_sample(video_id)
        if sample is None:
            return jsonify({"ok": False, "error": "That sample video was not found."}), 404
        job = video_jobs.enqueue_new_job(
            source_type="sample",
            original_filename=sample["filename"],
            source_id=sample["id"],
            settings=settings.as_dict(),
        )
    elif source_type == "upload":
        upload_id = str(payload.get("upload_id") or "").strip()
        try:
            upload = video_jobs.resolve_upload(upload_id)
        except video_jobs.UploadError as exc:
            return jsonify({"ok": False, "error": str(exc)}), 404
        job = video_jobs.enqueue_new_job(
            source_type="upload",
            original_filename=upload.get("filename") or "uploaded video",
            source_id=upload["upload_id"],
            stored_path=f"{upload['upload_id']}/{upload.get('stored_name')}",
            settings=settings.as_dict(),
        )
    else:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "Choose a sample video or upload a video before starting analysis.",
                }
            ),
            400,
        )

    return (
        jsonify(
            {
                "ok": True,
                "job_id": job["id"],
                "status": job["status"],
                "job_url": url_for("video_result_page", job_id=job["id"]),
                "status_url": url_for("api_video_job", job_id=job["id"]),
                "message": "Video analysis queued.",
            }
        ),
        202,
    )


@app.get("/api/videos/jobs/<job_id>")
def api_video_job(job_id: str):
    job = _video_job_or_404(job_id)
    return jsonify({"ok": True, "job": _public_job(job)})


@app.get("/api/videos/jobs/<job_id>/events")
def api_video_job_events(job_id: str):
    _video_job_or_404(job_id)
    return jsonify({"ok": True, "events": storage.list_video_events(job_id)})


@app.get("/api/videos/jobs/<job_id>/result")
def api_video_job_result(job_id: str):
    job = _video_job_or_404(job_id)
    return jsonify(
        {
            "ok": True,
            "job": _public_job(job, include_result=True),
            "events": storage.list_video_events(job_id),
            "summary": job.get("summary") or {},
            "result": job.get("result") or {},
        }
    )


@app.get("/api/videos/jobs/<job_id>/output")
def api_video_job_output(job_id: str):
    job = _video_job_or_404(job_id)
    try:
        path = video_jobs.job_output_path(job)
    except video_jobs.UploadError:
        path = None
    if path is None:
        return jsonify({"ok": False, "error": "The annotated video is not ready yet."}), 404
    return send_file(
        path,
        mimetype="video/mp4",
        conditional=True,
        max_age=0,
        download_name=f"siteguard_{job_id}.mp4",
    )


@app.get("/api/videos/jobs/<job_id>/source")
def api_video_job_source(job_id: str):
    job = _video_job_or_404(job_id)
    if job.get("source_type") == "sample":
        try:
            path = sample_library.resolve_sample_path(str(job.get("source_id") or ""))
        except sample_library.SampleVideoError:
            abort(404)
    elif job.get("source_type") == "upload":
        try:
            path = Path(video_jobs.resolve_upload(str(job.get("source_id") or ""))["path"])
        except video_jobs.UploadError:
            abort(404)
    else:
        abort(404)
    return send_file(
        path,
        conditional=True,
        max_age=0,
        download_name=job.get("original_filename") or path.name,
    )


@app.get("/api/settings")
def api_get_settings():
    return jsonify({"ok": True, "settings": storage.get_settings()})


@app.post("/api/settings")
def api_save_settings():
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        payload = request.form.to_dict()
    settings, errors = storage.save_settings(payload)
    if errors:
        return jsonify({"ok": False, "errors": errors, "settings": settings}), 400
    # Force a fresh model check because the selected chat model may have changed.
    get_assistant().status(force=True)
    return jsonify({"ok": True, "settings": settings})


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "5000")),
        debug=os.environ.get("FLASK_DEBUG") == "1",
    )
