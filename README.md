# SiteGuardVision

SiteGuardVision is a Flask application for construction-site safety monitoring.
It detects people and Personal Protective Equipment (PPE) in uploaded photos,
analyses uploaded or bundled sample videos with persistent worker tracking and
timestamped safety events, and offers a retrieval-augmented **Safety Assistant**
that can explain safety practices and detection results.

Everything runs locally on CPU. No cloud vision API, paid service or GPU is
required.

---

## Contents

- [Architecture](#architecture)
- [Features](#features)
- [Models and bundled assets](#models-and-bundled-assets)
- [Installation](#installation)
- [Running locally](#running-locally)
- [Image analysis](#image-analysis)
- [Video analysis](#video-analysis)
- [Sample videos](#sample-videos)
- [Safety Assistant (Ollama)](#safety-assistant-ollama)
- [Environment variables](#environment-variables)
- [Tests](#tests)
- [Deploying to Render](#deploying-to-render)
- [Storage layout](#storage-layout)
- [Known limitations](#known-limitations)
- [Project structure](#project-structure)

---

## Architecture

```
Browser
  │  HTML (Jinja) + vanilla JS
  ▼
Flask  server.py  ──────────────────────────────────────────────┐
  │                                                             │
  ├── Image pipeline  infer_engine.py                           │
  │     ├── models/person_yolov8n.pt   (person detection)       │
  │     ├── models/yolo_ppe_core.pt    (PPE detection)          │
  │     └── worker ↔ PPE association + evidence-based verdict   │
  │                                                             │
  ├── Video pipeline  video_processing/                         │
  │     ├── sample_library.py  discovery, metadata, thumbnails  │
  │     ├── jobs.py            uploads + single background queue│
  │     └── video_engine.py    sampling, ByteTrack, events, MP4 │
  │                                                             │
  ├── Safety Assistant  rag/                                    │
  │     ├── embeddings.py       Ollama chat + embeddings        │
  │     ├── vector_store.py     ChromaDB + manifest             │
  │     └── safety_assistant.py grounded answers / offline mode │
  │                                                             │
  └── Persistence  storage.py (SQLite) + runtime_config.py       │
```

- **Image analysis is independent** of video processing and of Ollama.
- **Video jobs run in one background thread**; HTTP polling reads progress
  from SQLite, so the dashboard stays responsive while a video is processed.
- **SQLite is the system of record** for image analyses, video jobs/events and
  settings. WAL mode and a 15 s busy timeout are enabled.

---

## Features

### Image PPE analysis
- Upload JPG/PNG/WEBP/BMP (max 15 MB) or pick a bundled sample photo.
- Separate person detector and PPE detector; a person box and PPE box are
  associated before a verdict is produced.
- Detects hard hats, safety vests, gloves, goggles, masks and the matching
  `no-*` violation classes, plus safety cones and ladders.
- Evidence-based assessment: `potential_violation`,
  `no_violation_detected` or `insufficient_evidence` (zero people is never
  treated as compliance).
- Annotated image output and a stored analysis record.
- Adjustable confidence (0.05–0.95) and inference size (640/960/1280).

### Video PPE analysis
- Discover sample videos from `test_videos/` with real OpenCV metadata and
  cached thumbnails.
- Upload videos up to 256 MB (MP4/AVI/MOV/MKV/WEBM/M4V), validated by
  extension, MIME type and actual decodability.
- Frame sampling with configurable maximum (default 180 sampled frames).
- Per-frame person and PPE detection, filtered to the person class where
  required.
- ByteTrack worker tracking with persistent worker IDs inside a job.
- Temporal verification of PPE state across frames to reduce single-frame
  false positives.
- Safety events with timestamps, worker IDs and severity.
- Annotated MP4 rendering (H.264, width capped at 1280 by default).
- Job progress, result dashboard, original playback and annotated download.
- One job at a time; queued jobs are recovered as failed after a restart.

### Safety Assistant
- Chat over a bundled/local knowledge base using Ollama for generation and
  embeddings, with ChromaDB as the vector store.
- Answers can be grounded in a stored image/video result (explicit IDs, no
  hallucinated measurements).
- Document upload (PDF/TXT/MD), re-index and delete.
- When Ollama is unavailable the assistant returns a clear offline message
  instead of crashing; the rest of the application keeps working.

### Operational pages
- Dashboard with recent analyses and statistics.
- Reports with CSV export and stored analysis detail pages.
- Dataset & Evaluation page reading real evaluation JSON.
- Settings for thresholds, inference size and assistant model.

---

## Models and bundled assets

| Asset | Purpose | Size |
|---|---|---|
| `models/yolo_ppe_core.pt` | PPE detector (14 classes) | ~49.6 MB |
| `models/person_yolov8n.pt` | Dedicated person detector (COCO YOLOv8n) | ~6.2 MB |
| `yolov8n.pt` (project root) | Legacy fallback for the person detector | ~6.2 MB |
| `test_videos/` | 13 sample videos (1080p/4K/vertical clip) | ~231 MB |
| `samples/` | Bundled sample photos | ~7.5 MB |

The checkpoints are used as-is. This project does not retrain or replace them.
All individual files are below GitHub's 100 MB per-file limit.

---

## Installation

Requirements: Python 3.12 (tested with 3.12.6), a C++-free CPU environment,
and ~2 GB of free RAM for image analysis (more for 4K video).

```powershell
cd C:\Users\ayush\Downloads\SiteGuardVision

# 1. Create and activate a virtual environment (already present in this copy)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> On Windows, `gunicorn` is listed only for Render. Use the Flask development
> server locally.

### Optional: Safety Assistant

Install [Ollama](https://ollama.com), then pull the default models:

```powershell
ollama pull qwen2.5:3b
ollama pull nomic-embed-text
```

The assistant is optional: every other page works without Ollama.

---

## Running locally

```powershell
.\.venv\Scripts\python.exe server.py
```

Open <http://127.0.0.1:5000>.

The development server binds to `127.0.0.1:5000` by default. Set `HOST`, `PORT`
or `FLASK_DEBUG=1` only for local experimentation.

Production entry point: `server:app` (module-level Flask object in
`server.py`).

Health check: <http://127.0.0.1:5000/health>.

---

## Image analysis

1. Open **Image Analysis** (`/`).
2. Choose a sample photo or upload an image (max 15 MB).
3. Optionally adjust confidence and inference size.
4. Select **Analyze**.
5. The result page shows the annotated image, per-class counts, the assessment
   and its reasons. The analysis is saved to Reports (except under `TESTING`).

Thresholds can be changed in **Settings**. Defaults: confidence `0.25`,
inference size `960`.

---

## Video analysis

1. Open **Video Analysis** (`/videos`).
2. Pick a sample video or upload one (max 256 MB).
3. Choose detection settings (confidence, sampled frames, output width).
4. Start analysis and watch progress on the job page.
5. When the job completes, the result dashboard shows:
   - verdict and summary counts,
   - timestamped safety events per worker,
   - original video playback,
   - annotated video playback and download.

Only one video is processed at a time. Queued jobs are lost on process restart;
a job left `running` by a crash is marked failed at startup.

---

## Sample videos

The 13 videos in `test_videos/` are discovered dynamically — no manifest needs
to be edited. Metadata and thumbnails are read with OpenCV.

```text
test_videos/
  13569683-hd_1920_1080_60fps.mp4
  13753913_3840_2160_50fps.mp4
  14626386_2160_3840_30fps.mp4
  16533594_1920_1080_25fps.mp4
  19832490-hd_1920_1080_25fps.mp4
  5434218-hd_1920_1080_24fps.mp4
  5434223-hd_1920_1080_24fps.mp4
  8486681-uhd_3840_2160_30fps.mp4
  8964930-uhd_3840_2160_25fps.mp4
  8965526-uhd_2160_3840_25fps.mp4
  Recording 2026-11-18 093655.mp4
  Recording 2026-11-18 093828.mp4
  Recording 2026-11-18 093935.mp4
```

To use another folder locally, set `SITEGUARD_SAMPLE_VIDEO_DIR`. The directory
is allowlisted and every request path is resolved inside it.

---

## Safety Assistant (Ollama)

The assistant talks to any Ollama-compatible HTTP endpoint:

```powershell
$env:OLLAMA_BASE_URL = "http://localhost:11434"
$env:OLLAMA_CHAT_MODEL = "qwen2.5:3b"
$env:OLLAMA_EMBED_MODEL = "nomic-embed-text"
```

- If Ollama is unreachable, `/api/assistant/status` reports offline and chat
  returns a clear offline answer. Image/video analysis is unaffected.
- For a remote host, terminate TLS and authentication in front of Ollama
  (reverse proxy or private network). **Never expose an unsecured Ollama port
  to the public internet**, and never commit credentials or tokens.
- Uploaded documents (PDF/TXT/MD) are chunked, embedded and stored in
  ChromaDB. Re-index after changing the embedding model.

---

## Environment variables

All variables are optional locally. `.env.example` documents the same set with
placeholders.

| Variable | Default | Purpose |
|---|---|---|
| `SITEGUARD_DATA_DIR` | `./data` | Base directory for writable state (set to `/var/data` on Render) |
| `SITEGUARD_DB_PATH` | `<data>/siteguard.db` | SQLite database path |
| `SITEGUARD_IMAGE_UPLOAD_DIR` | `./static/incoming` | Temporary image uploads |
| `SITEGUARD_RENDERED_DIR` | `./static/rendered` | Annotated images |
| `SITEGUARD_VIDEO_UPLOAD_DIR` | `<data>/video_uploads` | Uploaded videos |
| `SITEGUARD_VIDEO_OUTPUT_DIR` | `<data>/video_outputs` | Annotated MP4s |
| `SITEGUARD_VIDEO_THUMB_DIR` | `<data>/video_thumbnails` | Cached thumbnails |
| `SITEGUARD_SAMPLE_VIDEO_DIR` | `./test_videos` | Read-only sample library |
| `SITEGUARD_VECTOR_DB` | `./vector_db` | ChromaDB directory |
| `SITEGUARD_KNOWLEDGE_BASE` | `./knowledge_base/documents` | Assistant documents |
| `SITEGUARD_COLLECTION` | `safety_documents` | Chroma collection name |
| `SITEGUARD_TOP_K` | `5` | Retrieved chunks per answer |
| `SITEGUARD_MIN_SIMILARITY` | `0.18` | Retrieval cut-off |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b` | Generation model |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `OLLAMA_TIMEOUT` | `300` | Chat request timeout (seconds) |
| `OLLAMA_STATUS_TIMEOUT` | `2.5` | Status probe timeout (seconds) |
| `SITEGUARD_BEHIND_PROXY` | `0` | `1` trusts one `X-Forwarded-*` hop (Render) |
| `SITEGUARD_LOG_LEVEL` | `INFO` | Python/Gunicorn log level |
| `FLASK_DEBUG` | `0` | `1` enables the local debugger only |
| `HOST` / `PORT` | `127.0.0.1` / `5000` | Development server bind |
| `WEB_CONCURRENCY` | `1` | Gunicorn workers (keep at 1) |
| `SITEGUARD_GUNICORN_THREADS` | `8` | Gunicorn threads |

---

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The suite currently contains **113 tests** covering the image pipeline, Flask
routes, video engine/jobs/samples and the RAG assistant. Tests use temporary
databases and directories; they do not delete production data.

---

## Deploying to Render

### Why a persistent disk is mandatory

Render's default filesystem is **ephemeral**. Without a disk, the SQLite
database, uploads, annotated images/videos, thumbnails and the Chroma index are
lost on every deploy or restart. Models and sample videos are baked into the
Git checkout and do not need a disk.

`render.yaml` defines:

- one `1c-2g` instance (1 CPU, 2 GB RAM) — no GPU,
- one Gunicorn worker with 8 threads (`gunicorn.conf.py`),
- a 10 GB persistent disk mounted at `/var/data`,
- all writable paths pointed at `/var/data`,
- `SITEGUARD_BEHIND_PROXY=1` and log level `INFO`,
- health check at `/health`.

### Steps

1. Push the repository to GitHub (see the final audit report for the exact
   state of the remote).
2. In Render, choose **New → Blueprint**, select the repository and branch
   `main`. Render reads `render.yaml`.
3. Set `OLLAMA_BASE_URL` to a reachable Ollama endpoint. If you do not have
   one, deploy anyway: the assistant reports offline and all detection
   features work.
4. Wait for the build. CPU PyTorch wheels are installed first, then
   `requirements.txt`.
5. Open the service URL and check `/health`.

### Expected cost (checked 18 Nov 2026)

| Item | Price |
|---|---|
| `1c-2g` compute (1 CPU / 2 GB) | $25.00 / month |
| 10 GB persistent disk | $2.50 / month |
| Hobby workspace | $0 |
| **Approximate total** | **$27.50 / month** |

Render bills compute by the second while an instance is running. The free and
`0.5c-512mb` plans cannot attach a disk and do not have enough RAM for
PyTorch + Ultralytics + ChromaDB. Adding a disk also disables zero-downtime
deploys and horizontal scaling — both are intentional constraints of this
architecture.

### Manual equivalent start command

```bash
gunicorn --config gunicorn.conf.py server:app
```

---

## Storage layout

| Class | Examples | Location | Persistence |
|---|---|---|---|
| Bundled read-only | models, sample photos/videos, templates, CSS/JS | repository | Redeployed from Git |
| Database | `siteguard.db`, WAL/SHM | `SITEGUARD_DB_PATH` | Persistent disk |
| User uploads | temporary image uploads, video uploads | upload dirs | Persistent disk (images are deleted after analysis) |
| Generated output | annotated images, annotated MP4s, thumbnails | rendered/output/thumb dirs | Persistent disk |
| Assistant index | ChromaDB files, manifest, documents | vector DB / knowledge base dirs | Persistent disk |

Do not place user data inside `static/rendered` or `data/` in Git — both are
ignored by `.gitignore`.

---

## Known limitations

- **CPU only.** A 4K, 60 fps clip is slow (minutes) and 4K output is downscaled
  to at most 1280 px wide by default. Prefer 1080p clips under a minute for
  demonstrations.
- **Single video at a time.** The queue is an in-process thread. There is no
  Celery/Redis, no cancellation and no per-job timeout. Restarting the service
  loses queued jobs and marks interrupted running jobs failed.
- **SQLite + one instance.** The deployment shape is one web instance with one
  Gunicorn worker. Scaling horizontally requires PostgreSQL (and a shared job
  queue) instead.
- **ByteTrack fragmentation.** Busy scenes can split one worker into several
  track IDs; events remain per-track and are not globally de-duplicated.
- **Heuristic temporal verification.** PPE state changes across sampled frames
  are smoothed, but short occlusions can still produce `unconfirmed` or
  low-confidence events. A low-confidence vest event was observed on one
  sample clip.
- **Assistant quality depends on the model and documents.** Without Ollama the
  assistant is offline by design. The bundled knowledge base is intentionally
  small; upload site-specific documents for better answers.
- **SQLite backups are not automatic.** Render snapshots the disk daily, but
  export the database before destructive maintenance.

---

## Project structure

```text
.
├── server.py                  Flask app, routes, health check
├── infer_engine.py            image inference + annotation
├── storage.py                 SQLite persistence
├── runtime_config.py          environment-driven writable paths
├── evaluation_loader.py       evaluation JSON loader
├── gunicorn.conf.py           production WSGI configuration
├── render.yaml                Render Blueprint
├── requirements.txt
├── models/                    YOLO checkpoints (read-only)
├── test_videos/               13 sample videos (read-only)
├── samples/                   sample photos
├── video_processing/          sample library, job queue, video engine
├── rag/                       Ollama client, Chroma store, assistant
├── templates/                 Jinja templates
├── static/                    CSS, JS, runtime uploads/renders
├── tests/                     113 unit/integration tests
├── evaluation/                evaluation reports shown in the UI
├── reports/                   audit, validation and deployment reports
└── data/                      local SQLite DB and video outputs
```
