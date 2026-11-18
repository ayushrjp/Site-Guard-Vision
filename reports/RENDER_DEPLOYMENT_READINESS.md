# SiteGuardVision — Render Deployment Readiness

Date: 18 November 2026
Target: Render native Python web service
Entry point: **`server:app`** (verified module-level `app = Flask(__name__)`)

---

## 1. Verdict

| Area | Status |
|---|---|
| Application entry point and WSGI object | **Verified** (`server:app`) |
| Production server configuration | **Prepared** (`gunicorn.conf.py`, 1 worker / 8 threads) |
| Health check | **Verified** (`/health`, returns 200 with model/video/database status) |
| Environment-driven paths | **Implemented and smoke-tested** with temporary directories |
| Persistent storage design | **Documented and wired** (`/var/data`) |
| Debug disabled in production | **Verified** (debug only when `FLASK_DEBUG=1`) |
| Secrets in repository | **None found** |
| Render deployment executed | **No** — this audit ran on Windows only |
| GitHub push | **Blocked** — remote repository not found / no credentials |

**Bottom line:** the code and configuration are ready to attempt a Render
deploy, but Render readiness has **not** been proven by running on Render.
Two external prerequisites remain: a reachable Ollama endpoint (assistant only)
and a GitHub repository that accepts the push. One platform risk is flagged in
§9 (native OpenCV system libraries).

---

## 2. Start command and process model

```bash
gunicorn --config gunicorn.conf.py server:app
```

- **1 Gunicorn worker process, 8 threads.** This is deliberate:
  - the video queue is an in-process `queue.Queue` plus one worker thread; a
    second process would have its own queue and would never see submitted jobs;
  - SQLite is the system of record; one writer process avoids cross-process
    write contention;
  - models are loaded lazily once per process, so extra workers multiply RAM.
- **No `preload_app`** — the video worker must start inside the serving
  process.
- **600 s request timeout** to allow 256 MB video uploads; video inference
  itself runs in a background thread, so normal page requests stay responsive.
- `$PORT` is honoured automatically by `gunicorn.conf.py`.
- Gunicorn is POSIX-only and therefore could not be executed on this Windows
  machine. Its configuration file was syntax-checked; the Flask app and all
  routes were exercised through the development server and test client.

---

## 3. Storage strategy

Render's default filesystem is **ephemeral**: everything outside the mounted
disk is lost on deploy or restart. A 10 GB disk is mounted at `/var/data`.

| Class | Data | Local default | Render value | Why |
|---|---|---|---|---|
| Bundled read-only | `models/*.pt`, `test_videos/*`, `samples/*`, templates, CSS/JS, `evaluation/` | repository | repository | Deployed from Git; ~300 MB of assets, no disk needed |
| Database | `siteguard.db` + WAL/SHM | `data/` | `/var/data/siteguard.db` | Must survive redeploys |
| User uploads | image uploads, video uploads | `static/incoming`, `data/video_uploads` | `/var/data/image_uploads`, `/var/data/video_uploads` | Persistent input |
| Generated output | annotated images, annotated MP4s, thumbnails | `static/rendered`, `data/video_outputs`, `data/video_thumbnails` | `/var/data/rendered`, `/var/data/video_outputs`, `/var/data/video_thumbnails` | Referenced by stored records |
| Assistant | ChromaDB, manifest, uploaded documents | `vector_db/`, `knowledge_base/documents` | `/var/data/vector_db`, `/var/data/knowledge_base` | Persist index and documents |

Only one variable (`SITEGUARD_DATA_DIR=/var/data`) would be enough for the
derived paths; the Blueprint also sets each path explicitly for clarity.

### 3.1 SQLite viability

SQLite is safe for **this** deployment shape:

- single Render instance,
- single Gunicorn worker,
- WAL enabled, 15 s busy timeout,
- all writes are short-lived connections,
- video progress updates are small and serialized.

SQLite is **not** safe if:

- the service is scaled to more than one instance (a disk cannot be shared or
  attached to multiple instances anyway),
- a second process/worker is added,
- high write concurrency is expected.

If any of those become requirements, migrate to Render Postgres (the storage
module would need a small adapter) and move the job queue to a shared service.
No migration was attempted or implied here.

**Backups:** Render snapshots the disk daily. Before destructive maintenance,
export `siteguard.db` (SQLite `.backup`/`VACUUM INTO`) rather than restoring a
disk snapshot of a live database.

### 3.2 Disk constraints

- Persistent disks are available on **paid compute plans only**.
- Attaching a disk disables **zero-downtime deploys** (the old instance is
  stopped before the new one starts) and **horizontal scaling**.
- Disk size can be increased but never decreased. 10 GB leaves room for four
  maximum-size (256 MB) uploads plus outputs and future records.
- Uploaded videos are retained; image uploads are deleted after analysis.

---

## 4. Resource sizing and cost

Verified against `https://render.com/pricing` and
`https://render.com/docs/disks` on 18 November 2026.

| Resource | Requirement | Price |
|---|---|---:|
| Compute `1c-2g` (1 CPU / 2 GB RAM) | Minimum realistic for PyTorch + Ultralytics + ChromaDB | $25.00 / month |
| Persistent disk | 10 GB | $2.50 / month |
| Workspace (Hobby) | 1 workspace | $0 |
| **Total** | | **≈ $27.50 / month** |

- Compute is billed by the second while the instance runs.
- **Free (512 MB) and `0.5c-512mb` plans are not viable**: no disk support and
  insufficient RAM. Loading torch, two YOLO checkpoints and ChromaDB can
  exceed 512 MB before any inference.
- **No GPU** exists on Render. The build installs CPU-only PyTorch wheels.
- 4K clips will take minutes on one CPU and may approach the 2 GB limit. For a
  demonstration, prefer 720p/1080p clips under one minute.

---

## 5. Required Render environment variables

| Key | Value |
|---|---|
| `PYTHON_VERSION` | `3.12.6` (Render's current default is 3.14.10; the pinned ML stack is tested on 3.12) |
| `SITEGUARD_BEHIND_PROXY` | `1` |
| `SITEGUARD_LOG_LEVEL` | `INFO` |
| `WEB_CONCURRENCY` | `1` |
| `SITEGUARD_GUNICORN_THREADS` | `8` |
| `SITEGUARD_DATA_DIR` | `/var/data` |
| `SITEGUARD_DB_PATH` | `/var/data/siteguard.db` |
| `SITEGUARD_IMAGE_UPLOAD_DIR` | `/var/data/image_uploads` |
| `SITEGUARD_RENDERED_DIR` | `/var/data/rendered` |
| `SITEGUARD_VIDEO_UPLOAD_DIR` | `/var/data/video_uploads` |
| `SITEGUARD_VIDEO_OUTPUT_DIR` | `/var/data/video_outputs` |
| `SITEGUARD_VIDEO_THUMB_DIR` | `/var/data/video_thumbnails` |
| `SITEGUARD_VECTOR_DB` | `/var/data/vector_db` |
| `SITEGUARD_KNOWLEDGE_BASE` | `/var/data/knowledge_base` |
| `OLLAMA_BASE_URL` | your reachable Ollama-compatible endpoint (required for assistant; `sync: false` in the Blueprint) |
| `OLLAMA_CHAT_MODEL` | `qwen2.5:3b` |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` |
| `OLLAMA_TIMEOUT` | `300` |

No secret values are committed. `FLASK_DEBUG` is unset (debug off). The
Blueprint prompts for `OLLAMA_BASE_URL` instead of storing it.

---

## 6. Build

```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

- CPU-only torch avoids multi-GB CUDA wheels that Render cannot use.
- `requirements.txt` pins Flask, Ultralytics, Pillow, ChromaDB, pypdf,
  requests, `lapx` (ByteTrack) and Gunicorn.
- Build time is dominated by torch + ChromaDB. Expect several minutes on the
  first deploy; subsequent builds reuse the dependency cache when possible.
- Bundled assets (models + 13 videos) are part of the checkout; the repository
  is large (~300 MB of runtime assets). All individual files are under
  GitHub's 100 MB limit.

---

## 7. Video processing on Render

| Concern | Behaviour |
|---|---|
| Does a long video block the dashboard? | No. Jobs run in one daemon thread; HTTP handlers only enqueue/poll SQLite. |
| Concurrency | One video at a time per instance. |
| Timeout | No per-job timeout or cancellation. Gunicorn's 600 s timeout applies only to HTTP requests. |
| Restart recovery | `ensure_worker()` calls `fail_stale_video_jobs()`; interrupted `running` jobs become `failed`. Queued (not started) jobs are lost. |
| Output storage | Annotated MP4s go to `/var/data/video_outputs`; they are not automatically pruned. Monitor disk usage. |
| CPU / RAM | 4K input is sampled (default max 180 frames) and output is capped at 1280 px. Still CPU-bound; long clips can take minutes and are the main OOM risk. |
| Concurrent requests | Threaded Gunicorn + WAL SQLite; short-lived connections avoid long locks. |

No Redis, Celery or other infrastructure was added. It is not necessary for a
single-instance demonstration and would add cost and failure modes.

---

## 8. Ollama deployment

- `OLLAMA_BASE_URL` is read by `rag/config.py`; models and timeouts are also
  configurable.
- Local Ollama is **not** reachable from Render. Options:
  1. a private Ollama host reachable over HTTPS with authentication;
  2. an Ollama-compatible hosted endpoint. This is an external service — no
     paid provider was added or assumed.
- If the endpoint is missing/unreachable:
  - the assistant page and status API still return 200,
  - chat returns a clear offline answer,
  - image and video analysis are unaffected.
- **Security:** never expose an unsecured Ollama port to the public internet.
  Put TLS and authentication in front of it (reverse proxy or private
  network). Do not put tokens in the repository; use Render environment
  variables.

---

## 9. Risks and mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| **Not yet deployed to Render** | Runtime differences are unproven | Deploy with the Blueprint; watch build and first-request logs |
| Native OpenCV may need `libGL.so.1` on Render's Python image | `import cv2` failure at runtime (could not be tested on Windows) | If logs show it, switch the service to a Docker runtime that installs `libgl1`/`libglib2.0-0`, or use an OpenCV headless build in a controlled image |
| 512 MB/free plan | OOM during import/inference | Use `1c-2g` or larger (Blueprint default) |
| Disk-backed deploy has no zero-downtime | A few seconds of downtime per deploy | Acceptable for a demo; schedule deploys |
| Single instance + SQLite | No horizontal scaling | Upgrade to Postgres + shared queue if scaling is required |
| Video queue lost on restart | Queued jobs disappear; running jobs fail | Re-submit after deploy; documented in UI/README |
| 4K video time/RAM | Slow or OOM | Use 1080p clips; lower sampled frames/output width in job settings |
| Ollama unreachable | Assistant offline | Expected degradation; configure a reachable endpoint |
| Model false positives | Misleading events | Temporal verification + confidence thresholds; results are advisory |

---

## 10. Exact deployment steps

1. Create the GitHub repository `ayushrjp/SiteGuardVision` (or restore access)
   and push branch `main`.
2. In Render: **New → Blueprint**, connect the repository, select `main`.
   Render reads `render.yaml`.
3. Provide `OLLAMA_BASE_URL` when prompted (or leave it blank for offline
   assistant mode).
4. Deploy and wait for the build (CPU torch + ML wheels).
5. Confirm:
   - `/health` returns `status: ok`, `models: present`,
     `sample_videos: present`;
   - `/` and `/videos` load;
   - a sample image analysis produces an annotated image;
   - a short 1080p sample video completes and downloads.
6. Check the disk usage after the first video. Increase `sizeGB` before it
   fills; disk size cannot be decreased.
7. For a custom domain, add it in Render and update DNS. TLS is automatic.

**Manual start command if not using the Blueprint:**

```bash
gunicorn --config gunicorn.conf.py server:app
```

---

## 11. What was verified locally vs. what remains

**Verified locally**

- `server:app` imports and serves every page.
- Runtime path overrides work end-to-end (temp DB, temp renders, temp video
  paths) with no writes to production data.
- `/health`, image analysis, annotated image serving, 13 sample videos,
  assistant offline mode, DB integrity and file references.
- 113/113 tests pass.

**Not verified locally**

- Gunicorn execution (POSIX-only; Windows host).
- Render's build environment, native OpenCV system libraries and actual
  memory headroom.
- A real remote Ollama endpoint.
- GitHub push (repository not found / no credentials).

These gaps are the reason the project is described as **code-ready but not
deployment-proven**.
