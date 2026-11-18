# SiteGuardVision — Final Project Audit

Date: 18 November 2026
Auditor role: senior engineer / QA / DevOps review
Project root: `C:\Users\ayush\Downloads\SiteGuardVision`

> All results below were produced by executing the project in this folder.
> Nothing in this report is inferred from filenames or from the presence of a
> configuration file.

---

## 1. Project root and repository state

| Item | Result |
|---|---|
| Actual working directory | `C:\Users\ayush\Downloads\SiteGuardVision` |
| Nested duplicate `SiteGuardVision/` folder | None found |
| Git repository at audit start | **None** (no `.git` in the project or any parent up to `C:\Users\ayush\Downloads`) |
| Current branch after `git init` | `main` |
| Git repository root | `C:\Users\ayush\Downloads\SiteGuardVision` |
| Remote configured | `origin` → `https://github.com/ayushrjp/SiteGuardVision.git` |

Required components verified present in the main folder: `server.py`,
`infer_engine.py`, `storage.py`, `video_processing/`, `models/`, `templates/`,
`static/`, `test_videos/`, `tests/`, `reports/`, `requirements.txt`, `README.md`.

The video implementation is integrated into the existing Flask application:
`server.py` imports `video_processing.jobs`, `video_processing.sample_library`
and `video_processing.video_engine`, and serves the video pages and APIs
(`/videos`, `/videos/jobs/<id>`, `/api/videos/*`). It is not a separate app.

---

## 2. Changes made in this audit

### Files created (8)

| File | Purpose |
|---|---|
| `runtime_config.py` | Environment-driven writable paths (`SITEGUARD_DATA_DIR` etc.) with local defaults unchanged |
| `.gitignore` | Excludes venv, caches, secrets, runtime data, training artifacts |
| `.env.example` | Placeholder environment template (no secrets) |
| `gunicorn.conf.py` | Production WSGI config: 1 worker, 8 threads, 600 s timeout |
| `render.yaml` | Render Blueprint with persistent disk, env vars, health check |
| `reports/PROJECT_CLEANUP_PLAN.md` | Cleanup plan written before deletion |
| `reports/FINAL_PROJECT_AUDIT.md` | This report |
| `reports/RENDER_DEPLOYMENT_READINESS.md` | Deployment analysis and constraints |

### Files modified (9)

| File | Change |
|---|---|
| `server.py` | Runtime paths, `/rendered/<file>` route, `annotated_url` template helper, logging config, ProxyFix, production 500 handler, richer health check, env `HOST`/`PORT`, debug forced off unless `FLASK_DEBUG=1` |
| `infer_engine.py` | Annotated images written to `runtime_config.RENDERED_DIR` |
| `storage.py` | Default DB path comes from `runtime_config.DB_PATH` |
| `video_processing/jobs.py` | Default upload/output dirs follow `SITEGUARD_DATA_DIR` |
| `video_processing/sample_library.py` | Default thumbnail dir follows `SITEGUARD_DATA_DIR` |
| `requirements.txt` | Added `lapx==0.13.1` (ByteTrack) and `gunicorn==23.0.0` |
| `templates/analysis.html` | Annotated image uses `annotated_url()` (works off `static/`) |
| `templates/home.html` | Recent-analysis thumbnails use `annotated_url()` |
| `README.md` | Full rewrite (architecture, features, run/deploy, limitations) |

All changes are inside the main project folder. No duplicate application was
created.

---

## 3. Functional audit

### A. Image PPE detection — PASS

- Both checkpoints exist and load: `yolo_ppe_core.pt` (49.6 MB) and
  `person_yolov8n.pt` (6.2 MB). Load verified in-process and by
  `TestRealInference`.
- Separate person and PPE detectors are used; the PPE checkpoint is not
  retrained or modified.
- Upload, format validation, readability validation, worker↔PPE association,
  evidence-based assessment and annotated image generation all covered by
  tests and by the live smoke test.
- Real inference on a sample produced an annotated JPEG served through
  `/rendered/<file>` with `Content-Type: image/jpeg` (182,978 bytes).
- Dashboard result consistency verified: rendered status matches
  `result.assessment_status` for all three verdicts.

### B. Video PPE detection — PASS

Verified by 50 video tests plus a real end-to-end video test:

- Sample discovery, OpenCV metadata and cached thumbnails: PASS (13 samples).
- Upload validation (extension, MIME, decodability) and cleanup: PASS.
- Job creation, progress persistence and completion: PASS.
- Frame sampling bounded by `max_sampled_frames`; 4K/60 fps is not processed
  frame-by-frame: PASS.
- Person detection, PPE detection, person filtering and ByteTrack worker IDs:
  PASS (`lapx` present).
- Temporal verification, timestamps, event persistence and summary counts:
  PASS.
- Annotated MP4 generated and playable (H.264 `avcC` verified in the existing
  validation report): PASS.
- Original and annotated playback, range requests, download endpoint: PASS.
- The OpenH264 DLL warning during tests is benign: OpenCV falls back to
  another H.264 encoder and the output is playable.

### C. Safety Assistant — PASS (with external dependency)

- `OLLAMA_BASE_URL`, chat/embed models, timeouts, vector DB and knowledge-base
  paths are environment-configurable.
- Chat API, session IDs, document listing/upload/reindex/delete and settings
  are covered by `test_rag_assistant.py`.
- Offline behaviour verified twice: with the configured endpoint unreachable,
  `/assistant` and `/api/assistant/status` return 200 and the chat layer
  returns an explicit offline message. Detection features are unaffected.
- No credentials are stored in the repository.

### D. Database and storage — PASS

| Check | Result |
|---|---|
| Engine | SQLite (WAL, busy timeout 15 s) |
| `PRAGMA integrity_check` | `ok` |
| Tables | `analyses`, `video_jobs`, `video_events`, `settings` |
| Image analyses | 8 |
| Video jobs | 4 (all completed) |
| Video events | 10 |
| Stored analyses with missing annotated image | 0 |
| Stored jobs with missing MP4 output | 0 |
| Production DB opened destructively | No |

### E. Frontend — PASS

Live server checks (isolated storage): `/`, `/videos`, `/reports`,
`/evaluation`, `/settings`, `/assistant` all returned 200. Navigation,
sample cards, upload forms, result dashboards and the sample library render.
No browser console capture was performed in this environment; the pages were
validated via HTTP and the JavaScript files are static assets with no test
harness. Responsive layout was reviewed in the templates, not on a device
matrix.

---

## 4. Test results

### Full suite (after cleanup)

```
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
Ran 113 tests in 56.406s
OK
```

- No import errors, syntax errors, missing dependencies or broken routes.
- No image-pipeline regression.
- No video-pipeline regression.

### Isolated Flask startup and smoke test

The real development server was started as a subprocess with a temporary
database and temporary upload/render/output directories. The production
database was not opened for writes.

```
10/10 checks passed
GET / 200 · /videos 200 · /reports 200 · /evaluation 200 · /settings 200
GET /assistant 200 (Ollama offline)
GET /health -> status=ok, models=present, sample_videos=present
GET /api/videos/samples -> 13 samples + metadata
POST /analyze -> 200, annotated image served (182,978 bytes)
GET /api/assistant/status -> 200 offline
```

---

## 5. Models and original assets

| Asset | Bytes | SHA-256 |
|---|---:|---|
| `models/yolo_ppe_core.pt` | 52,056,210 | `2b33d4d016f9751c5a25f4a72ce050e0a7e4e140b11c1669978d7154003a4f61` |
| `models/person_yolov8n.pt` | 6,549,796 | `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` |
| `test_videos/` (13 files) | 242,607,738 | — |
| Largest sample video | 79,197,915 (`14626386_2160_3840_30fps.mp4`) | — |
| `yolov8n.pt` (legacy person fallback) | 6,549,796 | preserved |

All 13 sample videos remain available. No model was retrained, replaced,
relabelled or threshold-changed. No sample video, original sample image,
production record or evaluation report was deleted.

---

## 6. Cleanup performed

Plan: `reports/PROJECT_CLEANUP_PLAN.md` (written before deletion).

### Removed (confirmed safe)

- 10 `__pycache__/` directories outside `.venv` (31 `.pyc` files, ~661 KB).
- `project_structure.txt` (3.72 MB raw `tree` output including `.venv`,
  unreferenced).

**Disk space saved: 4.36 MB.**

### Preserved deliberately

- `static/rendered/` (317 files, 55.2 MB) and `static/incoming/` (24 files,
  4.3 MB): user data / annotated outputs.
- `data/siteguard.db`, `data/video_outputs/`, `data/video_thumbnails/`.
- Models, all 13 sample videos, `samples/`, evaluation reports, validation
  reports and logs.
- `dataset/`, `runs/`, `annotation_audit/`, `diagnostics/`: preserved on disk,
  excluded from Git because they are training/audit-only and not read at
  runtime.

### Uncertain (not deleted)

`static/incoming/` is listed for manual review. The application deletes
successful image uploads, but 24 files from earlier sessions remain. They were
kept because they may be referenced by manual workflows; `.gitignore` excludes
them from the repository.

### Verification after cleanup

The 113-test run and the 10-check live smoke test above were both executed
**after** the deletions. Model loading, sample discovery, DB integrity and all
13 videos were re-verified.

---

## 7. Known issues and caveats

1. **Tracker fragmentation** on busy clips can split one worker into multiple
   track IDs (observed: 10 workers in 9 seconds). Events are per-track.
2. **One low-confidence false-positive risk**: a vest event at confidence
   0.4235 with 2 observations was produced on
   `19832490-hd_1920_1080_25fps.mp4`. Temporal verification reduces but does
   not eliminate this.
3. **Video queue is in-process.** No cancellation, no per-job timeout; queued
   jobs are lost on restart and interrupted running jobs are marked failed.
4. **4K video is CPU-heavy** on Render. Defaults cap sampling at 180 frames
   and output width at 1280 px.
5. **Flask debug** is off by default; only `FLASK_DEBUG=1` enables it.
6. **No automated browser console test** was possible in this environment.

---

## 8. Git and GitHub push

- Branch: `main` (created locally).
- Remote: `https://github.com/ayushrjp/SiteGuardVision.git`.
- Local commit created with the full verified tree (303 files, 337.2 MB,
  largest file 75.53 MB). No file exceeds GitHub's 100 MB limit, so Git LFS is
  not required (`git-lfs 3.6.1` is installed but no `.gitattributes` tracking
  was added).
- Push: **failed / blocked**. Both `git ls-remote origin` and
  `git push -u origin main` returned exit code 128:

  ```
  remote: Repository not found.
  fatal: repository 'https://github.com/ayushrjp/SiteGuardVision.git/' not found
  ```

- No GitHub CLI (`gh` absent), no `GITHUB_TOKEN`/`GH_TOKEN`, and no configured
  credential helper. The repository must exist and be authorized by the owner
  before any push can succeed. Git terminal prompting was disabled for the
  attempt so it failed immediately instead of hanging.
- No force push was attempted. No remote change was overwritten.
- See the final chat report for the exact commit hash and next steps.
- Excluded from Git by design (present on disk): `.venv/`, `data/`,
  `static/incoming/`, `static/rendered/`, `vector_db/`, `dataset/`, `runs/`,
  `annotation_audit/`, `diagnostics/` and caches.

---

## 9. Overall verdict

The application is functionally complete and passes its full test suite and a
live isolated smoke test. Code-level Render preparation is in place and the
storage architecture matches Render's disk constraints. Deployment itself has
**not** been executed on Render; the readiness assessment and remaining risks
are documented in `reports/RENDER_DEPLOYMENT_READINESS.md`.
