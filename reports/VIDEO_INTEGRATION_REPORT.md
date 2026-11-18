# Video PPE Integration Report

**Project:** SiteGuardVision
**Scope:** Uploaded-video and bundled-sample PPE analysis added alongside the existing image workflow
**Date:** 18 November 2026
**Validation host:** Windows, Python 3.12.6 (`.venv`), OpenCV FFmpeg build, CPU inference

---

## 1. Summary

The image-based PPE system is unchanged. A separate video pipeline now:

- discovers the 13 bundled clips in `test_videos/` without hardcoded filenames;
- validates uploaded video by extension, MIME type, size, and actual OpenCV decodability;
- samples frames at a configurable rate and hard frame budget instead of decoding every source frame;
- tracks people with Ultralytics ByteTrack and remaps tracker IDs to a per-job sequence starting at `#1`;
- associates Hardhat / Safety Vest / Gloves / Goggles / Mask and their `NO-*` counterparts with workers geometrically;
- keeps three evidence states — **Compliant**, **Potential Violation**, **Unconfirmed** — and never promotes Unconfirmed into another state;
- emits a safety event only after explicit negative observations pass configurable temporal verification, with first/last seen times, worker ID, category, confidence, and evidence text;
- writes a browser-playable annotated MP4 with worker boxes/IDs, PPE boxes, compliance state, violation labels, and timestamps;
- persists job metadata and events in SQLite using **additive** tables only;
- exposes sample, upload, analysis, status, event, result, and safe playback APIs;
- renders a result dashboard with the original name, processing metrics, worker counts, a timestamped event table, original/annotated players, and a return-to-library action.

**Verification status**

| Check | Result |
|---|---|
| Full automated suite | **113 tests passed, exit 0** (`reports/video_full_test_run.log`) |
| Video-specific tests | **50 passed** (`tests/test_video_processing.py`) |
| Real sample validation | **13 of 13 videos processed**; **13 of 13 annotated outputs readable** |
| Real UI verification | Sample selection, upload validation, analysis progress, result dashboard, both players, and event seek exercised in a browser |
| Existing image workflow | Existing tests still pass; no image thresholds, labels, or checkpoints changed |

---

## 2. Files created and modified

### Created

| Path | Purpose |
|---|---|
| `video_processing/__init__.py` | Package marker and public helpers |
| `video_processing/sample_library.py` | Safe sample discovery, OpenCV metadata probe, thumbnails/placeholders, report-derived categories |
| `video_processing/video_engine.py` | Sampling plan, ByteTrack worker tracking, PPE detection/association, temporal verification, annotation, video writer |
| `video_processing/jobs.py` | Upload validation/storage, safe upload resolution, deletion, single background worker queue, job execution |
| `templates/videos.html` | Video workspace: upload, sample library, preview, settings, job history |
| `templates/video_result.html` | Result dashboard: verdict, KPIs, annotated/original playback, event table, worker table |
| `static/js/videos.js` | Sample selection, upload progress, settings, analysis polling, live job refresh, event seeking |
| `tests/test_video_processing.py` | 50 video tests (pure logic, library, routes, worker, real model) |
| `reports/VIDEO_INTEGRATION_REPORT.md` | This report |
| `reports/video_validation_results.json` | Real per-video validation metrics and events |
| `reports/video_validation_run.log` | Console log of the 13-video validation run |
| `reports/video_full_test_run.log` | Console log of the full test suite |

### Modified

| Path | Change |
|---|---|
| `server.py` | `/videos` and `/videos/jobs/<id>` pages; all video APIs; 256 MB global limit with explicit 15 MB checks on image/document routes; 413 handling; safe Range-capable streaming; `video_state` filter and `video_time` helper |
| `storage.py` | Additive `video_jobs` and `video_events` tables, CRUD, stale-job recovery; existing tables untouched |
| `infer_engine.py` | Added `prediction_lock()` so video prediction and image prediction cannot race; loaders, labels, and association logic reused |
| `templates/layout.html` | Video navigation entry, active state, `videos.js` include |
| `templates/_icons.html` | Added `video` and `play` icons |
| `static/css/siteguard.css` | Video workspace/result styles; global `[hidden] { display: none !important; }` fix |

No checkpoints, class labels, image thresholds, existing images, the production database, or ChromaDB data were changed or deleted.

---

## 3. Architecture

```
Browser
  ├─ /videos                     sample list + upload + settings + job history
  └─ /videos/jobs/<job_id>       live progress -> result dashboard
        │
        ▼
Flask (server.py)
  ├─ sample API  -> sample_library.py -> test_videos/ (allowlisted, ID-addressed)
  ├─ upload API  -> jobs.save_upload() -> data/video_uploads/<upload_id>/
  └─ analyze API -> storage.create_video_job() -> single worker queue
        │
        ▼
Background worker (jobs.py, one job at a time)
  └─ video_engine.process_video()
       ├─ probe + sampling plan (fps, stride, frame budget, output size)
       ├─ per sampled frame:
       │    ├─ ByteTrack person tracking (person checkpoint, class person)
       │    ├─ PPE detection (existing PPE checkpoint)
       │    ├─ geometric association (head/torso/hands regions)
       │    └─ temporal observation recording (present / missing / conflict / unknown)
       ├─ temporal verification (runs, min observations, support ratio, cooldown)
       ├─ annotated MP4 writer (avc1, mp4v fallback)
       └─ persist job result + events
        │
        ▼
SQLite (additive video_jobs / video_events) + data/video_outputs/<job_id>.mp4
```

### Sampling and resource control

- Default `sample_fps = 2.0`, hard cap `max_sampled_frames = 180`.
- Source stride = `round(source_fps / sample_fps)`; if the cap would be exceeded, the stride is raised so the budget is respected.
- Inference default `imgsz = 640` (allowed 640 / 960 / 1280); source aspect ratio is preserved.
- Output width caps at 1280 px; height is derived, and both dimensions are forced even for codec compatibility.
- Output FPS = `source_fps / stride`, so playback duration matches the source.
- One worker thread processes one job at a time; Flask request handlers only read/write short-lived SQLite connections.
- `cv2.VideoCapture` / `VideoWriter` are released in `finally` blocks; a global prediction lock serializes model access with image inference.

### Tracking and association

- A dedicated person-model instance with ByteTrack is used for video; the image engine state is not disturbed.
- Only class `person` is accepted; detections below the person confidence floor are dropped.
- Tracker state is reset in place between jobs (`tracker.reset()`), not by nulling `predictor.trackers`.
- Raw tracker IDs are remapped per job to `#1..N` in first-seen order.
- Hard hat/vest association reuses `infer_engine.associate_detections()`.
- A video-only contradiction pass attaches an opposite-polarity hat/vest box to a worker that already has a positive one, so genuine conflict is visible instead of silently dropped.
- Gloves use hand regions; goggles and masks use head regions; each slot keeps separate positive/negative polarity keys.

### Temporal verification

- Observations are recorded per `(worker, slot)` with timestamp and frame index.
- A run must contain at least `min_observations` explicit negative observations.
- The surrounding window (last observation before, first after) must meet `min_ratio`.
- Cooldown merges adjacent qualified runs instead of creating duplicate alerts; merged evidence text is rebuilt to match the merged totals.
- Absence of a positive PPE box alone never creates an event.
- Worker status:
  - **Potential Violation** when at least one verified event exists;
  - **Compliant** only when hard hat and vest are `present` with enough observations and no verified event;
  - **Unconfirmed** otherwise, including conflicts and missing evidence.

### Annotated output

- Worker box, `#ID`, overall state, and per-slot letters (`H`, `V`, `G`, `E`, `M`).
- PPE boxes with label and confidence.
- Violation labels and timestamps.
- Output is written outside `static/` and streamed only through `/api/videos/jobs/<id>/output`.
- Codec preference `avc1` (H.264) with `mp4v` fallback. The OpenH264 DLL warning is benign on this host: the `avc1` file contains an `avcC` box and decodes correctly.

### Storage

Additive tables only:

- `video_jobs`: id, timestamps, source type/id, original filename, stored path, status/progress/message/error, media metadata, processed/sampled/total frame counts, output path, aggregate counts, settings/summary/result JSON.
- `video_events`: job id, worker id, category, label, status, confidence, first/last seconds, first/last frame, observations, evidence, created timestamp.

Uploads live in `data/video_uploads/<24-hex-id>/`, outputs in `data/video_outputs/<32-hex-job-id>.mp4`. No arbitrary filesystem path is returned by an API or served by a static route.

---

## 4. APIs

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/videos` | Video workspace page |
| GET | `/videos/jobs/<job_id>` | Job status or result dashboard |
| GET | `/api/videos/samples` | Discovered samples with metadata and safe URLs |
| GET | `/api/videos/samples/<video_id>/metadata` | One sample's OpenCV metadata |
| GET | `/api/videos/samples/<video_id>/thumbnail` | Cached thumbnail or placeholder |
| GET | `/api/videos/samples/<video_id>/stream` | Range-capable sample playback |
| POST | `/api/videos/upload` | Validate and store an uploaded video |
| GET | `/api/videos/uploads/<upload_id>/stream` | Range-capable uploaded preview |
| POST | `/api/videos/analyze` | Queue a sample or upload for analysis |
| GET | `/api/videos/jobs/<job_id>` | Status/progress (no full result payload) |
| GET | `/api/videos/jobs/<job_id>/result` | Persisted result, workers, summary |
| GET | `/api/videos/jobs/<job_id>/events` | Persisted timestamped events |
| GET | `/api/videos/jobs/<job_id>/output` | Annotated MP4 playback/download |
| GET | `/api/videos/jobs/<job_id>/source` | Original sample/upload playback |

IDs are validated by regex. Unknown, malformed, or traversal-like IDs return 404. Upload rejection returns 400 with a clear message; oversized requests return 413.

---

## 5. Discovered and tested videos

All 13 files currently in `test_videos/` were discovered dynamically. Categories come from the existing sampled-frame detection report, not from filenames.

| # | File | Resolution | FPS | Duration | Category |
|---|---|---|---|---|---|
| 1 | `13569683-hd_1920_1080_60fps.mp4` | 1920×1080 | 59.94 | 15.52 s | PPE observed |
| 2 | `13753913_3840_2160_50fps.mp4` | 3840×2160 | 50.00 | 11.32 s | PPE observed |
| 3 | `14626386_2160_3840_30fps.mp4` | 2160×3840 | 29.97 | 19.82 s | PPE observed |
| 4 | `16533594_1920_1080_25fps.mp4` | 1920×1080 | 25.00 | 14.92 s | PPE observed |
| 5 | `19832490-hd_1920_1080_25fps.mp4` | 1920×1080 | 25.00 | 5.76 s | Missing PPE observed |
| 6 | `5434218-hd_1920_1080_24fps.mp4` | 1920×1080 | 23.98 | 6.96 s | PPE observed |
| 7 | `5434223-hd_1920_1080_24fps.mp4` | 1920×1080 | 23.98 | 24.07 s | PPE observed |
| 8 | `8486681-uhd_3840_2160_30fps.mp4` | 2560×1440 | 30.00 | 9.63 s | PPE observed |
| 9 | `8964930-uhd_3840_2160_25fps.mp4` | 3840×2160 | 25.00 | 8.04 s | PPE observed |
| 10 | `8965526-uhd_2160_3840_25fps.mp4` | 2160×3840 | 25.00 | 12.60 s | PPE observed |
| 11 | `Recording 2026-11-18 093655.mp4` | 1918×998 | 30.00 | 16.83 s | Missing PPE observed |
| 12 | `Recording 2026-11-18 093828.mp4` | 1918×946 | 30.00 | 9.00 s | Missing PPE observed |
| 13 | `Recording 2026-11-18 093935.mp4` | 1906×978 | 30.00 | 14.57 s | Missing PPE observed |

Vertical clips correctly produce portrait annotated output (`720×1280`), confirming aspect preservation.

---

## 6. Real validation results

**Command used for validation:** a one-off harness invoking the same `process_video()` used by the worker.
**Settings:** 1 fps, up to 8 sampled frames, 640 px inference, confidence 0.25, 2 minimum observations, 0.5 support ratio, 5 s cooldown, 1280 px max output width.
**Artifacts:** annotated files were written to a temporary directory and deleted after metrics were captured; only `reports/video_validation_results.json` is retained.

| # | Video | Processed | Annotated output | Workers | Compliant | Violations | Unconfirmed | Events | Readable |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `13569683-hd_1920_1080_60fps.mp4` | 8 | 1280×720 @ 0.512 fps | 7 | 0 | 0 | 7 | 0 | yes |
| 2 | `13753913_3840_2160_50fps.mp4` | 8 | 1280×720 @ 0.704 fps | 5 | 0 | 0 | 5 | 0 | yes |
| 3 | `14626386_2160_3840_30fps.mp4` | 8 | 720×1280 @ 0.400 fps | 9 | 0 | 0 | 9 | 0 | yes |
| 4 | `16533594_1920_1080_25fps.mp4` | 8 | 1280×720 @ 0.532 fps | 2 | **2** | 0 | 0 | 0 | yes |
| 5 | `19832490-hd_1920_1080_25fps.mp4` | 6 | 1280×720 @ 1.000 fps | 7 | 0 | **1** | 6 | **1** | yes |
| 6 | `5434218-hd_1920_1080_24fps.mp4` | 7 | 1280×720 @ 0.999 fps | 2 | 0 | 0 | 2 | 0 | yes |
| 7 | `5434223-hd_1920_1080_24fps.mp4` | 8 | 1280×720 @ 0.328 fps | 2 | 0 | 0 | 2 | 0 | yes |
| 8 | `8486681-uhd_3840_2160_30fps.mp4` | 8 | 1280×720 @ 0.811 fps | 1 | **1** | 0 | 0 | 0 | yes |
| 9 | `8964930-uhd_3840_2160_25fps.mp4` | 8 | 1280×720 @ 0.962 fps | 3 | 0 | 0 | 3 | 0 | yes |
| 10 | `8965526-uhd_2160_3840_25fps.mp4` | 8 | 720×1280 @ 0.625 fps | 2 | **2** | 0 | 0 | 0 | yes |
| 11 | `Recording 2026-11-18 093655.mp4` | 8 | 1280×666 @ 0.469 fps | 5 | 0 | **3** | 2 | **3** | yes |
| 12 | `Recording 2026-11-18 093828.mp4` | 8 | 1280×630 @ 0.882 fps | 10 | 0 | **4** | 6 | **4** | yes |
| 13 | `Recording 2026-11-18 093935.mp4` | 8 | 1280×656 @ 0.545 fps | 3 | 0 | **1** | 2 | **1** | yes |

Total: **13/13 processed, 13/13 outputs readable**, 0 pipeline failures.

### Timestamped events produced

| Video | Worker | Category | First–last | Observations | Confidence |
|---|---|---|---|---|---|
| `19832490-hd_1920_1080_25fps.mp4` | #7 | Safety vest | 4.00–5.00 s | 2 | 0.4235 |
| `Recording 2026-11-18 093655.mp4` | #1 | Hard hat | 0.00–14.93 s | 8 | 0.6700 |
| `Recording 2026-11-18 093655.mp4` | #3 | Hard hat | 0.00–12.80 s | 2 | 0.5748 |
| `Recording 2026-11-18 093655.mp4` | #5 | Hard hat | 4.27–6.40 s | 2 | 0.6103 |
| `Recording 2026-11-18 093828.mp4` | #8 | Hard hat | 2.27–4.53 s | 2 | 0.5019 |
| `Recording 2026-11-18 093828.mp4` | #5 | Hard hat | 3.40–7.93 s | 4 | 0.6800 |
| `Recording 2026-11-18 093828.mp4` | #2 | Hard hat | 4.53–6.80 s | 3 | 0.3932 |
| `Recording 2026-11-18 093828.mp4` | #1 | Hard hat | 5.67–6.80 s | 2 | 0.5850 |
| `Recording 2026-11-18 093935.mp4` | #3 | Hard hat | 7.33–9.17 s | 2 | 0.7045 |

The three newly recorded negative-PPE clips therefore produced real, timestamped events, while the representative compliant clips produced **no** negative events. This directionally matches the expected labels.

### Live UI verification

The running app was exercised through the browser, not just the API:

- `/videos` rendered 13 sample cards with thumbnails, metadata, and categories, with no console errors.
- Selecting a sample populated the preview, resolution, duration, and FPS, and enabled analysis.
- Starting analysis showed live progress (`Processing · 2 of 6 sampled frames · 33%`), then navigated to the result page when complete.
- `19832490-hd_1920_1080_25fps.mp4` (6 sampled frames): 7 workers, 0 violations, 0 events; both annotated (1280×720) and original (1920×1080) players reached `readyState = 4` and played.
- `Recording 2026-11-18 093655.mp4` (17 sampled frames via the UI): 5 workers, 3 potential violations, 2 unconfirmed, 3 events. The event table rendered, and clicking a timestamp seeked the annotated player to that moment and started playback.
- Verdict wording, KPI cards, processing metadata, worker table, notes, and the “Video Library” return link all rendered correctly.
- Two issues were found during this review and fixed:
  - a merged event kept the evidence text of one sub-run (`3 sampled frames`) while the Frames column showed the merged total (`29`); the temporal engine now rebuilds the text from the merged totals, and a regression assertion covers it (`8 sampled frame(s) across 2 verified runs`);
  - the result page’s live polling block was nested inside the workspace-only branch, so a job opened while still running showed `0%` indefinitely; it now runs on the result page and auto-reloads when the job finishes (verified end-to-end).
- The full 113-test suite and the 13-video validation were re-run after these fixes.

---

## 7. Model observations

- Hard hat and safety vest are the reliable signals; all verified events in this validation involved hard hats or a vest, which matches the checkpoint's stronger classes.
- Gloves, goggles, and masks stayed Unconfirmed for most workers because the checkpoint rarely emitted those boxes at the sampled frames. This is reported as Unconfirmed, not as compliance or violation.
- `19832490-hd_1920_1080_25fps.mp4` produced a vest event at confidence 0.42 with only 2 observations. It is a plausible real negative observation, but it is near the threshold and should be manually reviewed before any action.
- In busy or fast-moving footage, ByteTrack fragmentation is visible: `Recording 2026-11-18 093828.mp4` produced 10 workers in 9 seconds, and `19832490` produced 7 in 5.8 seconds. Some fragments are the same physical person.
- Some tracked workers are marked Present for a helmet but Unconfirmed for the vest, so they remain Unconfirmed overall. The three-state model is intentionally conservative.
- Sampling changes event counts: the same clip at 34 sampled frames produced 4 events, while the 8-frame validation produced 3. Short violations can fall between samples.

---

## 8. Limitations

1. **Sampling is not exhaustive.** At the default 2 fps and 180-frame cap, short violations between samples can be missed. The frame budget protects CPU time rather than guaranteeing full coverage.
2. **Tracker identity is not globally stable.** ByteTrack IDs are stable only within a job; fragmentation on crowded footage can split one person into several workers. IDs are not biometric identities.
3. **Event confidence is detector confidence, not calibrated probability.** Low-confidence events (roughly 0.39–0.42 here) require manual review.
4. **Only explicit `NO-*` detections create events.** A missing positive box does not. This avoids false alarms but means a missed negative detection yields Unconfirmed rather than a violation.
5. **Compliant is narrow.** It requires sufficient present hard-hat and vest evidence and no verified event. Gloves/goggles/mask do not currently upgrade or downgrade the overall status unless they produce verified evidence.
6. **One job at a time.** The queue is deliberate for CPU inference; a second submission waits until the first finishes. There is no cancellation or priority control.
7. **Storage grows with uploads and outputs.** There is no automatic retention policy. Delete upload/output directories manually if disk use matters.
8. **Codec support is browser-dependent.** `avc1` output plays in current Chrome/Edge/Firefox. If the environment falls back to `mp4v`, some browsers may not play the annotated file even though OpenCV can read it.
9. **The OpenH264 warning is benign here.** OpenCV logs `Failed to load OpenH264 library`, then writes valid H.264 via its FFmpeg path; the produced files contain `avcC` and were read back and played successfully.
10. **Validation used 8 sampled frames per video.** The per-video event counts in this report are therefore a lower-bound-style sample, not a full-length audit.
11. **Not a compliance certificate.** Every result is decision support and requires review against the original footage and local rules.

---

## 9. How to run and demo

### Start the app

```powershell
cd C:\Users\ayush\Downloads\SiteGuardVision
.\.venv\Scripts\python.exe server.py
```

Open:

- Dashboard: <http://127.0.0.1:5000>
- Video workspace: <http://127.0.0.1:5000/videos>

The server is a development server; use a production WSGI server for deployment.

### Demo a bundled sample

1. Open <http://127.0.0.1:5000/videos>.
2. In **Sample videos**, click `Recording 2026-11-18 093655.mp4` (marked **Missing PPE observed**). The preview, resolution, duration, and FPS populate.
3. Adjust **Sampling rate**, **Frame budget**, **Inference resolution**, **PPE confidence**, **Verification observations**, and **Evidence proportion** if desired.
4. Click **Start video analysis**. The progress panel reports sampled-frame progress; the page opens the result automatically.
5. On the result page, review the verdict, KPI cards, processing metadata, **Timestamped Safety Events**, and per-worker evidence.
6. Click a timestamp in the event table to seek the annotated player to that moment.
7. Compare the **Worker & PPE Overlay** with **Original Video**.
8. Click **Download Annotated** to save the MP4, or **Video Library** to return.

### Demo an upload

1. On `/videos`, click **Upload Video** (or drop a file on the upload panel).
2. Choose an MP4, AVI, MOV, MKV, or WEBM up to 256 MB.
3. Click **Upload and validate**. Unsupported, oversized, or corrupt files are rejected with a clear message and are not kept.
4. After validation, the uploaded copy is selected automatically.
5. Click **Start video analysis** and follow the same result flow.
6. Playback of the uploaded source is available only through its validated upload ID route.

### Run the tests

```powershell
cd C:\Users\ayush\Downloads\SiteGuardVision
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Expected: **113 tests, OK**. Logs: `reports/video_full_test_run.log`.

### Re-run the 13-video validation

The one-off harness is at `C:\Users\ayush\AppData\Local\Temp\opencode\validate_videos.py` (temporary). It writes `reports/video_validation_results.json` and logs to `reports/video_validation_run.log`. It does not use the production database and deletes annotated temp files after measuring them.

### Environment overrides

Useful for isolated testing or relocating data:

```powershell
$env:SITEGUARD_DB_PATH = "C:\temp\siteguard-test.db"
$env:SITEGUARD_SAMPLE_VIDEO_DIR = "C:\temp\videos"
$env:SITEGUARD_VIDEO_UPLOAD_DIR = "C:\temp\uploads"
$env:SITEGUARD_VIDEO_OUTPUT_DIR = "C:\temp\outputs"
$env:SITEGUARD_VIDEO_THUMB_DIR = "C:\temp\thumbs"
$env:SITEGUARD_VIDEO_REPORT = "C:\temp\video_detection_report.json"
```

Remove overrides with `Remove-Item Env:SITEGUARD_DB_PATH` (and the same pattern for the others) before returning to the production database.

---

## 10. Configuration reference

| Setting | Default | Range used by the UI/API |
|---|---|---|
| Sampled frames per second | 2.0 | 0.25–10.0 |
| Max sampled frames | 180 | 6–600 |
| Inference size | 640 px | 640 / 960 / 1280 |
| PPE confidence | 0.25 | existing image minimum–maximum |
| Max output width | 1280 px | 320–3840 (server default 1280) |
| Minimum negative observations | 3 | 2–30 |
| Evidence proportion | 0.5 | 0.2–1.0 |
| Event cooldown | 8.0 s | 0.0–120.0 |

Global request limit is 256 MB for video; image and document routes enforce 15 MB explicitly so their behavior is unchanged.
