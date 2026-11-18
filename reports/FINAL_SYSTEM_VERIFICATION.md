# SiteGuardVision — Final System Verification

**Audit date:** 2026-11-15  
**Scope:** Read-only verification of the existing application, with tests and image analysis isolated from production persistence.  
**Overall verdict: READY WITH LIMITATIONS** for a supervised final-year project demonstration of image upload, real YOLO inference, worker-level evidence, backend assessment, dashboard results, and persisted analysis history. The live RAG assistant is not demonstration-ready unless Ollama and an indexed safety-document collection are made available and verified. This is decision-support software; predictions are not ground truth or a compliance certificate.

## Executive summary

- The existing automated suite completed with **56 passed, 0 failed, 0 errors, 0 skipped**. Tests ran with temporary database, vector-store, upload, and rendered-image paths.
- Five existing images were submitted through Flask’s test client and processed by the real inference pipeline. Worker/PPE counts, assessment status and title, worker states, and annotated-image availability/validity were checked against the captured backend result. All recorded consistency checks passed.
- The crowded `ppe_0675_jpg.rf.eff80639203aef025f9fee2bc42c3ac5.jpg` image was included. It produced 9 detected people, 11 hard hats, 1 vest, and `insufficient_evidence`.
- Production SQLite integrity is `ok`; it has 4 analysis records, all `insufficient_evidence`, and every referenced annotated image exists.
- All 2,424 files in the available SHA-256 audit baseline matched; no baseline files were missing and no new dataset files were found.
- Ollama at `localhost:11434` refused the connection. The vector manifest lists zero documents. Live model availability, embeddings, ingestion, retrieval, and generated answers therefore remain unverified.
- No Flask server was listening on port 5000 during the audit. Flask HTTP behavior was exercised with its test client, not a network server.

## Status by area

| Area | Status | Evidence and limits |
|---|---|---|
| Flask routes and page rendering | **PASS** (test-client scope) | Implemented pages and APIs were requested using Flask’s test client; expected success and error paths were exercised. This does not verify binding/listening or browser behavior over a live server. |
| Upload and validation | **PASS** (test-client scope) | Real sample-image analysis succeeded. Unsupported-extension and unreadable-image cases were checked. |
| Backend APIs, history, CSV, settings | **PASS** (isolated/test-client scope) | API responses and CSV route were checked; invalid settings were rejected and valid settings round-tripped in disposable SQLite. A real analysis was also saved/read back in the temporary database. |
| Static assets | **PASS** (test-client scope) | CSS and JavaScript static responses were requested and checked. Browser rendering, interactive behavior, and accessibility were not exhaustively tested. |
| YOLO checkpoint loading and real inference | **PASS** (sample scope) | Existing CPU inference tests passed, and real image POSTs exercised the configured PPE and person-detection pipeline. These checks do not establish general model accuracy. |
| Worker-to-PPE association | **PASS** (implementation/test scope) | Existing geometry, association, state, and assessment tests passed; image responses included worker-level results. Association remains model- and image-dependent and is not ground truth. |
| Safety assessment | **PASS** (logic/test scope) | Three-state assessment logic and dashboard status rendering were covered by tests; real-image results were compared with backend output. Do not interpret `no_violation_detected` as proof of compliance. |
| Dashboard/backend consistency | **PASS** (server-rendered response scope) | For all five tested images, displayed assessment status/title and headline counts matched captured backend values; worker evidence and annotated output were checked. The frontend JavaScript does not recalculate or override assessment/count fields in the inspected code. No full browser/E2E session was performed. |
| Production database | **PASS** (read-only inspection) | SQLite integrity check `ok`; four analyses, all insufficient-evidence; no settings rows; all referenced annotated images exist. Production database was not written by the audit. |
| Ollama / live RAG | **PARTIAL** | Offline handling and mocked/unit-test behaviors are covered by tests. Live Ollama, configured model availability, embeddings, semantic retrieval, ingestion, citations from a populated index, and generation could not be verified: Ollama refused the connection and manifest has zero indexed documents. |
| Automated tests | **PASS** | 56 passed; 0 failed; 0 errors; 0 skipped. |
| Network server startup | **NOT TESTED** | No server was running; no server was started to avoid creating a duplicate or changing runtime state. |
| Performance and memory | **PARTIAL** | The inference responses expose timing fields and request wall time was measured during image checks, but no stable benchmark or memory profile is recorded in the audit evidence. No performance claim is made. |

## Backend and frontend verification

The audit requested `/`, `/assistant`, `/reports`, `/settings`, `/evaluation`, `/health`, `/api/status`, assistant status/documents APIs, settings API, CSV export, and CSS/JavaScript assets through the Flask test client. Missing analysis/sample paths and invalid assistant/upload inputs were also exercised. Valid temporary settings saved and read back; an invalid image-size setting was rejected. This validates routing and application responses within Flask’s test client only, not live TCP serving.

The analysis template uses the backend `assessment_status` to choose its status treatment and renders backend-provided labels and counts. Insufficient evidence is styled amber, not green. Dashboard JavaScript was searched for assessment/count fields and does not independently calculate those values in the inspected scripts. Five real image results were compared with server-rendered values; each recorded count/status/title consistency check passed.

The report also confirms that an annotated image was generated, opened as a valid image, and served with HTTP 200 in the isolated test setup for each successful image case. This verifies the analyzed output path in that setup, not long-term retention or browser presentation.

## Models, association, and assessment

The application uses separate checkpoints: `models/yolo_ppe_core.pt` for PPE and `models/person_yolov8n.pt` for person detection. Existing tests include real CPU inference when model files are present and passed in this run. The audit did not alter either checkpoint or inference thresholds/settings. Real calls used confidence `0.25` and image size `960`.

Unit tests cover label/geometry/association/assessment behavior, including evidence states and avoiding treating absent detections as proof of compliance. The worker statuses and final assessment returned for the sample images were inspected. These checks establish implementation behavior for tested cases—not detector correctness on arbitrary scenes. Predicted boxes and application assessment must not be represented as ground truth.

### Real-image results

All five cases were submitted through Flask’s test client and the real CPU inference pipeline. In each case, backend response, rendered dashboard values, worker evidence, and annotated-image validity were compared. Counts below are the observed pipeline outputs.

| Image | Workers | Hard hats | NO-Hardhat | Vests | Assessment |
|---|---:|---:|---:|---:|---|
| `ppe_0009` | 3 | 1 | 0 | 1 | Insufficient Evidence |
| `ppe_0014` | 6 | 0 | 3 | 0 | Potential Violation |
| `ppe_0000` | 1 | 1 | 0 | 0 | Insufficient Evidence |
| `ppe_0008` | 9 | 1 | 0 | 4 | Insufficient Evidence |
| `ppe_0675_jpg.rf.eff80639203aef025f9fee2bc42c3ac5.jpg` | 9 | 11 | 0 | 1 | Insufficient Evidence |

For these test submissions, the dashboard’s displayed worker, hard-hat, vest, and missing-PPE counts matched the backend response; assessment status and title matched; and annotated images were valid and served successfully. These are model predictions, not a manual ground-truth adjudication.

The crowded image’s dataset label file contains 21 labeled objects, including no-hardhat/no-vest classes. The application output differs from those annotations (for example, it predicts no NO-Hardhat detections). This is a model/annotation disagreement, not proof that either is correct. The saved custom 90-image application-level evaluation likewise reports 21 true positives, 15 false negatives, 0 false positives, precision 1.0, recall 0.5833, F1 0.7368, and accuracy 0.8333 for its defined annotated-violation matching. These are custom application-level agreement metrics, not official detector validation metrics and not a substitute for reviewing this image.

## Persistence and RAG

Production SQLite was opened read-only. `PRAGMA integrity_check` returned `ok`; four analysis rows were present, each with status `insufficient_evidence`; no settings rows were present; and all referenced annotated files existed. A separate disposable database was used to verify a settings round-trip and one analysis save/read-back. No restart cycle was performed, so persistence across a real Flask process restart was not directly demonstrated.

The application’s configured defaults are `qwen2.5:3b` for chat and `nomic-embed-text` for embeddings. A direct Ollama tags request to `http://localhost:11434/api/tags` failed with connection refused. `vector_db/manifest.json` reports an empty document map. Therefore:

- **Verified offline:** defensive/offline behavior covered by automated tests; the application’s image-analysis path remains independent of Ollama.
- **Mocked/unit-test only:** assistant flows that use fake Ollama/session responses. These tests are not evidence of live generation.
- **Not verified live:** chat/embedding model presence, document upload/indexing against Ollama, populated Chroma retrieval, semantic relevance, context passed to a live model, generated answers, and live citations.

## Tests, performance, and reliability

The complete existing suite ran successfully: **56 tests, 56 passed, 0 failed, 0 errors, 0 skipped**. This includes route/validation, persistence/RAG helper tests, and real CPU inference tests. Test writes and rendered files were redirected to temporary locations.

Per-image inference and request timing fields were available during integration checks, but the exact timing samples and a repeatable benchmark summary are not retained in this report. No memory measurement was collected. Consequently performance and memory suitability for a live demonstration remain **PARTIAL / NOT BENCHMARKED**; no values are invented here.

## Issues, recommended actions, and limitations

1. **Live assistant dependency unavailable (demonstration limitation):** start/configure Ollama, ensure the configured chat and embedding models are installed, add safety documents, index them, then verify a live question, retrieved sources, and citations before demonstrating RAG.
2. **No network-server verification:** before the demonstration, start the application through its normal documented process and smoke-test the pages, upload, static assets, and APIs in a browser. No server was launched during this audit.
3. **Detection/annotation disagreement:** the crowded test image’s model predictions differ from its labels. Present model outputs as review aids, and explain the limits of the custom evaluation; do not claim perfect or generalizable model accuracy.
4. **Performance evidence incomplete:** collect repeatable latency and resource measurements on the intended demonstration machine if performance is part of the project claims.
5. **Assessment is deliberately cautious:** several images produce insufficient evidence. Do not manually recast this as compliant/green; have a reviewer inspect the source image and uncertain detections.

No confirmed dashboard/backend consistency defect was found in the tested cases. This is not an exhaustive visual, browser, concurrency, security, or generalization audit.

## Change-safety record

The audit used temporary paths for test persistence, uploads, vector data, and rendered images. It did not train models, modify checkpoints, alter dataset annotations or thresholds, change the production inference pipeline, or write production analysis history. All **2,424** files in the available SHA-256 baseline matched, with no missing baseline files, hash mismatches, or new dataset files. No annotation-audit process was running when the audit checks were made, and no Flask server was listening on port 5000; no server was started.

## Demonstration verdict

**READY WITH LIMITATIONS.** The core image-analysis demonstration has strong verification evidence: real checkpoint inference was exercised through the Flask request path, backend worker-level assessments were returned, and the dashboard values checked matched those backend results for five images, including the specified crowded case. The automated suite passed and production history remained untouched.

Keep the demonstration supervised and describe outputs as AI-assisted screening, not ground truth or a compliance certificate. Do not claim the live Safety Assistant/RAG works until Ollama is reachable, documents are indexed, and live retrieval/generation are tested. A live network-server/browser smoke test is also still recommended.
