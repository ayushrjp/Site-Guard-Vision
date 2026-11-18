# Application inference configuration audit

**Audit date:** 2026-11-16  
**Scope:** read-only inspection of `infer_engine.py`, `server.py`, `storage.py`,
`evaluation/evaluate_application.py`, templates, and existing tests. No
application or production configuration was changed.

## Actual active defaults

| Setting | Active value | Source |
|---|---:|---|
| PPE model | `models/yolo_ppe_core.pt` | `infer_engine.PPE_MODEL_PATH` |
| Dedicated person model | `models/person_yolov8n.pt` (COCO class 0) | `infer_engine.PERSON_MODEL_PATH` |
| PPE confidence | **0.25** | `infer_engine.DEFAULT_CONF`; `storage.DEFAULT_SETTINGS` |
| Person confidence | **0.25 minimum** | `max(requested_conf, PERSON_CONF_FLOOR)` |
| Production image size | **960px** | `infer_engine.DEFAULT_IMGSZ`; SQLite has no saved setting override |
| Allowed image sizes | 640, 960, 1280 | `infer_engine.ALLOWED_IMGSZ`, `storage.ALLOWED_IMGSZ` |
| Inference device | CPU | both `YOLO.predict` calls |
| NMS IoU | 0.70 | Ultralytics 8.3.40 default; `infer_engine` does not override it |
| Maximum detections | 300/model | `infer_engine.analyze_image` |
| Checkpoint hashes | PPE `2b33d4…a4f61`; person `f59b3d…83b36` | match `reports/ppe_evaluation/baseline_configuration.json` |

The current SQLite settings table has no overrides; `storage.DEFAULT_SETTINGS`
therefore supplies confidence 0.25 and image size 960. The existing SQLite
database contains five stored analyses. The diagnostic comparison does not
write to this database. The UI permits changing settings, but no user setting
was saved or changed during this task.

## Inference and filtering path

1. `server.analyze()` reads confidence/image-size form overrides; absent
   overrides, it uses saved settings. Invalid image sizes fall back to 960.
2. `infer_engine.analyze_image()` clamps PPE confidence to 0.05–0.95 and size
   to the allowed set.
3. The PPE checkpoint runs with the requested confidence and size on CPU.
   Ultralytics confidence filtering and its default NMS IoU 0.70 are applied;
   the engine limits output to 300 boxes. The PPE checkpoint's own Person class
   is discarded when the dedicated person detector is available.
4. The dedicated YOLOv8n person detector runs with class `[0]`, confidence
   `max(requested, 0.25)`, the same input size, CPU, default NMS IoU 0.70, and
   maximum 300 boxes.
5. Both weights and the filtering/association/assessment logic were held
   constant in the 640/960 comparison. Since production passes one `imgsz` to
   **both** model calls, changing application image size also changes person
   detector inference resolution. This is actual app behavior; it was not
   modified independently.

## Worker-to-PPE association

`associate_detections()` assigns PPE to at most one detected person. A helmet
or no-helmet box is eligible inside that worker's head region; vest/no-vest is
eligible inside the torso region. A candidate must have at least **0.30
intersection-over-detection** and its center must be inside the region.

- Head region: top of person box through 38% of height; 5% side inset.
- Torso region: 16%–82% of height; 3% side inset.
- Candidates are greedily assigned by overlap score; each worker receives at
  most one detection per helmet/vest slot.
- Same-category PPE boxes with at least 0.85 containment are deduplicated,
  retaining the higher-confidence detection. Opposite present/missing
  categories are not merged.

These constants and rules were not changed.

## PPE states, assessment, and downstream events

`build_person_records()` maps a matched hardhat/vest class to `present`, its
explicit no-PPE class to `missing`, and no assigned class evidence to `unknown`.
Worker status is:

- `potential_violation` if either assigned PPE slot is explicitly missing;
- `clear` only when both slots are present;
- `uncertain` otherwise.

The image-level `assess()` returns `potential_violation` for any explicit
no-hardhat/no-vest detection, even if that detection is unassigned. It returns
`no_violation_detected` only when people are found, all workers have positive
evidence for both PPE slots, and no missing-PPE detection passed the filter.
Otherwise it returns `insufficient_evidence`. The UI renders `unknown` as
**Unconfirmed** and includes the caveat that a clear result is not proof of
compliance.

No separate notification, dispatch, or safety-event service was found.
Downstream event-like behavior is the `potential_violation` assessment status,
the rendered review banner/flags, and persisted history/dashboard counters.
`storage.stats()` counts potential-violation images and missing-PPE detections.
The comparison therefore measures an image-level annotated-violation event,
not a real-world incident or site-compliance event.

## Pipeline coverage used for comparison

The application comparison calls the real `infer_engine.analyze_image()`
path, which includes both models, class filtering, association,
three-state worker assessment, image-level assessment, and annotated-image
rendering. The harness removes only its own generated `static/rendered/det_*.jpg`
previews and does not call the production persistence route. Flask/database,
API, and dashboard behavior are covered separately by the existing regression
suite.

The dataset test annotations are used as a review reference. For worker-level
comparison, ground-truth PPE boxes are associated to ground-truth person boxes
using the same production association geometry; predictions are matched to
annotated people at IoU 0.50. Unannotated worker PPE slots are not assumed to
be compliant. Two test images were previously flagged for possible annotation
disagreement; they remain unmodified and their outcomes are a limitation.
