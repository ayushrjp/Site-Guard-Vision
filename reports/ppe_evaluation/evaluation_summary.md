# PPE Model Evaluation — Final Technical Summary

**Run date:** 2026-11-15  
**Evaluation split:** `dataset/test`  
**Production baseline:** preserved; PPE confidence cutoff remains **0.25**. No checkpoint, dataset, person detector, association geometry, or production assessment code was changed.

## 1. Executive summary

The earlier 90-image run completed inference but did not save its metrics because the report assembly raised a `KeyError`: it looked up `mAP50_95` in an AP dictionary whose key was `ap50_95`. The lookup was corrected. A five-image smoke evaluation then passed, followed by a complete 90-image CPU evaluation at 960px. All 90 images and 623 labeled objects were evaluated.

At confidence 0.25 and class-aware greedy IoU matching at 0.50, the four PPE classes (excluding the PPE checkpoint’s separate `Person` class) achieved macro precision **0.797**, recall **0.565**, mAP@50 **0.747**, and mAP@50–95 **0.410**. Safety Vest precision was **0.900**, but recall was **0.558**. No threshold was selected or deployed.

The previously investigated P1 case was reproduced at production-aligned inference size 960: its Safety Vest box had confidence **0.143993**, IoU **0.8873** with the labeled vest, and was filtered before production association. The existing association geometry would accept it if it passed the confidence filter. This remains an unconfirmed production result; the evaluation does not override it.

## 2. Existing system architecture

- PPE checkpoint: `models/yolo_ppe_core.pt`, 14 model classes.
- Application person detector: `models/person_yolov8n.pt`, class 0 (`person`), used separately from the PPE model.
- Production PPE cutoff: `infer_engine.DEFAULT_CONF = 0.25`.
- Production person confidence floor: `0.25`.
- Association: head/torso region IoA, minimum score `0.30`, and center-inside requirement; a PPE detection is assigned to at most one worker slot.
- Worker IDs are per-image detection order (`P1`, `P2`, …); no temporal/video tracking implementation was found in the image inference pipeline.
- No restricted-zone monitoring implementation was found in the inspected pipeline.
- Assessment uses worker evidence and preserves Present / Missing / Unconfirmed and the three final assessment states.

This report’s detector metrics are **PPE-checkpoint model-level metrics**, not worker-level assessment metrics and not a compliance verdict.

## 3. Baseline configuration

The full baseline, dependency versions, dataset/model paths, class mapping, and SHA-256 hashes are in [`baseline_configuration.json`](baseline_configuration.json). Key values:

| Setting | Value |
|---|---|
| Production PPE confidence | 0.25 |
| Evaluation candidate floor | 0.001 (diagnostic only) |
| Evaluation image size | 960 |
| NMS IoU | 0.70 |
| Matching IoU | 0.50 for precision/recall; 0.50–0.95 at 0.05 steps for AP |
| Device | CPU; CUDA unavailable |
| Ultralytics / PyTorch | 8.3.40 / 2.15.0+cpu |

The diagnostic floor preserves low-confidence candidates for threshold analysis; it does not change production inference. Evaluation used the same image size as the current application default.

## 4. Dataset description and integrity

`dataset/data.yaml` declares five classes and YOLO normalized `xywh` labels:

| Dataset ID | Class | Checkpoint ID |
|---:|---|---:|
| 0 | helmet | 3 (`Hardhat`) |
| 1 | no-helmet | 8 (`NO-Hardhat`) |
| 2 | no-vest | 10 (`NO-Safety Vest`) |
| 3 | person | 11 (`Person`) |
| 4 | vest | 13 (`Safety Vest`) |

| Split | Images | Label files | Objects |
|---|---:|---:|---:|
| Train | 997 | 997 | 6,386 |
| Validation | 119 | 119 | 715 |
| Test | 90 | 90 | 623 |

No missing/orphan labels, malformed or out-of-range boxes, or corrupt images were found in the split audit. Per-class object counts and audit details are in [`dataset_audit.json`](dataset_audit.json).

## 5. Model evaluation results

The evaluator maps the five dataset class IDs to the checkpoint IDs above, runs CPU inference at `imgsz=960`, NMS IoU 0.70, and retains candidates from confidence 0.001. Matching is confidence-ordered, class-aware, greedy, and one-to-one within each image. AP uses 101-point interpolation at IoUs 0.50 through 0.95. These are custom metrics, not an official Ultralytics `val` result.

At report cutoff 0.25 and IoU 0.50:

| Class | GT | TP | FP | FN | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Helmet | 195 | 151 | 64 | 44 | 0.702 | 0.774 | 0.712 | 0.372 |
| No Helmet | 24 | 14 | 4 | 10 | 0.778 | 0.583 | 0.764 | 0.383 |
| No Vest | 61 | 21 | 5 | 40 | 0.808 | 0.344 | 0.657 | 0.342 |
| Safety Vest | 129 | 72 | 8 | 57 | 0.900 | 0.558 | 0.857 | 0.541 |
| PPE macro, excluding Person | — | — | — | — | **0.797** | **0.565** | **0.747** | **0.410** |

The PPE model’s own `Person` class produced no predictions at confidence 0.25 (214 labeled persons, 214 false negatives in that class row). This is not the application’s person detector: the application uses the dedicated `person_yolov8n.pt` model. Therefore the PPE-only macro excludes `Person`. The all-five-class macro is P **0.638**, R **0.452**, mAP50 **0.600**, and mAP50–95 **0.330**; it is not an application person-detection score.

The confusion matrix at IoU ≥ 0.50 and confidence ≥ 0.25 is exported in `model_metrics.json` and visualized in [`confusion_matrix.png`](confusion_matrix.png); its final row/column represent background false positives/missed objects.

## 6. Five-image smoke evaluation

The five-image smoke run completed before the 90-image run. It verified checkpoint loading, expected class names/IDs, label parsing, non-empty returned boxes and confidences, and class-aware one-to-one IoU matching. Full image-by-image predictions and matches are in [`sample_diagnostics.json`](sample_diagnostics.json).

| Image | GT objects | Candidate predictions | Matches at IoU ≥ 0.50 | Unmatched GT |
|---|---:|---:|---:|---:|
| `ppe_0000…jpg` | 3 | 5 | 2 | 1 |
| `ppe_0004…jpg` | 9 | 10 | 6 | 3 |
| `ppe_0008…jpg` | 11 | 13 | 6 | 5 |
| `ppe_0009…jpg` | 6 | 13 | 4 | 2 |
| `ppe_0013…jpg` | 3 | 18 | 2 | 1 |

These are model-level matches across all checkpoint classes at diagnostic candidate floor 0.001; unmatched PPE-checkpoint `Person` labels are expected to be handled by the separate application person model.

## 7. Low-confidence detection investigation

The sample and test image copies of `ppe_0004_jpg.rf.99a751a65b56257ea7c44b4de9cae4bb.jpg` have identical SHA-256 hashes.

At 960px, the PPE model returned:

- Safety Vest, checkpoint class 13 / dataset class 4
- Confidence **0.143993**, below the unchanged production cutoff 0.25
- Box `[298.855, 223.158, 385.406, 322.930]`
- P1 torso region `[299.511, 215.120, 453.389, 396.290]`
- IoA with P1 torso **0.9925**; center inside torso
- Same-class GT vest IoU **0.8873**

It is rejected by confidence filtering before production association. The association algorithm’s spatial gates (IoA ≥ 0.30 and center inside) would pass this candidate for P1; there is no evidence of a torso-geometry or wrong-worker assignment defect. The stored production response therefore correctly left P1 Unconfirmed and the image at Insufficient Evidence given its accepted detections.

The 640px diagnostic smoke invocation returned a different confidence (0.43392) for the same central vest. This demonstrates image-size sensitivity; the formal full evaluation used 960px to match the application default. The 640px result is not substituted for the stored 960px production result.

[`low_confidence_cases.csv`](low_confidence_cases.csv) contains **784** below-0.25 checkpoint prediction rows (779 PPE-class candidates plus 5 predictions from the PPE checkpoint’s `Person` class) and **365** unmatched-GT diagnostic rows (1,149 CSV rows total). Of those 365 unmatched labels, 214 are `Person` objects and do not measure the application’s dedicated person detector; the PPE-only FN total is 151. These rows are not production detections. Generic model-only rows do not have application worker IDs because the dedicated person detector/association pipeline was not run as part of the PPE checkpoint evaluation. The exact P1 association geometry is documented separately in [`association_analysis.csv`](association_analysis.csv).

## 8. Association analysis

The P1 case is **failure category A/B: candidate confidence below the production threshold; excluded before association**. It is not category C (accepted but unassociated) or F (assigned to another worker). The box overlaps P1 torso strongly, and the center-containment requirement passes. Existing accepted vest boxes in stored analysis #5 were assigned to P2 and P3.

Association was not comprehensively evaluated over all 90 images because the full evaluation intentionally isolated the PPE detector and did not run the application’s separate person detector. No association geometry was changed. The detector-level threshold sweep must not be interpreted as a worker-level association or safety-assessment sweep.

## 9. Offline threshold comparison

PPE macro values below average the four PPE classes and exclude the PPE checkpoint `Person` class. FP/FN are summed across those four classes. AP values are invariant across these operating thresholds because the same 0.001 candidate set and ranked detections are used to construct the PR/AP curves; the threshold controls the reported operating point.

| Confidence | Macro precision | Macro recall | FP | FN |
|---:|---:|---:|---:|---:|
| 0.15 | 0.764 | 0.723 | 122 | 86 |
| 0.20 | 0.777 | 0.643 | 102 | 119 |
| **0.25 production** | **0.797** | **0.565** | **81** | **151** |
| 0.30 | 0.827 | 0.503 | 63 | 188 |
| 0.35 | 0.831 | 0.422 | 51 | 230 |
| 0.40 | 0.810 | 0.280 | 41 | 292 |
| 0.50 | 0.662 | 0.159 | 27 | 345 |

Lower cutoffs raise recall but also false positives; higher cutoffs reduce false positives while increasing missed labeled objects. No alternative is recommended or deployed from this single split alone. Any threshold policy requires representative validation and explicit safety trade-off approval.

## 10. Failure analysis

At confidence 0.25 / IoU 0.50, notable detector-level issues are:

- Safety Vest: 57 false negatives (recall 0.558), despite high precision 0.900.
- No Vest: 40 false negatives (recall 0.344), precision 0.808.
- Helmet: 64 false positives and 44 false negatives.
- No Helmet: 10 false negatives and 4 false positives.
- P1 in `ppe_0004…jpg`: a labeled vest candidate exists at confidence 0.143993; it is threshold-filtered, but geometrically compatible with P1.

Detailed per-image candidate/GT correspondences, low-score boxes, and failure labels are in the CSV. Metrics indicate annotation agreement under the specified matching method; they do not prove actual site compliance or that every absent box is a true absence.

## 11. Regression test results

No production code or model was changed, so no post-fix regression run was necessary. The five-image diagnostic and full 90-image evaluation are model-evaluation runs, not application regression tests. The earlier project suite result of 56 passing tests belongs to the prior system audit and is not claimed as a new run here.

## 12. Limitations

- CPU-only evaluation; CUDA was unavailable. The pre-run OS snapshot showed 1,338 MB free of 7,877 MB visible memory. Peak inference memory was not measured.
- Custom greedy IoU matcher and AP implementation; results can differ from official Ultralytics validation.
- Only the test split was used for final reported metrics. The threshold sweep is diagnostic and must not be used for threshold selection without validation.
- Dataset labels and test distribution may contain annotation omissions or difficult/occluded examples; disagreement is not automatically model error or annotation error.
- PPE checkpoint `Person` metrics do not measure the application’s dedicated person detector.
- Detector-only sweep does not quantify worker-to-PPE association accuracy.

## 13. Recommendations

1. Keep the production confidence cutoff at 0.25 unless a separately approved evaluation supports a change.
2. Investigate vest/no-vest recall on a reviewed, stratified validation set, especially small, partially occluded, and distant PPE.
3. If evaluating a new cutoff, validate detector precision/recall and worker-level false-association behavior together before deployment.
4. Keep model-only diagnostics clearly distinct from production worker evidence and assessment.

## 14. Final technical conclusion

The prior failure was an evaluation report key mismatch, not a model-loading, dependency, dataset, or inference failure. The corrected run passed its five-image smoke stage and evaluated all 90 test images. The baseline remains intact. At production-aligned 960px and confidence 0.25, the checkpoint has strong Safety Vest precision but materially lower recall; the specific P1 miss is due to a low-confidence candidate filtered before association. No change to production settings is justified by this diagnostic alone.

## Artifacts

- Machine-readable metrics: [`model_metrics.json`](model_metrics.json)
- Class-wise metrics: [`class_wise_metrics.csv`](class_wise_metrics.csv)
- Threshold table: [`threshold_comparison.csv`](threshold_comparison.csv)
- Low-confidence/FN cases: [`low_confidence_cases.csv`](low_confidence_cases.csv)
- Baseline and dataset audit: [`baseline_configuration.json`](baseline_configuration.json), [`dataset_audit.json`](dataset_audit.json)
- Five-image diagnostics: [`sample_diagnostics.json`](sample_diagnostics.json)
- Association details: [`association_analysis.csv`](association_analysis.csv)
- Graphs: [`precision_recall.png`](precision_recall.png), [`threshold_comparison.png`](threshold_comparison.png), [`confusion_matrix.png`](confusion_matrix.png), [`confidence_distribution.png`](confidence_distribution.png)
- Immutable run-folder copy of all generated evaluation outputs: [`run_20261115/`](run_20261115/)
