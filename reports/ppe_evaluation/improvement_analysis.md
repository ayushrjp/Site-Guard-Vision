# PPE Model Error Analysis and Improvement Planning

**Date:** 2026-11-15  
**Baseline:** [`run_20261115/`](run_20261115/)  
**Diagnostic inference:** same PPE checkpoint, CPU, `imgsz=960`, NMS IoU 0.70, candidate floor 0.001; production cutoff remains 0.25.

## 1. Executive summary

The production baseline is unchanged. The separate diagnostic pass re-evaluated the same 90-image test split to retain individual boxes and confidence values; it generated **151 PPE-class false negatives** at confidence 0.25 / IoU 0.50:

- **44 Helmet**
- **10 No Helmet**
- **40 No Vest**
- **57 Safety Vest**

The largest recurring pattern is a below-cutoff, same-class candidate with IoU ≥0.50: **132 of the 151 PPE false negatives**. This includes 33/40 No Vest and 50/57 Safety Vest misses. The candidate is diagnostically useful, but it is not a production detection and does not justify changing the cutoff.

There are also misses with no same-class candidate (5 PPE objects), poor localization (5), a wrong-class prediction (1), and same-class predictions with no spatial overlap (8). The low recall is not explained by tiny vest boxes: test-set No Vest and Safety Vest false negatives were all in the medium/large analysis bins. Dataset class imbalance is strongest for No Helmet, while possible label disagreements were noticed during a small visual spot-check. Those annotations need adjudication before attributing all errors to the model.

## 2. Existing model performance

The baseline 90-image test evaluation at confidence 0.25 reports:

| Class | GT | TP | FP | FN | Precision | Recall | AP50 | AP50–95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Helmet | 195 | 151 | 64 | 44 | 0.702 | 0.774 | 0.712 | 0.372 |
| No Helmet | 24 | 14 | 4 | 10 | 0.778 | 0.583 | 0.764 | 0.383 |
| No Vest | 61 | 21 | 5 | 40 | 0.808 | 0.344 | 0.657 | 0.342 |
| Safety Vest | 129 | 72 | 8 | 57 | 0.900 | 0.558 | 0.857 | 0.541 |

PPE macro (excluding the PPE checkpoint’s `Person` class): P **0.797**, R **0.565**, mAP50 **0.747**, mAP50–95 **0.410**. The production threshold is confirmed as 0.25. The application’s person count comes from a separate person model and is not assessed by the PPE-class error analysis.

## 3. False-negative analysis

Categories use class-aware one-to-one matching at IoU 0.50 and the unchanged 0.25 operating point.

| Missed class | FN | Below 0.25, good IoU | No same-class candidate | Poor IoU | Wrong class | Same-class candidate, no overlap |
|---|---:|---:|---:|---:|---:|---:|
| Helmet | 44 | 41 | 0 | 3 | 0 | 0 |
| No Helmet | 10 | 8 | 1 | 0 | 1 | 0 |
| No Vest | 40 | 33 | 1 | 0 | 0 | 6 |
| Safety Vest | 57 | 50 | 3 | 2 | 0 | 2 |
| **PPE total** | **151** | **132** | **5** | **5** | **1** | **8** |

The No Helmet class error is concrete: in `ppe_0008…jpg`, an accepted Helmet prediction at confidence 0.2678 overlaps a No Helmet annotation at IoU 0.762. It is an incorrect class prediction, not a confidence-filtered No Helmet box.

Below-cutoff prediction counts (all checkpoint predictions with score ≥0.001 and <0.25) were Helmet **431**, No Helmet **82**, No Vest **140**, Safety Vest **126**. These include unmatched/duplicate diagnostic boxes and are not all rescued true positives. `false_negative_analysis.csv` records each unmatched PPE label, its best candidate, confidence, IoU, and failure category. Full raw per-prediction diagnostic records are in [`error_analysis_20261115/all_predictions.csv`](error_analysis_20261115/all_predictions.csv).

## 4. Class-wise confidence analysis

Correct/incorrect accepted detection means are computed from class-aware IoU matches at the production cutoff. “Below-cutoff” counts are diagnostic model outputs below 0.25; they were not admitted to production association or assessment.

| Class | Correct accepted mean confidence | Incorrect accepted mean confidence | Below 0.25 boxes | Below-cutoff boxes with same-class IoU ≥0.50 | Recall |
|---|---:|---:|---:|---:|---:|
| Helmet | 0.471 | 0.471 | 431 | 132 | 0.774 |
| No Helmet | 0.582 | 0.342 | 82 | 23 | 0.583 |
| No Vest | 0.366 | 0.357 | 140 | 40 | 0.344 |
| Safety Vest | 0.388 | 0.346 | 126 | 63 | 0.558 |

No Vest correct and incorrect accepted confidences are close (0.366 vs 0.357), so confidence alone is not a clean discriminator for that class. Safety Vest has a modest separation (0.388 vs 0.346), while many labeled Safety Vest misses still have low-score, well-overlapping candidates. This supports calibration and class-conditional PR investigation on validation data, not an automatic threshold change.

Plots: [`confidence_distribution_detailed.png`](confidence_distribution_detailed.png) and [`confidence_correct_vs_incorrect.png`](confidence_correct_vs_incorrect.png). Data: [`confidence_analysis.csv`](confidence_analysis.csv).

## 5. Dataset quality analysis

The dataset has 997 train, 119 validation, and 90 test images, all 640×640, with 6,386 / 715 / 623 labeled objects respectively. Training counts are Helmet 2,116, No Helmet 94, No Vest 741, Safety Vest 1,073, and Person 2,362. No Helmet is 22.5× less frequent than Helmet; No Vest is 69.1% as frequent as Safety Vest.

The test annotations contain no small No Vest or Safety Vest boxes under the explicitly defined 640-equivalent area bins. Of the false negatives, 24/40 No Vest and 45/57 Safety Vest labels are large; the rest are medium. Therefore, increasing input resolution or small-object augmentation alone is not yet evidenced as the main fix.

Five representative error images were visually checked. They show some difficult/crowded or partially obscured scenes, and two images (`ppe_0186…jpg`, `ppe_0828…jpg`) merit annotation review because the visible clothing may not agree with the corresponding labeled vest state. This is a review flag, **not** a confirmed label defect. A small spot-check cannot estimate how common lighting, occlusion, or annotation issues are.

Detailed split/class/bounding-box counts: [`dataset_quality_report.md`](dataset_quality_report.md), [`class_distribution.csv`](class_distribution.csv), and [`bounding_box_analysis.csv`](bounding_box_analysis.csv).

## 6. Root causes supported by evidence

1. **Confidence filtering is the dominant measured FN pathway:** 132/151 PPE FN labels have a same-class candidate below 0.25 that overlaps the GT at IoU ≥0.50.
2. **No Vest has weak recall despite moderate training frequency:** 33/40 No Vest FN labels have a below-cutoff, well-localized candidate; six additional labels have same-class candidates with no overlap and one has no same-class candidate.
3. **Safety Vest recall is limited by both score and detection gaps:** 50/57 FN labels have a below-cutoff, well-localized candidate; three have no same-class candidate and four more have localization/no-overlap problems.
4. **No Vest confidence separation is weak:** mean accepted correct and incorrect scores are close, so simply shifting a class cutoff may trade recall for false positives without a clear safety margin.
5. **Some class confusion exists:** one No Helmet GT was overlapped by a confident Helmet prediction.
6. **Annotation agreement may be imperfect:** the spot-check raised possible disagreements. Do not treat them as verified label errors until an independent review.
7. **Small box scale is not the primary vest explanation in this test set:** the false-negative vest annotations are medium or large by the documented bins.

## 7. Proposed improvement experiments — not implemented

| Experiment | Problem addressed / evidence | Expected benefit | Risk or downside | Validation plan |
|---|---|---|---|---|
| Annotation adjudication and dataset balancing | Possible vest-label disagreement in spot checks; No Helmet is 22.5× rarer than Helmet; No Vest is 0.69× Safety Vest count. | More trustworthy metrics and better coverage of rare/ambiguous cases. | Human review cost; oversampling can overfit duplicates or amplify inconsistent labels. | Independently review a stratified set of train/validation errors; document agreement; rebalance only training data after approval; preserve the test set. |
| Add targeted No Vest and Safety Vest training examples | 33/40 and 50/57 misses, respectively, had good-IoU candidates below 0.25; No Vest recall is 0.344 and Safety Vest recall 0.558. | Improve class evidence/confidence in hard conditions and reduce score-threshold misses. | More data may not improve calibration; class confusion or false positives can increase. | Collect diverse workers, vest styles, distance, lighting, occlusion, and negative examples; evaluate on held-out validation and unchanged test after model approval. |
| Controlled augmentations | Spot checks include crowded/partly obscured scenes and a visible bright glare example; frequency is not established. | Better robustness to scale, contrast, illumination, and occlusion. | Unrealistic augmentation can damage reflective-strip/color cues or shift class semantics. | Compare one augmentation family at a time; inspect generated samples; report class-wise PR/AP and false positives on validation. |
| Model architecture alternatives | Existing model has high Safety Vest precision but limited recall; no architecture comparison was run. | A detector with stronger small/occluded-object features may improve recall. | Compute/model-size increase, calibration differences, deployment complexity. | Fixed data split and inference policy; compare mAP50–95, class recall, latency, and error examples without replacing production weights. |
| Resolution ablation | Test images are 640×640, production inference uses 960; the P1 vest score differed between 640 and 960 diagnostic runs (0.43392 vs 0.143993). | Identify resolution sensitivity and a stable operating configuration. | Higher resolution costs latency/memory and can change confidence/NMS behavior. | Compare 640/960/1280 on validation using identical weights/NMS; report class-wise metrics, latency, and worker-level consistency before any approval. |
| Class-specific confidence calibration | 132 PPE false negatives have a sub-0.25 high-IoU candidate; No Vest correct/incorrect scores overlap substantially. | Quantify class-wise PR operating points and calibration. | Lower class cutoffs may increase false alarms and unsafe worker assignments; selecting on test leaks information. | Sweep per-class cutoffs on validation only, include association and FP review, predefine safety costs; keep global production 0.25 until an approved decision. |

## 8. Risks and limitations

- Dataset labels are evaluation references, not verified site truth; visual discrepancies need independent adjudication.
- The model-only run does not evaluate the application’s dedicated person detector, worker association, or final assessment.
- Diagnostic predictions below 0.001 are not retained by the inference API; all candidates at or above 0.001 were analyzed.
- Test-set results should not be used to tune thresholds, choose augmentations, or select architectures. Use train/validation for experiments and reserve test for final comparison.
- Lighting/occlusion comments come from five visual spot-checks, not exhaustive review.
- No training, weight modification, threshold change, or production-code change was performed. Automated application regression tests were not rerun because no production implementation changed.

## 9. Recommended next experiment

**First conduct a blinded annotation adjudication on the vest-related false negatives and the two visually flagged images, using the validation split for iterative work and keeping the test labels immutable.** The dominant measured error is sub-0.25 confidence, but possible label disagreement could distort both recall and threshold conclusions. After label quality is established, the next controlled experiment should compare inference resolution and per-class PR curves on validation only—without changing the production cutoff or deploying a candidate model.
