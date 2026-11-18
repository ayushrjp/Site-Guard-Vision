# Controlled PPE experiments — summary and recommendation

**Run date:** 2026-11-15  
**Scope:** the original PPE checkpoint on the 119-image validation split only, except the separate annotation-only review of two flagged test images. No training, checkpoint changes, annotation edits, production changes, or deployment occurred.

## Executive findings

On this validation set, increasing inference size did **not** improve recall. With the same checkpoint, candidate floor, NMS, metric method, and 0.25 reporting cutoff, 640px produced the best measured PPE macro recall (**0.789**) and mAP50 (**0.931**), compared with 0.543 and 0.810 at 960px. It also had the lowest mean CPU inference time, **2.10 s/image**, versus 3.98 s/image at 960px. All four PPE classes had lower false-negative rates at 640 than at 960.

The offline class-wise threshold sweep also shows a recall/F1 trade-off. The best F1 cutoffs within this same validation sweep were Helmet 0.20, No Helmet 0.10, No Vest 0.10, and Safety Vest 0.10. Their class-macro F1 is 0.795, compared with 0.644 at 0.25 on the cached 960px predictions; macro recall rises from 0.543 to 0.801, while macro precision falls from 0.848 to 0.807. Because the thresholds are selected and measured on the same small validation set, this is exploratory and likely optimistic—not a deployment recommendation.

Training counts show substantial Helmet/No Helmet imbalance (2,116 vs 94) and a smaller No Vest/Safety Vest gap (741 vs 1,073). Validation misses are worst at 960px for No Helmet and No Vest, but the No Vest count gap is modest and the resolution comparison improves their recall without retraining. Class imbalance is therefore a plausible contributor, especially for No Helmet, but these experiments do not establish it as the cause of low recall.

## Resolution results

All metrics are PPE-class macro averages excluding the checkpoint’s `Person` class. Precision, recall, and F1 use confidence 0.25 and class-aware greedy one-to-one matching at IoU 0.50. AP is computed over predictions retained at confidence 0.001 with 101-point interpolation; mAP50–95 averages IoU 0.50 through 0.95 in 0.05 increments.

| Inference size | Precision | Recall | F1 | mAP50 | mAP50–95 | Mean CPU ms/image |
|---:|---:|---:|---:|---:|---:|---:|
| 640 | 0.967 | **0.789** | **0.868** | **0.931** | **0.635** | **2,099** |
| 800 | **0.970** | 0.717 | 0.820 | 0.910 | 0.598 | 3,045 |
| 960 | 0.848 | 0.543 | 0.644 | 0.810 | 0.489 | 3,983 |

Class-wise precision / recall at confidence 0.25:

| Class | 640 P / R | 800 P / R | 960 P / R |
|---|---:|---:|---:|
| Helmet | 0.923 / 0.828 | 0.896 / 0.819 | 0.813 / 0.789 |
| No Helmet | 1.000 / 0.818 | 1.000 / 0.636 | 0.600 / 0.273 |
| No Vest | 0.954 / 0.689 | 1.000 / 0.633 | 1.000 / 0.422 |
| Safety Vest | 0.991 / 0.823 | 0.982 / 0.780 | 0.980 / 0.688 |

Relative to 960px, 640px reduced the measured mean inference time by 47.3% and raised macro recall by 24.6 percentage points and mAP50 by 12.1 points on these validation images. Runtime is wall-clock CPU timing in this environment; it is not a production-device latency guarantee.

## Threshold sweep results

| Class | F1 at 0.25 (P / R) | Best swept cutoff | F1 at best cutoff (P / R) |
|---|---:|---:|---:|
| Helmet | 0.801 (0.813 / 0.789) | 0.20 | 0.825 (0.788 / 0.866) |
| No Helmet | 0.375 (0.600 / 0.273) | 0.10 | 0.692 (0.600 / 0.818) |
| No Vest | 0.594 (1.000 / 0.422) | 0.10 | 0.785 (0.912 / 0.689) |
| Safety Vest | 0.808 (0.980 / 0.688) | 0.10 | 0.876 (0.929 / 0.830) |

The best individual cutoffs yield mean per-class P/R/F1 of 0.807 / 0.801 / 0.795, versus 0.848 / 0.543 / 0.644 when every class uses 0.25 at 960px. This is a potentially useful recall trade-off for future independent validation. It also admits more false positives: at the selected cutoff, No Helmet precision is 0.600 and Safety Vest precision is 0.929. The No Helmet result is especially uncertain because there are only 11 validation ground-truth objects.

## Imbalance and validation false-negative rates

Train / validation instance counts: Helmet 2,116 / 232; No Helmet 94 / 11; No Vest 741 / 90; Safety Vest 1,073 / 141. Helmet has 22.5 times the No Helmet training instances. No Vest has 69.1% as many training instances as Safety Vest.

At the unchanged 0.25 cutoff and IoU 0.50:

| Class | Validation FN rate at 640 | Validation FN rate at 800 | Validation FN rate at 960 |
|---|---:|---:|---:|
| Helmet | 17.2% (40/232) | 18.1% (42/232) | 21.1% (49/232) |
| No Helmet | 18.2% (2/11) | 36.4% (4/11) | 72.7% (8/11) |
| No Vest | 31.1% (28/90) | 36.7% (33/90) | 57.8% (52/90) |
| Safety Vest | 17.7% (25/141) | 22.0% (31/141) | 31.2% (44/141) |

The No Helmet results are consistent with concern about sparse training/validation examples, but the sample is small. No Vest also has weak recall at 960 despite a much smaller count gap than Helmet/No Helmet; it improves materially at 640. Counts alone cannot distinguish imbalance from resolution-dependent confidence, appearance coverage, or annotation disagreement. Class-aware sampling and new training examples remain untested and are not justified as an immediate intervention by these measurements alone.

## Annotation review

The two flagged test images were reviewed using original images, ground-truth boxes, and existing diagnostic predictions with labels/confidences. They were **not** included in the controlled validation metrics or threshold selection. The overlays and per-box comparisons are in [`annotation_review.md`](annotation_review.md). Possible vest-state disagreements remain review flags, not confirmed annotation errors; no annotation was modified.

## Reference baseline and comparability

The original test-set report (`reports/ppe_evaluation/run_20261115/`) remains unchanged: 90 test images, PPE macro precision 0.797, recall 0.565, mAP50 0.747, and mAP50–95 0.410 at the reported 960px / 0.25 operating point. These test metrics are not directly compared numerically with the validation rows above because they come from different image splits. The resolution comparison uses the same original checkpoint and validation examples at every input size; the 960px validation result is the within-experiment reference.

The model SHA-256 in [`experiment_settings.json`](experiment_settings.json) matches the recorded original checkpoint. NMS IoU remained 0.70; inference candidate floor remained 0.001; confidence 0.25 remained the reference for fixed-threshold metrics. The custom evaluator is documented in [`experiment_settings.json`](experiment_settings.json).

## Recommendation: next experiment

**Most promising next experiment: confirm the 640px inference result on a new, independent validation sample using the unchanged checkpoint and fixed confidence 0.25.** Do not tune on the test split and do not deploy 640px based on this result alone.

1. **Evidence:** 640px had higher macro recall (0.789 vs 0.543), higher mAP50 (0.931 vs 0.810), and lower mean CPU time (2.10 vs 3.98 s/image) than 960px on identical validation images. Each PPE class had lower FN rate at 640px.
2. **Expected benefit:** if replicated, a lower inference size could improve recall and reduce compute without retraining or changing class cutoffs.
3. **Risks:** the 119-image validation set may not represent deployment scenes; resolution-dependent confidence can be dataset-specific; CPU timing may not transfer to target hardware; annotation disagreements can bias metrics.
4. **Validation criteria:** on an independently collected/adjudicated validation set, compare 640 and 960 with identical weights, 0.25 confidence, NMS IoU 0.70, and matching method; report class-wise precision/recall/F1, mAP50–95, FN examples, and target-device latency. Predefine acceptable precision and recall criteria before running.
5. **Retraining:** **not justified yet.** The observed resolution and threshold effects come from the same weights. Imbalance is descriptive, not causal. Consider a separate class-aware sampling or targeted-data training ablation only if the independent validation confirms a class-specific deficit after label review.

No experimental setting was deployed. Production model weights, confidence cutoff, person detector, association geometry, and test data remain unchanged.

## Files

- [`resolution_comparison.csv`](resolution_comparison.csv) and [`resolution_class_metrics.csv`](resolution_class_metrics.csv)
- [`class_threshold_comparison.csv`](class_threshold_comparison.csv)
- [`class_imbalance_analysis.md`](class_imbalance_analysis.md)
- [`annotation_review.md`](annotation_review.md)
- Resolution recall, mAP, and timing plots: [`resolution_vs_recall.png`](resolution_vs_recall.png), [`resolution_vs_map.png`](resolution_vs_map.png), [`inference_time_vs_resolution.png`](inference_time_vs_resolution.png)
- Class and threshold plots: [`classwise_precision_recall.png`](classwise_precision_recall.png), [`class_f1_vs_threshold.png`](class_f1_vs_threshold.png)
