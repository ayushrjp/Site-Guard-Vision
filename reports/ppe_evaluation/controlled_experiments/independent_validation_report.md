# Independent validation: 640px vs 960px PPE inference

## Dataset separation and prior use

Dataset inventory: train has 997 images/997 label files; validation has 119/119; test has 90/90. No additional labeled dataset or unused validation subset was found. All 119 validation images had already been used in prior resolution/threshold experiments, so they were not reused for this confirmation.

The pre-existing 90-image test split (623 annotations) was used as the separate confirmation split. It had already been used for the original 960px baseline report and diagnostic error review; therefore this is an independent comparison with respect to the 640px selection (which used validation only), but not a previously untouched dataset. With explicit authorization, all 90 test images—including the two visually flagged images—were included unchanged. No labels were edited and no test-set tuning occurred.

Evaluated **90 images**, **623 total labeled objects**, including **409 PPE objects**. Counts by label: Helmet 195, No Helmet 24, No Vest 61, Person 214, Safety Vest 129.

## Controlled settings and method

- Same checkpoint SHA-256 for both sizes: `2b33d4d016f9751c5a25f4a72ce050e0a7e4e140b11c1669978d7154003a4f61`; matches the recorded baseline: **True**.
- Input sizes: 640 and 960; confidence cutoff **0.25** for reported P/R/F1; inference candidate floor **0.001** for AP; NMS IoU **0.7**.
- Same class mapping, CPU device, max detections (300), test image order, custom class-aware confidence-ordered one-to-one matching, and evaluator implementation.
- Precision/recall/F1 use IoU 0.50. mAP50–95 averages AP over IoU 0.50 through 0.95 in 0.05 steps, using 101-point interpolation. Macro metrics exclude the checkpoint’s Person class, consistent with prior PPE reporting.
- Inference time is CPU wall time around each Ultralytics prediction; one first-image warm-up per size was excluded from per-image timings.

## Overall results

| Input size | Precision | Recall | F1 | mAP50 | mAP50–95 |
|---:|---:|---:|---:|---:|---:|
| 640 | 0.842 | 0.734 | 0.781 | 0.847 | 0.521 |
| 960 | 0.797 | 0.565 | 0.644 | 0.747 | 0.410 |

Paired bootstrap 95% confidence intervals for differences (**640 minus 960**) are image-cluster percentile intervals. They resample the same test images for both settings, preserving the paired comparison.
These intervals quantify sampling uncertainty for this fixed split; they are not a formal hypothesis test and do not adjust for multiple class-wise comparisons.

| Macro metric | Difference (640−960) | Paired bootstrap 95% CI |
|---|---:|---:|
| precision | +0.045 | [-0.013, +0.101] |
| recall | +0.169 | [+0.111, +0.236] |
| f1 | +0.138 | [+0.093, +0.198] |
| AP50 | +0.100 | [+0.048, +0.155] |
| mAP50_95 | +0.112 | [+0.081, +0.141] |

## Class-wise results

| Class | GT | P 640 | R 640 | F1 640 | P 960 | R 960 | F1 960 | Δ recall (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Helmet | 195 | 0.836 | 0.887 | 0.861 | 0.702 | 0.774 | 0.737 | +0.113 (+0.054, +0.172) |
| No Helmet | 24 | 0.842 | 0.667 | 0.744 | 0.778 | 0.583 | 0.667 | +0.083 (-0.083, +0.286) |
| Safety Vest | 129 | 0.883 | 0.760 | 0.817 | 0.900 | 0.558 | 0.689 | +0.202 (+0.122, +0.292) |
| No Vest | 61 | 0.809 | 0.623 | 0.704 | 0.808 | 0.344 | 0.483 | +0.279 (+0.155, +0.436) |

640px has higher point recall for each class in this test split. The paired class-wise recall intervals are above zero for Helmet, No Vest, and Safety Vest; the No Helmet interval crosses zero and is wide. Precision differences are mixed: Helmet and No Helmet are higher at 640, No Vest is nearly unchanged, and Safety Vest is slightly lower. No Helmet has only 24 labeled test objects, so its class estimate is less stable. Bootstrap intervals describe sampling uncertainty over images, not annotation error, distribution shift, or all future operating conditions.

## Inference performance

| Size | Mean ms/image | Median ms/image | P95 ms/image | Total inference seconds | Evaluation wall seconds* |
|---:|---:|---:|---:|---:|---:|
| 640 | 2029.0 | 2009.6 | 2212.3 | 182.61 | 205.60 |
| 960 | 4040.7 | 4019.1 | 4447.5 | 363.66 | 369.44 |

*Evaluation wall time includes one warm-up, inference, matching, and point-metric calculation, but excludes model loading and bootstrap reporting. Mean CPU inference was 49.8% faster at 640px in this environment. Timing is hardware/runtime specific.

## Limitations

- Test is a distinct split from validation, but it is not pristine: 960px test baseline metrics and diagnostic error analysis were already available. The resolution choice (640px) came from validation; the test comparison was fixed in advance and not tuned.
- Two test images had possible annotation disagreements flagged during visual review. They remain in this full-split scoring set because no label was adjudicated or changed; their uncertain labels may affect metrics.
- The No Helmet class has only 24 objects; class-level uncertainty is substantial. Bootstrap resampling cannot compensate for missing diversity or systematic annotation issues.
- This evaluates PPE checkpoint detections only, not the application’s dedicated person detector, worker association, site compliance, target hardware, or video performance.
- The experiment compares the same checkpoint and fixed 0.25 threshold; it does not validate class-specific thresholds or retrained models.

## Conclusion

See [`final_recommendation.md`](final_recommendation.md) for the production-readiness decision. No experimental settings were deployed; weights, confidence cutoff, association geometry, person detection, and test annotations were not modified.
