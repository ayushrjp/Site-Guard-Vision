# Final recommendation: 640px PPE configuration

## Decision

The evidence supports considering a 640px configuration for a future controlled production change, subject to target-hardware testing and annotation review; it does not authorize deployment.

## Measured evidence

- Evaluated **90 test images** and **623 annotations** (409 PPE objects).
- Macro recall difference (640−960): **+0.169**; paired image-bootstrap 95% CI **[+0.111, +0.236]**. The paired bootstrap interval for PPE macro recall is above zero.
- Macro precision difference: +0.045; F1: +0.138; mAP50: +0.100; mAP50–95: +0.112.
- Mean CPU inference: 2029.0 ms/image at 640 versus 4040.7 ms/image at 960.
- Class-wise point estimates and bootstrap intervals are reported in `independent_validation_report.md` and `class_wise_comparison.csv`; No Helmet has only 24 test annotations.

## Conditions before any future change

1. Do not deploy on this analysis alone; retain production weights, 0.25 threshold, person detector, and association geometry.
2. Review the two flagged annotations under a documented protocol without changing labels in this experiment.
3. If the confidence interval is inconclusive or class-specific trade-offs are unacceptable, acquire a larger independent/adjudicated validation set and predefine precision/recall acceptance margins.
4. Verify inference latency on target hardware and run application-level regressions before any separately approved production release.

No retraining is required to test inference size, and no weights or production behavior were changed.
