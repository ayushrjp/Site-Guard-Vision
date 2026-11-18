# Annotation review: flagged test images

These two test images are reviewed only for annotation quality. They are excluded from all controlled experiment scoring and threshold/resolution decisions. No labels were changed. Bounding boxes below come from the existing diagnostic predictions at 960px, NMS IoU 0.70, candidate floor 0.001; the production review cutoff remains 0.25. Overlays show all ground-truth boxes (green) and predicted boxes with confidence ≥0.10 (red).

Classes: Helmet, No Helmet, No Vest, Safety Vest. Image coordinates are pixels in the original 640×640 image. The visual review is not a relabeling adjudication.

## `ppe_0186_jpg.rf.e2e0fe8a088ba9480c8c6d4c2ccece79.jpg`

- Original: `../../../dataset/test/images/ppe_0186_jpg.rf.e2e0fe8a088ba9480c8c6d4c2ccece79.jpg`
- Ground truth: `dataset/test/labels/ppe_0186_jpg.rf.e2e0fe8a088ba9480c8c6d4c2ccece79.txt`
- Overlay: [`ppe_0186_jpg.rf.e2e0fe8a088ba9480c8c6d4c2ccece79_review_overlay.jpg`](ppe_0186_jpg.rf.e2e0fe8a088ba9480c8c6d4c2ccece79_review_overlay.jpg)
- Ground-truth classes: Helmet × 2, Person × 2, Safety Vest × 2
- Red predictions shown (confidence ≥0.10): Helmet 0.755, box [402.2232971191406, 99.9049072265625, 511.9091796875, 226.65403747558594]; Helmet 0.678, box [170.04490661621094, 37.21342849731445, 312.836181640625, 161.9026336669922]

| GT class | GT box (xyxy) | Best same-class prediction | Confidence | IoU |
|---|---|---|---:|---:|
| Helmet | `[403.5, 102.2, 504.5, 227.8]` | `[402.2, 99.9, 511.9, 226.7]` | 0.755007 | 0.897 |
| Person | `[10.3, 29.7, 304.8, 604.2]` | none | — | — |
| Safety Vest | `[14.0, 157.2, 228.0, 502.8]` | none | — | — |
| Safety Vest | `[284.2, 195.8, 478.8, 477.2]` | none | — | — |
| Person | `[297.2, 106.3, 528.8, 597.8]` | none | — | — |
| Helmet | `[176.5, 27.0, 308.5, 164.0]` | `[170.0, 37.2, 312.8, 161.9]` | 0.677546 | 0.847 |

## `ppe_0828_jpg.rf.71ebe13b3ae36071d34b5ee25d1549a7.jpg`

- Original: `../../../dataset/test/images/ppe_0828_jpg.rf.71ebe13b3ae36071d34b5ee25d1549a7.jpg`
- Ground truth: `dataset/test/labels/ppe_0828_jpg.rf.71ebe13b3ae36071d34b5ee25d1549a7.txt`
- Overlay: [`ppe_0828_jpg.rf.71ebe13b3ae36071d34b5ee25d1549a7_review_overlay.jpg`](ppe_0828_jpg.rf.71ebe13b3ae36071d34b5ee25d1549a7_review_overlay.jpg)
- Ground-truth classes: Helmet × 2, No Helmet × 1, No Vest × 1, Person × 2, Safety Vest × 2
- Red predictions shown (confidence ≥0.10): Helmet 0.814, box [399.9117431640625, 257.3191833496094, 466.1959228515625, 349.3138427734375]; No Helmet 0.416, box [347.7627868652344, 58.18992233276367, 392.567138671875, 108.2507553100586]; Helmet 0.334, box [152.39637756347656, 83.50421142578125, 220.66217041015625, 161.54261779785156]; Safety Vest 0.194, box [299.4560241699219, 101.98587799072266, 382.3149108886719, 250.6488037109375]

| GT class | GT box (xyxy) | Best same-class prediction | Confidence | IoU |
|---|---|---|---:|---:|
| Helmet | `[405.3, 264.2, 458.8, 330.8]` | `[399.9, 257.3, 466.2, 349.3]` | 0.814336 | 0.583 |
| Person | `[379.7, 261.5, 531.2, 637.5]` | none | — | — |
| Safety Vest | `[388.5, 352.0, 516.5, 584.0]` | `[387.9, 355.7, 514.2, 593.4]` | 0.007996 | 0.925 |
| No Helmet | `[347.5, 58.8, 392.5, 112.3]` | `[347.8, 58.2, 392.6, 108.3]` | 0.416070 | 0.909 |
| Helmet | `[153.5, 85.2, 217.5, 154.8]` | `[152.4, 83.5, 220.7, 161.5]` | 0.333658 | 0.835 |
| No Vest | `[313.8, 103.8, 384.2, 253.3]` | none | — | — |
| Person | `[294.5, 61.2, 390.5, 418.8]` | none | — | — |
| Safety Vest | `[42.7, 162.8, 166.2, 325.2]` | `[48.0, 162.7, 177.6, 323.4]` | 0.077970 | 0.867 |

## Review conclusion

The existing spot-check raised possible vest-state annotation disagreements in both images; this review records the original labels and model boxes, but does not adjudicate either as an error. In particular, model confidence is not evidence that an annotation is correct or incorrect. A trained annotator should compare each person’s clothing against the project labeling policy, document the decision independently, and leave the test labels immutable for these experiments.
