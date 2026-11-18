# PPE Dataset Quality Report

**Source:** project `dataset/{train,valid,test}` and its YOLO labels  
**Audit date:** 2026-11-15  
**Scope:** counts, image dimensions, normalized box geometry, and a small qualitative visual spot-check. No labels were edited.

## Dataset integrity and size

The dataset contains **1,206 images and 7,724 labeled objects**:

| Split | Images | Objects | Dimensions |
|---|---:|---:|---|
| Train | 997 | 6,386 | 640×640 for all images |
| Validation | 119 | 715 | 640×640 for all images |
| Test | 90 | 623 | 640×640 for all images |

Each image had a corresponding label file. The previous integrity audit found no malformed/out-of-range boxes, orphan labels, missing label files, or corrupt images. The dataset uses five YOLO classes: Helmet, No Helmet, No Vest, Person, Safety Vest.

## Class distribution and imbalance

| Class | Train | Validation | Test | Train share |
|---|---:|---:|---:|---:|
| Helmet | 2,116 | 232 | 195 | 33.1% |
| No Helmet | 94 | 11 | 24 | 1.5% |
| No Vest | 741 | 90 | 61 | 11.6% |
| Person | 2,362 | 241 | 214 | 37.0% |
| Safety Vest | 1,073 | 141 | 129 | 16.8% |

Among PPE classes, Helmet has **22.5×** as many training instances as No Helmet. No Vest has 741 training instances versus 1,073 Safety Vest instances (69.1% as many); it is less common, but not by an order of magnitude. This imbalance is evidence to consider during experiment design, not proof that class count caused the observed errors.

Machine-readable counts: [`class_distribution.csv`](class_distribution.csv).

## Bounding-box size and shape

Boxes are normalized YOLO `xywh`. For descriptive size groups, box area was converted to an equivalent 640×640 pixel area; “small” is area below 32² px, “medium” 32²–<96² px, and “large” at least 96² px. These are analysis bins, not model settings.

Across all splits, 442/7,724 boxes (5.7%) were small, 2,540 (32.9%) medium, and 4,742 (61.4%) large. In the test split:

| Class | Instances | Small | Medium | Large | Median equivalent side |
|---|---:|---:|---:|---:|---:|
| Helmet | 195 | 19 | 124 | 52 | 62.8 px |
| No Helmet | 24 | 2 | 19 | 3 | 60.6 px |
| No Vest | 61 | 0 | 22 | 39 | 111.9 px |
| Safety Vest | 129 | 0 | 25 | 104 | 146.8 px |

Among the false negatives, 45/57 Safety Vest boxes and 24/40 No Vest boxes fall in the large bin; 12 Safety Vest and 16 No Vest misses are medium, with none of these test-set vest-class misses in the small bin. Thus, tiny box size alone does not explain most vest-class false negatives. Detailed per-split width, height, area percentiles, and size counts are in [`bounding_box_analysis.csv`](bounding_box_analysis.csv).

## Image conditions and visual spot-check

All images are stored at 640×640; the 90-image model evaluation used `imgsz=960`. Only selected error examples were visually reviewed, not all 1,206 images, so the following observations are illustrative and not dataset prevalence estimates:

- `ppe_0186…jpg`: bright/high-key background; one orange vest is visually apparent, while the other worker appears to wear a light-blue shirt. The label file has Safety Vest boxes for both. This is a **possible annotation disagreement requiring human review**, not a confirmed relabeling.
- `ppe_0828…jpg`: several workers and a large foreground road sign obscure portions of the scene. A No Vest label falls near a worker whose clothing appears high-visibility in the image; this also merits adjudication rather than an automatic claim of bad labeling.
- `ppe_0637…jpg`: overhead, small workers and a bright welding-like glare are visible. This is a plausible difficult-visibility example, but one image does not establish a general lighting trend.
- `ppe_0008…jpg`: multiple workers and PPE items are close together; some people/PPE are small or near image edges.
- `ppe_0347…jpg`: the worker and orange vest are relatively large and visible, yet the model’s vest candidate is very low confidence and has poor overlap with its annotation. This indicates that small-object scale is not the sole issue.

Visual notes are not ground-truth corrections. The labels and original images should be reviewed under a consistent annotation protocol before changing training data.

## Interpretation

The training set has a marked No Helmet instance deficit and a moderate No Vest/Safety Vest count difference. At the same time, most test vest boxes are medium/large, and many false negatives have a same-class, well-localized candidate whose score is below 0.25. The evidence therefore supports investigating annotation consistency, confidence calibration, class-specific decision behavior, and image-size sensitivity—not simply adding generic small-object augmentation.
