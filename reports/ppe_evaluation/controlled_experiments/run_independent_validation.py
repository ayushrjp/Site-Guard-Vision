"""Fixed-setting 640 vs 960 confirmation on the existing held-out test split.

No threshold or model selection occurs here. The test split is used only for
this pre-specified comparison and the two previously flagged test images are
included, following explicit user authorization.
"""
from __future__ import annotations

import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports" / "ppe_evaluation"
OUT = BASE / "controlled_experiments"
MODEL_PATH = ROOT / "models" / "yolo_ppe_core.pt"
sys.path.insert(0, str(BASE))
import evaluate_dataset as ev  # noqa: E402

SIZES = (640, 960)
CONF_FLOOR = 0.001
CONF_REPORT = 0.25
NMS_IOU = 0.70
IOU_THRESHOLDS = ev.IOU_THRESHOLDS
PPE_CLASSES = ev.PPE_CLASSES
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20261116


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def infer(model, samples, size):
    wall_start = time.perf_counter()
    # Warm-up is excluded from the timed per-image samples.
    if samples:
        model.predict(source=str(samples[0][0]), conf=CONF_FLOOR,
                      iou=NMS_IOU, imgsz=size, device="cpu",
                      verbose=False, max_det=300, save=False)
    images, inference_ms = [], []
    for index, (path, gts) in enumerate(samples, 1):
        started = time.perf_counter()
        result = model.predict(
            source=str(path), conf=CONF_FLOOR, iou=NMS_IOU, imgsz=size,
            device="cpu", verbose=False, max_det=300, save=False,
        )[0]
        inference_ms.append((time.perf_counter() - started) * 1000)
        predictions = []
        for box, confidence, model_class in zip(
            result.boxes.xyxy.tolist(),
            result.boxes.conf.tolist(),
            result.boxes.cls.tolist(),
        ):
            model_class = int(model_class)
            dataset_class = next(
                (ds for ds, mc in ev.DATASET_TO_MODEL.items() if mc == model_class),
                None,
            )
            if dataset_class is not None:
                predictions.append({
                    "class": dataset_class,
                    "confidence": float(confidence),
                    "bbox": [float(v) for v in box],
                    "tp": [],
                })
        images.append({
            "image": path.name,
            "predictions": predictions,
            "ground_truth": gts,
        })
        print(f"test imgsz={size} [{index}/{len(samples)}] {path.name}", flush=True)
    wall_seconds = time.perf_counter() - wall_start
    return images, inference_ms, wall_seconds


def match_predictions(images):
    for image in images:
        for pred in image["predictions"]:
            pred["tp"] = []
    for threshold in IOU_THRESHOLDS:
        for image in images:
            ev.match_image(image["predictions"], image["ground_truth"], threshold)


def gt_count_by_image(images):
    result = np.zeros((len(images), len(ev.NAMES)), dtype=int)
    for ii, image in enumerate(images):
        for gt in image["ground_truth"]:
            result[ii, gt["class"]] += 1
    return result


def point_metrics(images):
    counts = Counter(
        gt["class"] for image in images for gt in image["ground_truth"]
    )
    class_rows = []
    for cls in PPE_CLASSES:
        metric = ev.metrics(images, counts, cls, CONF_REPORT, 0)
        aps = [
            ev.class_ap(images, counts, cls, j)
            for j in range(len(IOU_THRESHOLDS))
        ]
        class_rows.append({
            "class_id": cls,
            "class": ev.NAMES[cls],
            "ground_truth": counts[cls],
            **metric,
            "AP50": aps[0],
            "mAP50_95": float(np.mean(aps)),
        })
    macro = {
        key: float(np.mean([row[key] for row in class_rows]))
        for key in ("precision", "recall", "f1", "AP50", "mAP50_95")
    }
    return counts, class_rows, macro


def ap101(recall, precision):
    if len(recall) == 0:
        return 0.0
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(p)-2, -1, -1):
        p[i] = max(p[i], p[i+1])
    return float(np.mean([
        np.max(p[r >= point]) if np.any(r >= point) else 0
        for point in np.linspace(0, 1, 101)
    ]))


def prepare_bootstrap_arrays(images):
    gt_image_class = gt_count_by_image(images)
    predictions = {}
    for cls in PPE_CLASSES:
        records = []
        for image_index, image in enumerate(images):
            for pred in image["predictions"]:
                if pred["class"] == cls:
                    records.append((image_index, pred))
        records.sort(key=lambda item: item[1]["confidence"], reverse=True)
        if records:
            predictions[cls] = {
                "image": np.asarray([x[0] for x in records], dtype=int),
                "confidence": np.asarray(
                    [x[1]["confidence"] for x in records], dtype=float),
                "tp": np.asarray([x[1]["tp"] for x in records], dtype=bool),
            }
        else:
            predictions[cls] = {
                "image": np.zeros(0, dtype=int),
                "confidence": np.zeros(0, dtype=float),
                "tp": np.zeros((0, len(IOU_THRESHOLDS)), dtype=bool),
            }
    return gt_image_class, predictions


def weighted_class_stats(weights, gt_counts, pred, confidence=CONF_REPORT):
    selected = pred["confidence"] >= confidence
    image_indices = pred["image"][selected]
    pred_weights = weights[image_indices]
    n_pred = int(np.sum(pred_weights))
    tp = int(np.sum(pred_weights * pred["tp"][selected, 0]))
    n_gt = int(gt_counts)
    fp = n_pred - tp
    fn = n_gt - tp
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_gt if n_gt else float("nan")
    f1 = (2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def weighted_ap(weights, gt_count, pred, iou_index):
    if gt_count <= 0:
        return float("nan")
    if not len(pred["image"]):
        return 0.0
    image_weights = weights[pred["image"]]
    positive = image_weights > 0
    image_weights = image_weights[positive]
    tp = pred["tp"][positive, iou_index].astype(float) * image_weights
    fp = (1.0 - pred["tp"][positive, iou_index].astype(float)) * image_weights
    if len(tp) == 0:
        return 0.0
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recall = cum_tp / gt_count
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    return ap101(recall, precision)


def bootstrap_metric(weights, gt_by_image, predictions):
    gt_counts = np.sum(gt_by_image * weights[:, None], axis=0)
    per_class, ap_by_class = {}, {}
    for cls in PPE_CLASSES:
        per_class[cls] = weighted_class_stats(
            weights, gt_counts[cls], predictions[cls])
        aps = [
            weighted_ap(weights, gt_counts[cls], predictions[cls], j)
            for j in range(len(IOU_THRESHOLDS))
        ]
        ap_by_class[cls] = np.asarray(aps, dtype=float)
    macro = {}
    for metric in ("precision", "recall", "f1"):
        vals = np.asarray([per_class[c][metric] for c in PPE_CLASSES])
        vals = vals[np.isfinite(vals)]
        macro[metric] = float(np.mean(vals)) if len(vals) else float("nan")
    macro["AP50"] = float(np.nanmean([ap_by_class[c][0] for c in PPE_CLASSES]))
    macro["mAP50_95"] = float(np.nanmean([
        np.nanmean(ap_by_class[c]) for c in PPE_CLASSES
    ]))
    return macro, per_class


def bootstrap_differences(images_by_size):
    prepared = {
        size: prepare_bootstrap_arrays(images_by_size[size])
        for size in SIZES
    }
    n = len(images_by_size[SIZES[0]])
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    metric_names = ("precision", "recall", "f1", "AP50", "mAP50_95")
    macro_delta = {metric: [] for metric in metric_names}
    class_delta = {
        (cls, metric): []
        for cls in PPE_CLASSES
        for metric in ("precision", "recall", "f1")
    }
    present_reps = Counter()
    for rep in range(BOOTSTRAP_REPS):
        sampled = rng.integers(0, n, size=n)
        weights = np.bincount(sampled, minlength=n)
        macro_results, class_results = {}, {}
        for size in SIZES:
            gt_by_image, predictions = prepared[size]
            macro_results[size], class_results[size] = bootstrap_metric(
                weights, gt_by_image, predictions)
        for metric in metric_names:
            macro_delta[metric].append(
                macro_results[640][metric] - macro_results[960][metric])
        for cls in PPE_CLASSES:
            gt_counts_640 = np.sum(prepared[640][0] * weights[:, None], axis=0)
            gt_counts_960 = np.sum(prepared[960][0] * weights[:, None], axis=0)
            if gt_counts_640[cls] > 0 and gt_counts_960[cls] > 0:
                present_reps[cls] += 1
            for metric in ("precision", "recall", "f1"):
                a = class_results[640][cls][metric]
                b = class_results[960][cls][metric]
                if np.isfinite(a) and np.isfinite(b):
                    class_delta[(cls, metric)].append(a-b)
        if (rep + 1) % 500 == 0:
            print(f"Bootstrap {rep+1}/{BOOTSTRAP_REPS}", flush=True)

    rows = []
    for metric in metric_names:
        values = np.asarray(macro_delta[metric], dtype=float)
        rows.append({
            "scope": "PPE_macro",
            "class": "PPE_macro",
            "metric": metric,
            "delta_640_minus_960": float(np.mean(values)),
            "ci95_low": float(np.percentile(values, 2.5)),
            "ci95_high": float(np.percentile(values, 97.5)),
            "bootstrap_replicates": len(values),
            "replicates_with_class_gt": "",
            "method": "paired image-cluster percentile bootstrap",
        })
    for cls in PPE_CLASSES:
        for metric in ("precision", "recall", "f1"):
            values = np.asarray(class_delta[(cls, metric)], dtype=float)
            rows.append({
                "scope": "class",
                "class": ev.NAMES[cls],
                "metric": metric,
                "delta_640_minus_960": float(np.mean(values)) if len(values) else "",
                "ci95_low": float(np.percentile(values, 2.5)) if len(values) else "",
                "ci95_high": float(np.percentile(values, 97.5)) if len(values) else "",
                "bootstrap_replicates": len(values),
                "replicates_with_class_gt": present_reps[cls],
                "method": "paired image-cluster percentile bootstrap",
            })
    return rows


def main():
    samples = ev.load_split("test")
    model = YOLO(str(MODEL_PATH))
    all_images, all_elapsed, total_wall = {}, {}, {}
    summary_rows, class_rows, performance_rows = [], [], []
    for size in SIZES:
        evaluation_started = time.perf_counter()
        image_cache = OUT / f"independent_test_predictions_{size}.json"
        timing_cache = OUT / f"independent_test_timing_{size}.json"
        cache_loaded = image_cache.is_file() and timing_cache.is_file()
        if cache_loaded:
            images = json.loads(image_cache.read_text(encoding="utf-8"))
            timing = json.loads(timing_cache.read_text(encoding="utf-8"))
            elapsed = timing["per_image_ms"]
            wall_seconds = timing["inference_loop_wall_seconds"]
        else:
            images, elapsed, wall_seconds = infer(model, samples, size)
            # Persist raw predictions before metrics/bootstrap so a later
            # reporting failure does not require repeating CPU inference.
            image_cache.write_text(
                json.dumps(images, separators=(",", ":")), encoding="utf-8")
            timing_cache.write_text(json.dumps({
                "per_image_ms": elapsed,
                "inference_loop_wall_seconds": wall_seconds,
                "warmup_included_in_loop_wall": True,
                "warmup_excluded_from_per_image_ms": True,
            }), encoding="utf-8")
        metrics_started = time.perf_counter()
        match_predictions(images)
        gt_counts, per_class, macro = point_metrics(images)
        metric_seconds = time.perf_counter() - metrics_started
        evaluation_wall_seconds = (
            wall_seconds + metric_seconds if cache_loaded
            else time.perf_counter() - evaluation_started
        )
        all_images[size] = images
        all_elapsed[size] = elapsed
        total_wall[size] = wall_seconds
        summary_rows.append({
            "imgsz": size,
            "split": "test_holdout",
            "images": len(images),
            "ground_truth_ppe_objects": sum(gt_counts[c] for c in PPE_CLASSES),
            "confidence_threshold": CONF_REPORT,
            "candidate_floor": CONF_FLOOR,
            "nms_iou": NMS_IOU,
            "evaluation_iou_threshold": 0.50,
            "precision": macro["precision"],
            "recall": macro["recall"],
            "f1": macro["f1"],
            "mAP50": macro["AP50"],
            "mAP50_95": macro["mAP50_95"],
        })
        for row in per_class:
            class_rows.append({
                "imgsz": size,
                "split": "test_holdout",
                **row,
            })
        inference_total = sum(elapsed) / 1000
        performance_rows.append({
            "imgsz": size,
            "images": len(images),
            "device": "CPU",
            "mean_inference_ms_per_image": float(np.mean(elapsed)),
            "median_inference_ms_per_image": float(np.median(elapsed)),
            "p95_inference_ms_per_image": float(np.percentile(elapsed, 95)),
            "total_inference_seconds": inference_total,
            "end_to_end_inference_loop_seconds": wall_seconds,
            "evaluation_wall_seconds_including_warmup_matching_metrics":
                evaluation_wall_seconds,
            "warmup": "one first-image prediction excluded from per-image timing",
        })
        print("TEST SUMMARY", json.dumps(summary_rows[-1]), flush=True)

    uncertainty_rows = bootstrap_differences(all_images)
    write_csv(OUT / "resolution_final_comparison.csv", summary_rows,
              list(summary_rows[0]))
    write_csv(OUT / "class_wise_comparison.csv", class_rows, list(class_rows[0]))
    write_csv(OUT / "inference_performance.csv", performance_rows,
              list(performance_rows[0]))
    write_csv(OUT / "paired_bootstrap_uncertainty.csv", uncertainty_rows,
              list(uncertainty_rows[0]))

    baseline = json.loads((BASE / "run_20261115" / "model_metrics.json").read_text(
        encoding="utf-8"))
    class_lookup = {
        (int(row["imgsz"]), row["class"]): row for row in class_rows
    }
    macro_lookup = {int(row["imgsz"]): row for row in summary_rows}
    delta = {
        key: macro_lookup[640][key] - macro_lookup[960][key]
        for key in ("precision", "recall", "f1", "mAP50", "mAP50_95")
    }
    test_gt = sum(
        int(class_lookup[(640, ev.NAMES[c])]["ground_truth"])
        for c in PPE_CLASSES
    )
    annotation_counts = Counter(
        gt["class"] for _path, labels in samples for gt in labels
    )
    total_gt = sum(annotation_counts.values())
    annotation_counts = {
        ev.NAMES[c]: int(annotation_counts[c]) for c in ev.ALL_CLASSES
    }
    run_config = {
        "model_sha256": hash_file(MODEL_PATH),
        "baseline_model_sha256": json.loads(
            (BASE / "baseline_configuration.json").read_text(encoding="utf-8")
        )["ppe_model_sha256"],
        "split": "test",
        "images": len(samples),
        "annotations": total_gt,
        "annotations_by_class": annotation_counts,
        "ppe_annotations": test_gt,
        "image_sizes": list(SIZES),
        "confidence_threshold": CONF_REPORT,
        "inference_candidate_floor": CONF_FLOOR,
        "nms_iou": NMS_IOU,
        "class_mapping": ev.DATASET_TO_MODEL,
        "iou_thresholds_ap": IOU_THRESHOLDS,
        "matching": "same custom class-aware greedy one-to-one matching methodology as evaluate_dataset.py",
        "matching_iou_pr_f1": 0.50,
        "metric_aggregation": "PPE macro excluding the checkpoint Person class",
        "uncertainty": {
            "method": "paired image-cluster percentile bootstrap",
            "replicates": BOOTSTRAP_REPS,
            "seed": BOOTSTRAP_SEED,
            "paired_resampling": True,
        },
        "comparison_delta_640_minus_960": delta,
        "prior_use_disclosure": "Test split already had the 960px baseline evaluation and diagnostic review; 640px resolution was selected using validation data only. The fixed 640-vs-960 comparison was run on all test images after explicit authorization; no test-set tuning was performed.",
    }
    (OUT / "independent_validation_settings.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8")
    write_reports(run_config, summary_rows, class_rows, performance_rows,
                  uncertainty_rows, baseline)


def format_ci(row):
    return f"[{float(row['ci95_low']):+.3f}, {float(row['ci95_high']):+.3f}]"


def write_reports(config, summary_rows, class_rows, performance_rows,
                  uncertainty_rows, baseline):
    macro = {int(r["imgsz"]): r for r in summary_rows}
    classes = {(int(r["imgsz"]), r["class"]): r for r in class_rows}
    perf = {int(r["imgsz"]): r for r in performance_rows}
    uncertainty = {
        (r["scope"], r["class"], r["metric"]): r for r in uncertainty_rows
    }
    class_metrics = ["Helmet", "No Helmet", "Safety Vest", "No Vest"]
    lines = [
        "# Independent validation: 640px vs 960px PPE inference",
        "",
        "## Dataset separation and prior use",
        "",
        "Dataset inventory: train has 997 images/997 label files; validation has 119/119; test has 90/90. No additional labeled dataset or unused validation subset was found. All 119 validation images had already been used in prior resolution/threshold experiments, so they were not reused for this confirmation.",
        "",
        "The pre-existing 90-image test split (623 annotations) was used as the separate confirmation split. It had already been used for the original 960px baseline report and diagnostic error review; therefore this is an independent comparison with respect to the 640px selection (which used validation only), but not a previously untouched dataset. With explicit authorization, all 90 test images—including the two visually flagged images—were included unchanged. No labels were edited and no test-set tuning occurred.",
        "",
        f"Evaluated **{config['images']} images**, **{config['annotations']} total labeled objects**, including **{config['ppe_annotations']} PPE objects**. Counts by label: " + ", ".join(f"{k} {v}" for k, v in config["annotations_by_class"].items()) + ".",
        "",
        "## Controlled settings and method",
        "",
        f"- Same checkpoint SHA-256 for both sizes: `{config['model_sha256']}`; matches the recorded baseline: **{config['model_sha256'] == config['baseline_model_sha256']}**.",
        f"- Input sizes: 640 and 960; confidence cutoff **{CONF_REPORT}** for reported P/R/F1; inference candidate floor **{CONF_FLOOR}** for AP; NMS IoU **{NMS_IOU}**.",
        "- Same class mapping, CPU device, max detections (300), test image order, custom class-aware confidence-ordered one-to-one matching, and evaluator implementation.",
        "- Precision/recall/F1 use IoU 0.50. mAP50–95 averages AP over IoU 0.50 through 0.95 in 0.05 steps, using 101-point interpolation. Macro metrics exclude the checkpoint’s Person class, consistent with prior PPE reporting.",
        "- Inference time is CPU wall time around each Ultralytics prediction; one first-image warm-up per size was excluded from per-image timings.",
        "",
        "## Overall results",
        "",
        "| Input size | Precision | Recall | F1 | mAP50 | mAP50–95 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        r = macro[size]
        lines.append(
            f"| {size} | {r['precision']:.3f} | {r['recall']:.3f} | "
            f"{r['f1']:.3f} | {r['mAP50']:.3f} | {r['mAP50_95']:.3f} |"
        )
    lines += [
        "",
        "Paired bootstrap 95% confidence intervals for differences (**640 minus 960**) are image-cluster percentile intervals. They resample the same test images for both settings, preserving the paired comparison.",
        "These intervals quantify sampling uncertainty for this fixed split; they are not a formal hypothesis test and do not adjust for multiple class-wise comparisons.",
        "",
        "| Macro metric | Difference (640−960) | Paired bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for metric in ("precision", "recall", "f1", "AP50", "mAP50_95"):
        row = uncertainty[("PPE_macro", "PPE_macro", metric)]
        point_metric = "mAP50" if metric == "AP50" else metric
        lines.append(
            f"| {metric} | "
            f"{macro[640][point_metric] - macro[960][point_metric]:+.3f} | "
            f"{format_ci(row)} |"
        )
    lines += [
        "",
        "## Class-wise results",
        "",
        "| Class | GT | P 640 | R 640 | F1 640 | P 960 | R 960 | F1 960 | Δ recall (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cls in class_metrics:
        a, b = classes[(640, cls)], classes[(960, cls)]
        ci = uncertainty[("class", cls, "recall")]
        lines.append(
            f"| {cls} | {int(a['ground_truth'])} | {a['precision']:.3f} | "
            f"{a['recall']:.3f} | {a['f1']:.3f} | {b['precision']:.3f} | "
            f"{b['recall']:.3f} | {b['f1']:.3f} | "
            f"{a['recall']-b['recall']:+.3f} ({float(ci['ci95_low']):+.3f}, "
            f"{float(ci['ci95_high']):+.3f}) |"
        )
    lines += [
        "",
        "640px has higher point recall for each class in this test split. The paired class-wise recall intervals are above zero for Helmet, No Vest, and Safety Vest; the No Helmet interval crosses zero and is wide. Precision differences are mixed: Helmet and No Helmet are higher at 640, No Vest is nearly unchanged, and Safety Vest is slightly lower. No Helmet has only 24 labeled test objects, so its class estimate is less stable. Bootstrap intervals describe sampling uncertainty over images, not annotation error, distribution shift, or all future operating conditions.",
        "",
        "## Inference performance",
        "",
        "| Size | Mean ms/image | Median ms/image | P95 ms/image | Total inference seconds | Evaluation wall seconds* |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for size in SIZES:
        r = perf[size]
        lines.append(
            f"| {size} | {r['mean_inference_ms_per_image']:.1f} | "
            f"{r['median_inference_ms_per_image']:.1f} | "
            f"{r['p95_inference_ms_per_image']:.1f} | "
            f"{r['total_inference_seconds']:.2f} | "
            f"{r['evaluation_wall_seconds_including_warmup_matching_metrics']:.2f} |"
        )
    faster = 100 * (1 - perf[640]["mean_inference_ms_per_image"] /
                    perf[960]["mean_inference_ms_per_image"])
    lines += [
        "",
        f"*Evaluation wall time includes one warm-up, inference, matching, and point-metric calculation, but excludes model loading and bootstrap reporting. Mean CPU inference was {faster:.1f}% faster at 640px in this environment. Timing is hardware/runtime specific.",
        "",
        "## Limitations",
        "",
        "- Test is a distinct split from validation, but it is not pristine: 960px test baseline metrics and diagnostic error analysis were already available. The resolution choice (640px) came from validation; the test comparison was fixed in advance and not tuned.",
        "- Two test images had possible annotation disagreements flagged during visual review. They remain in this full-split scoring set because no label was adjudicated or changed; their uncertain labels may affect metrics.",
        "- The No Helmet class has only 24 objects; class-level uncertainty is substantial. Bootstrap resampling cannot compensate for missing diversity or systematic annotation issues.",
        "- This evaluates PPE checkpoint detections only, not the application’s dedicated person detector, worker association, site compliance, target hardware, or video performance.",
        "- The experiment compares the same checkpoint and fixed 0.25 threshold; it does not validate class-specific thresholds or retrained models.",
        "",
        "## Conclusion",
        "",
        "See [`final_recommendation.md`](final_recommendation.md) for the production-readiness decision. No experimental settings were deployed; weights, confidence cutoff, association geometry, person detection, and test annotations were not modified.",
        "",
    ]
    (OUT / "independent_validation_report.md").write_text(
        "\n".join(lines), encoding="utf-8")

    delta = {metric: macro[640][metric]-macro[960][metric]
             for metric in ("precision","recall","f1","mAP50","mAP50_95")}
    recall_ci = uncertainty[("PPE_macro","PPE_macro","recall")]
    rec_low, rec_high = float(recall_ci["ci95_low"]), float(recall_ci["ci95_high"])
    if rec_low > 0:
        inference = "The paired bootstrap interval for PPE macro recall is above zero."
        readiness = "The evidence supports considering a 640px configuration for a future controlled production change, subject to target-hardware testing and annotation review; it does not authorize deployment."
    elif rec_high < 0:
        inference = "The paired bootstrap interval for PPE macro recall is below zero."
        readiness = "The test comparison does not support changing to 640px for recall."
    else:
        inference = "The paired bootstrap interval for PPE macro recall includes zero, so the test split does not establish a reliable recall improvement."
        readiness = "640px is not sufficiently supported for a production change yet; keep 960px until additional independent data or a pre-specified acceptance criterion confirms the result."
    recommendation = [
        "# Final recommendation: 640px PPE configuration",
        "",
        "## Decision",
        "",
        readiness,
        "",
        "## Measured evidence",
        "",
        f"- Evaluated **{config['images']} test images** and **{config['annotations']} annotations** ({config['ppe_annotations']} PPE objects).",
        f"- Macro recall difference (640−960): **{delta['recall']:+.3f}**; paired image-bootstrap 95% CI **[{rec_low:+.3f}, {rec_high:+.3f}]**. {inference}",
        f"- Macro precision difference: {delta['precision']:+.3f}; F1: {delta['f1']:+.3f}; mAP50: {delta['mAP50']:+.3f}; mAP50–95: {delta['mAP50_95']:+.3f}.",
        f"- Mean CPU inference: {perf[640]['mean_inference_ms_per_image']:.1f} ms/image at 640 versus {perf[960]['mean_inference_ms_per_image']:.1f} ms/image at 960.",
        "- Class-wise point estimates and bootstrap intervals are reported in `independent_validation_report.md` and `class_wise_comparison.csv`; No Helmet has only 24 test annotations.",
        "",
        "## Conditions before any future change",
        "",
        "1. Do not deploy on this analysis alone; retain production weights, 0.25 threshold, person detector, and association geometry.",
        "2. Review the two flagged annotations under a documented protocol without changing labels in this experiment.",
        "3. If the confidence interval is inconclusive or class-specific trade-offs are unacceptable, acquire a larger independent/adjudicated validation set and predefine precision/recall acceptance margins.",
        "4. Verify inference latency on target hardware and run application-level regressions before any separately approved production release.",
        "",
        "No retraining is required to test inference size, and no weights or production behavior were changed.",
        "",
    ]
    (OUT / "final_recommendation.md").write_text(
        "\n".join(recommendation), encoding="utf-8")


if __name__ == "__main__":
    main()
