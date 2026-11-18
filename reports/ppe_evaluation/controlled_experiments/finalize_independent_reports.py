"""Regenerate independent-validation reports from completed measured CSVs."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports" / "ppe_evaluation"
OUT = BASE / "controlled_experiments"
MODEL = ROOT / "models" / "yolo_ppe_core.pt"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(OUT))
import evaluate_dataset as ev  # noqa: E402
from run_independent_validation import write_reports, SIZES, CONF_REPORT, CONF_FLOOR, NMS_IOU, IOU_THRESHOLDS, BOOTSTRAP_REPS, BOOTSTRAP_SEED  # noqa: E402


def rows(path):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main():
    summary = rows(OUT / "resolution_final_comparison.csv")
    class_rows = rows(OUT / "class_wise_comparison.csv")
    performance = rows(OUT / "inference_performance.csv")
    uncertainty = rows(OUT / "paired_bootstrap_uncertainty.csv")
    float_fields = {
        "precision", "recall", "f1", "mAP50", "mAP50_95",
        "delta_640_minus_960", "ci95_low", "ci95_high",
        "mean_inference_ms_per_image", "median_inference_ms_per_image",
        "p95_inference_ms_per_image", "total_inference_seconds",
        "end_to_end_inference_loop_seconds",
        "evaluation_wall_seconds_including_warmup_matching_metrics",
        "AP50", "mAP50_95",
    }
    int_fields = {
        "imgsz", "images", "ground_truth_ppe_objects", "ground_truth",
        "predictions", "tp", "fp", "fn", "bootstrap_replicates",
        "replicates_with_class_gt",
    }
    def convert(records):
        for row in records:
            for key in list(row):
                if row[key] == "":
                    continue
                if key in float_fields:
                    row[key] = float(row[key])
                elif key in int_fields:
                    row[key] = int(row[key])
        return records
    summary = convert(summary)
    class_rows = convert(class_rows)
    performance = convert(performance)
    uncertainty = convert(uncertainty)

    samples = ev.load_split("test")
    counts = Counter(
        gt["class"] for _path, labels in samples for gt in labels
    )
    baseline_cfg = json.loads(
        (BASE / "baseline_configuration.json").read_text(encoding="utf-8"))
    settings = {
        "model_sha256": hashlib.sha256(MODEL.read_bytes()).hexdigest(),
        "baseline_model_sha256": baseline_cfg["ppe_model_sha256"],
        "split": "test",
        "images": len(samples),
        "annotations": sum(counts.values()),
        "annotations_by_class": {ev.NAMES[c]: counts[c] for c in ev.ALL_CLASSES},
        "ppe_annotations": sum(counts[c] for c in ev.PPE_CLASSES),
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
        "comparison_delta_640_minus_960": {
            key: next(r for r in summary if r["imgsz"] == 640)[key]
                 - next(r for r in summary if r["imgsz"] == 960)[key]
            for key in ("precision", "recall", "f1", "mAP50", "mAP50_95")
        },
        "prior_use_disclosure": "Test split already had the 960px baseline evaluation and diagnostic review; 640px resolution was selected using validation data only. The fixed 640-vs-960 comparison was run on all test images after explicit authorization; no test-set tuning was performed.",
    }
    (OUT / "independent_validation_settings.json").write_text(
        json.dumps(settings, indent=2), encoding="utf-8")
    baseline = json.loads(
        (BASE / "run_20261115" / "model_metrics.json").read_text(encoding="utf-8"))
    write_reports(settings, summary, class_rows, performance, uncertainty, baseline)
    print("Independent validation reports regenerated from measured outputs.")


if __name__ == "__main__":
    main()
