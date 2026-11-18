"""
Evaluate the dedicated COCO person detector on the project test split.

This is a *custom* evaluation: ground-truth boxes come from class id 3
(``person``) of ``dataset/data.yaml`` and predictions come from class 0
(``person``) of ``models/person_yolov8n.pt``.  It uses the same greedy
IoU matcher and 101-point interpolated AP as ``evaluate_model.py``.

It is NOT an official Ultralytics validation run, and the numbers must not be
reported as such.  Use ``yolo val`` with a matching dataset for official
metrics.

Run:

    .\\.venv\\Scripts\\python.exe evaluation\\evaluate_person.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "person_yolov8n.pt"
TEST_IMAGES = PROJECT_ROOT / "dataset" / "test" / "images"
TEST_LABELS = PROJECT_ROOT / "dataset" / "test" / "labels"
OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"
OUTPUT_PATH = OUTPUT_DIR / "person_evaluation_report.json"

IMAGE_SIZE = 640
CONF_THRESHOLD = 0.001          # low threshold for the AP curve
REPORT_CONF_THRESHOLD = 0.25    # same floor the Flask app uses
NMS_IOU_THRESHOLD = 0.7
PERSON_DATASET_CLASS = 3
PERSON_MODEL_CLASS = 0

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_person_ground_truth(label_path: Path) -> np.ndarray:
    """Return normalized xyxy boxes for dataset class 3 (person)."""
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {label_path}")

    boxes: list[list[float]] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.strip().split()
        if not values:
            continue
        if len(values) != 5:
            raise ValueError(f"Invalid annotation at {label_path}:{line_number}")
        if int(values[0]) != PERSON_DATASET_CLASS:
            continue
        x_center, y_center, width, height = map(float, values[1:])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid person box at {label_path}:{line_number}")
        boxes.append(
            [
                x_center - width / 2,
                y_center - height / 2,
                x_center + width / 2,
                y_center + height / 2,
            ]
        )
    return np.asarray(boxes, dtype=float).reshape(-1, 4)


def calculate_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros(0)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    box_area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-12)


def match_predictions(
    pred_boxes: np.ndarray,
    pred_conf: np.ndarray,
    gt_boxes: np.ndarray,
    iou_threshold: float,
) -> np.ndarray:
    """Greedy single-class matching, highest confidence first."""
    true_positive = np.zeros(len(pred_boxes), dtype=bool)
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return true_positive

    matched: set[int] = set()
    for pred_index in np.argsort(-pred_conf):
        ious = calculate_iou(pred_boxes[pred_index], gt_boxes)
        best_local = int(np.argmax(ious))
        if ious[best_local] >= iou_threshold and best_local not in matched:
            true_positive[pred_index] = True
            matched.add(best_local)
    return true_positive


def calculate_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    if len(recall) == 0:
        return 0.0
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([1.0], precision, [0.0]))
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    recall_points = np.linspace(0.0, 1.0, 101)
    interpolated = [
        float(np.max(precision[recall >= point])) if np.any(recall >= point) else 0.0
        for point in recall_points
    ]
    return float(np.mean(interpolated))


def evaluate_threshold(
    predictions: list[dict],
    total_gt: int,
    threshold_index: int,
) -> dict[str, float]:
    predictions = sorted(predictions, key=lambda p: p["confidence"], reverse=True)
    tp = np.asarray([p["tp"][threshold_index] for p in predictions], dtype=float)
    fp = 1.0 - tp
    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)
    recall_curve = cumulative_tp / max(total_gt, 1)
    precision_curve = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return {
        "ap": calculate_ap(recall_curve, precision_curve),
        "curve_recall": recall_curve,
        "curve_precision": precision_curve,
    }


def precision_recall_at_cutoff(predictions: list[dict], total_gt: int, threshold_index: int) -> tuple[float, float]:
    selected = [p for p in predictions if p["confidence"] >= REPORT_CONF_THRESHOLD]
    true_positives = sum(1 for p in selected if p["tp"][threshold_index])
    false_positives = len(selected) - true_positives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(total_gt, 1)
    return precision, recall


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Person model not found: {MODEL_PATH}")
    if not TEST_IMAGES.is_dir() or not TEST_LABELS.is_dir():
        raise FileNotFoundError("dataset/test images or labels are missing")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in TEST_IMAGES.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError("No test images found")

    print(f"Loading {MODEL_PATH.name} ...")
    load_started = time.perf_counter()
    model = YOLO(str(MODEL_PATH))
    model_load_ms = round((time.perf_counter() - load_started) * 1000.0, 1)
    print(f"Model loaded in {model_load_ms} ms")

    predictions: list[dict] = []
    total_gt = 0
    inference_times: list[float] = []
    images_with_gt = 0

    for index, image_path in enumerate(image_paths, 1):
        gt_boxes = load_person_ground_truth(TEST_LABELS / f"{image_path.stem}.txt")
        total_gt += len(gt_boxes)
        if len(gt_boxes):
            images_with_gt += 1

        result = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device="cpu",
            classes=[PERSON_MODEL_CLASS],
            verbose=False,
        )[0]

        height, width = result.orig_shape
        pred_boxes: list[list[float]] = []
        pred_conf: list[float] = []
        if result.boxes is not None and len(result.boxes):
            for bbox, confidence, class_id in zip(
                result.boxes.xyxy.tolist(),
                result.boxes.conf.tolist(),
                result.boxes.cls.tolist(),
            ):
                if int(class_id) != PERSON_MODEL_CLASS:
                    continue
                pred_boxes.append(
                    [bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height]
                )
                pred_conf.append(float(confidence))

        pred_boxes_array = np.asarray(pred_boxes, dtype=float).reshape(-1, 4)
        pred_conf_array = np.asarray(pred_conf, dtype=float)

        tp_matrix = np.zeros((len(pred_boxes_array), len(IOU_THRESHOLDS)), dtype=bool)
        for threshold_index, threshold in enumerate(IOU_THRESHOLDS):
            tp_matrix[:, threshold_index] = match_predictions(
                pred_boxes_array, pred_conf_array, gt_boxes, float(threshold)
            )
        for pred_index in range(len(pred_boxes_array)):
            predictions.append(
                {
                    "confidence": float(pred_conf_array[pred_index]),
                    "tp": tp_matrix[pred_index],
                }
            )

        inference_times.append(float(result.speed.get("inference", 0.0)))
        print(f"[{index}/{len(image_paths)}] {image_path.name}: gt={len(gt_boxes)} pred={len(pred_boxes_array)}")

    if total_gt == 0:
        raise RuntimeError("No person ground-truth boxes found in the test split")

    threshold_index_50 = int(np.argmin(np.abs(IOU_THRESHOLDS - 0.50)))
    precision, recall = precision_recall_at_cutoff(predictions, total_gt, threshold_index_50)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    aps: list[float] = []
    for threshold_index in range(len(IOU_THRESHOLDS)):
        aps.append(evaluate_threshold(predictions, total_gt, threshold_index)["ap"])

    selected_predictions = [p for p in predictions if p["confidence"] >= REPORT_CONF_THRESHOLD]
    selected_tp = sum(1 for p in selected_predictions if p["tp"][threshold_index_50])

    report = {
        "metric_type": "custom greedy IoU matching + 101-point interpolated AP (not official Ultralytics val)",
        "model": str(MODEL_PATH),
        "model_task": "COCO person detector (class 0 = person)",
        "ground_truth": "dataset/test/labels, class 3 = person",
        "images_evaluated": len(image_paths),
        "images_with_person_ground_truth": images_with_gt,
        "total_ground_truth_persons": total_gt,
        "confidence_threshold_for_AP": CONF_THRESHOLD,
        "confidence_threshold_for_precision_recall": REPORT_CONF_THRESHOLD,
        "iou_thresholds": IOU_THRESHOLDS.tolist(),
        "predictions_at_report_threshold": len(selected_predictions),
        "true_positives_at_report_threshold": selected_tp,
        "false_positives_at_report_threshold": len(selected_predictions) - selected_tp,
        "false_negatives_at_report_threshold": total_gt - selected_tp,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(float(f1), 4),
        "AP50": round(float(aps[threshold_index_50]), 4),
        "AP50_95": round(float(np.mean(aps)), 4),
        "model_load_ms": model_load_ms,
        "inference_ms_mean": round(float(np.mean(inference_times)), 1) if inference_times else 0.0,
        "inference_ms_min": round(float(np.min(inference_times)), 1) if inference_times else 0.0,
        "inference_ms_max": round(float(np.max(inference_times)), 1) if inference_times else 0.0,
    }

    OUTPUT_PATH.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\n" + "=" * 60)
    print("PERSON DETECTOR (custom evaluation, not Ultralytics val)")
    print("=" * 60)
    for key in (
        "images_evaluated",
        "total_ground_truth_persons",
        "precision",
        "recall",
        "f1",
        "AP50",
        "AP50_95",
        "model_load_ms",
        "inference_ms_mean",
    ):
        print(f"{key:28}: {report[key]}")
    print(f"\nReport saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
