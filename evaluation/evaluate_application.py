"""
Application-level evaluation on the project test split.

This runs the real :func:`infer_engine.analyze_image` pipeline (PPE model +
COCO person detector + association + three-state assessment) and compares the
dashboard-level output with the test annotations.

Label semantics
---------------
A test image is treated as containing an *annotated* violation when its label
file contains class 1 (no-helmet) or class 2 (no-vest).  The absence of those
boxes is not proof of compliance, so the resulting numbers describe agreement
with recorded annotations, not ground-truth site compliance.

Everything here is custom application-level measurement, not Ultralytics
validation.  Annotated previews created during the run are deleted afterwards.

Run:

    .\\.venv\\Scripts\\python.exe evaluation\\evaluate_application.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infer_engine import analyze_image  # noqa: E402

TEST_IMAGES = PROJECT_ROOT / "dataset" / "test" / "images"
TEST_LABELS = PROJECT_ROOT / "dataset" / "test" / "labels"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PERSON_CLASS = 3
VIOLATION_CLASSES = {1, 2}  # no-helmet, no-vest
DEFAULT_CONF = 0.25
DEFAULT_IMAGE_SIZE = 640


def read_annotation(label_path: Path) -> tuple[int, bool]:
    if not label_path.is_file():
        raise FileNotFoundError(f"Missing annotation file: {label_path}")

    people = 0
    violation = False
    for line in label_path.read_text(encoding="utf-8").splitlines():
        values = line.strip().split()
        if not values:
            continue
        class_id = int(values[0])
        if class_id == PERSON_CLASS:
            people += 1
        elif class_id in VIOLATION_CLASSES:
            violation = True
    return people, violation


def safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def main(conf: float = DEFAULT_CONF, image_size: int = DEFAULT_IMAGE_SIZE, suffix: str = "") -> None:
    if not TEST_IMAGES.is_dir() or not TEST_LABELS.is_dir():
        raise FileNotFoundError("dataset/test images or labels are missing")

    output_name = "application_evaluation_report.json"
    if suffix:
        output_name = f"application_evaluation_report_{suffix}.json"
    output_path = RESULTS_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in TEST_IMAGES.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError("No test images found")

    rows: list[dict] = []
    statuses = Counter()
    total_ms: list[float] = []
    inference_ms: list[float] = []
    ppe_ms: list[float] = []
    person_ms: list[float] = []
    gt_people_total = 0
    pred_people_total = 0
    abs_error = 0
    count_bias = 0.0
    exact_count = 0
    tp = fp = fn = tn = 0
    green_on_violation = 0
    green_on_clean = 0
    failures = 0

    for index, image_path in enumerate(image_paths, 1):
        gt_people, gt_violation = read_annotation(TEST_LABELS / f"{image_path.stem}.txt")

        started = time.perf_counter()
        result = analyze_image(image_path, conf=conf, imgsz=image_size)
        wall_ms = (time.perf_counter() - started) * 1000.0

        if not result.get("success"):
            failures += 1
            rows.append(
                {
                    "image": image_path.name,
                    "error": result.get("error"),
                }
            )
            print(f"[{index}/{len(image_paths)}] FAILED {image_path.name}: {result.get('error')}")
            continue

        annotated = result.get("annotated_abs")
        if annotated:
            Path(annotated).unlink(missing_ok=True)

        status = result["assessment_status"]
        predicted_people = int(result["person_count"])
        predicted_violation = status == "potential_violation"

        statuses[status] += 1
        gt_people_total += gt_people
        pred_people_total += predicted_people
        abs_error += abs(predicted_people - gt_people)
        count_bias += predicted_people - gt_people
        if predicted_people == gt_people:
            exact_count += 1
        if gt_violation and predicted_violation:
            tp += 1
        elif gt_violation:
            fn += 1
        elif predicted_violation:
            fp += 1
        else:
            tn += 1
        if gt_violation and status == "no_violation_detected":
            green_on_violation += 1
        if not gt_violation and status == "no_violation_detected":
            green_on_clean += 1

        total_ms.append(result["total_ms"])
        inference_ms.append(result["inference_ms"])
        ppe_ms.append(result["ppe_inference_ms"])
        person_ms.append(result["person_inference_ms"])

        rows.append(
            {
                "image": image_path.name,
                "gt_people": gt_people,
                "predicted_people": predicted_people,
                "gt_annotated_violation": gt_violation,
                "status": status,
                "hardhats": result["hardhat_count"],
                "no_hardhats": result["no_hardhat_count"],
                "vests": result["vest_count"],
                "no_vests": result["no_vest_count"],
                "inference_ms": result["inference_ms"],
                "wall_ms": round(wall_ms, 1),
            }
        )
        print(
            f"[{index}/{len(image_paths)}] {image_path.name}: "
            f"people gt={gt_people} pred={predicted_people} "
            f"violation gt={gt_violation} pred={predicted_violation} status={status}"
        )

    evaluated = len(rows) - failures
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    report = {
        "metric_type": (
            "custom application-level agreement with annotations; "
            "not official Ultralytics metrics"
        ),
        "pipeline": "PPE model + models/person_yolov8n.pt + association + three-state assessment",
        "images_evaluated": evaluated,
        "images_failed": failures,
        "confidence_threshold": conf,
        "image_size": image_size,
        "label_semantics": (
            "Annotated violation = label file contains no-helmet (1) or no-vest (2). "
            "Absence of these boxes is not proof of compliance."
        ),
        "status_counts": dict(statuses),
        "annotated_violation_detection": {
            "true_positive": tp,
            "false_positive": fp,
            "false_negative": fn,
            "true_negative": tn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(safe_div(tp + tn, tp + tn + fp + fn), 4),
        },
        "safety": {
            "green_on_annotated_violation": green_on_violation,
            "green_on_clean_image": green_on_clean,
            "note": (
                "green_on_annotated_violation counts images where explicit no-PPE "
                "annotations exist but the dashboard returned no_violation_detected. "
                "This is the most important failure mode to inspect manually."
            ),
        },
        "person_counts": {
            "ground_truth": gt_people_total,
            "predicted": pred_people_total,
            "mean_absolute_error": round(safe_div(abs_error, evaluated), 2),
            "mean_signed_error": round(safe_div(count_bias, evaluated), 2),
            "exact_count_match_rate": round(safe_div(exact_count, evaluated), 4),
        },
        "timing_ms": {
            "mean_total": round(safe_div(sum(total_ms), len(total_ms)), 1),
            "mean_model_inference": round(safe_div(sum(inference_ms), len(inference_ms)), 1),
            "mean_ppe_inference": round(safe_div(sum(ppe_ms), len(ppe_ms)), 1),
            "mean_person_inference": round(safe_div(sum(person_ms), len(person_ms)), 1),
        },
        "per_image": rows,
    }

    output_path.write_text(json.dumps(report, indent=4), encoding="utf-8")

    print("\n" + "=" * 66)
    print("APPLICATION EVALUATION (custom, annotated-agreement)")
    print("=" * 66)
    print(f"images evaluated          : {evaluated} (failures: {failures})")
    print(f"status counts             : {dict(statuses)}")
    print(f"violation P / R / F1      : {precision:.4f} / {recall:.4f} / {f1:.4f}")
    print(f"violation TP FP FN TN     : {tp} {fp} {fn} {tn}")
    print(f"green on annotated risk   : {green_on_violation}")
    print(
        "person count MAE / bias   : "
        f"{report['person_counts']['mean_absolute_error']} / "
        f"{report['person_counts']['mean_signed_error']}"
    )
    print(f"mean inference ms         : {report['timing_ms']['mean_model_inference']}")
    print(f"\nReport saved: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()
    main(conf=args.conf, image_size=args.imgsz, suffix=args.suffix)
