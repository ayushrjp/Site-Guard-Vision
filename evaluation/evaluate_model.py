
from pathlib import Path
import json
import numpy as np
from ultralytics import YOLO


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "yolo_ppe_core.pt"

TEST_IMAGES = PROJECT_ROOT / "dataset" / "test" / "images"
TEST_LABELS = PROJECT_ROOT / "dataset" / "test" / "labels"

OUTPUT_DIR = PROJECT_ROOT / "evaluation" / "results"

IMAGE_SIZE = 640

# Low confidence threshold for AP calculation
CONF_THRESHOLD = 0.001

# Confidence threshold for reported precision and recall
REPORT_CONF_THRESHOLD = 0.25

NMS_IOU_THRESHOLD = 0.7

IOU_THRESHOLDS = np.round(
    np.arange(0.50, 0.96, 0.05), 2
)

# Roboflow class ID -> Existing YOLO model class ID
DATASET_TO_MODEL = {
    0: 3,    # helmet -> Hardhat
    1: 8,    # no-helmet -> NO-Hardhat
    2: 10,   # no-vest -> NO-Safety Vest
    3: 11,   # person -> Person
    4: 13    # vest -> Safety Vest
}

CLASS_NAMES = {
    0: "helmet",
    1: "no-helmet",
    2: "no-vest",
    3: "person",
    4: "vest"
}

# Reverse mapping
MODEL_TO_DATASET = {
    model_id: dataset_id
    for dataset_id, model_id in DATASET_TO_MODEL.items()
}


# ============================================================
# 2. VALIDATE PROJECT PATHS
# ============================================================

def validate_paths():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    if not TEST_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Test images folder not found: {TEST_IMAGES}"
        )

    if not TEST_LABELS.is_dir():
        raise FileNotFoundError(
            f"Test labels folder not found: {TEST_LABELS}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


# ============================================================
# 3. LOAD YOLO GROUND-TRUTH ANNOTATIONS
# ============================================================

def load_ground_truth(label_path):

    if not label_path.is_file():
        raise FileNotFoundError(
            f"Missing annotation file: {label_path}"
        )

    boxes = []
    classes = []

    with open(label_path, "r") as file:

        for line_number, line in enumerate(file, start=1):

            values = line.strip().split()

            if not values:
                continue

            if len(values) != 5:
                raise ValueError(
                    f"Invalid annotation format: "
                    f"{label_path}, line {line_number}"
                )

            class_id = int(values[0])

            if class_id not in DATASET_TO_MODEL:
                raise ValueError(
                    f"Unknown class ID {class_id} "
                    f"in {label_path}"
                )

            x_center, y_center, width, height = map(
                float, values[1:]
            )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Invalid bounding box in {label_path}"
                )

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise ValueError(
                    f"Out-of-range normalized coordinates "
                    f"in {label_path}"
                )

            # YOLO xywh -> xyxy
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes.append([x1, y1, x2, y2])
            classes.append(class_id)

    return (
        np.asarray(boxes, dtype=float).reshape(-1, 4),
        np.asarray(classes, dtype=int)
    )


# ============================================================
# 4. CALCULATE INTERSECTION OVER UNION
# ============================================================

def calculate_iou(box, boxes):

    if len(boxes) == 0:
        return np.zeros(0)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])

    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    box_area = (
        max(0, box[2] - box[0])
        * max(0, box[3] - box[1])
    )

    boxes_area = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = box_area + boxes_area - intersection

    return intersection / np.maximum(union, 1e-12)


# ============================================================
# 5. MATCH PREDICTIONS WITH GROUND TRUTH
# ============================================================

def match_predictions(
    pred_boxes,
    pred_classes,
    pred_conf,
    gt_boxes,
    gt_classes,
    iou_threshold
):

    true_positive = np.zeros(
        len(pred_boxes),
        dtype=bool
    )

    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return true_positive

    # Highest confidence predictions first
    order = np.argsort(-pred_conf)

    matched_ground_truth = set()

    for pred_index in order:

        # Only compare objects of the same class
        candidates = np.where(
            gt_classes == pred_classes[pred_index]
        )[0]

        if len(candidates) == 0:
            continue

        ious = calculate_iou(
            pred_boxes[pred_index],
            gt_boxes[candidates]
        )

        best_local = int(np.argmax(ious))
        best_gt = int(candidates[best_local])

        if (
            ious[best_local] >= iou_threshold
            and best_gt not in matched_ground_truth
        ):

            true_positive[pred_index] = True

            matched_ground_truth.add(best_gt)

    return true_positive


# ============================================================
# 6. CORRECTED 101-POINT INTERPOLATED AVERAGE PRECISION
# ============================================================

def calculate_ap(recall, precision):

    if len(recall) == 0:
        return 0.0

    # Add boundary values
    recall = np.concatenate(
        ([0.0], recall, [1.0])
    )

    precision = np.concatenate(
        ([1.0], precision, [0.0])
    )

    # Construct monotonically decreasing precision envelope
    for i in range(
        len(precision) - 2,
        -1,
        -1
    ):

        precision[i] = max(
            precision[i],
            precision[i + 1]
        )

    # COCO-style 101-point interpolation
    recall_points = np.linspace(
        0.0,
        1.0,
        101
    )

    interpolated_precision = []

    for threshold in recall_points:

        valid = precision[recall >= threshold]

        if len(valid):
            interpolated_precision.append(
                np.max(valid)
            )
        else:
            interpolated_precision.append(0.0)

    return float(
        np.mean(interpolated_precision)
    )


# ============================================================
# 7. CALCULATE CLASS-WISE METRICS
# ============================================================

def evaluate_class(
    predictions,
    total_ground_truth,
    class_id,
    iou_threshold
):

    class_predictions = [
        p for p in predictions
        if p["class"] == class_id
    ]

    if total_ground_truth == 0:
        return None

    if not class_predictions:

        return {
            "precision": 0.0,
            "recall": 0.0,
            "ap": 0.0
        }

    class_predictions.sort(
        key=lambda p: p["confidence"],
        reverse=True
    )

    threshold_index = int(
        np.argmin(
            np.abs(IOU_THRESHOLDS - iou_threshold)
        )
    )

    tp = np.asarray([
        p["tp"][threshold_index]
        for p in class_predictions
    ], dtype=float)

    fp = 1.0 - tp

    cumulative_tp = np.cumsum(tp)
    cumulative_fp = np.cumsum(fp)

    recall_curve = (
        cumulative_tp / total_ground_truth
    )

    precision_curve = (
        cumulative_tp /
        np.maximum(
            cumulative_tp + cumulative_fp,
            1e-12
        )
    )

    # AP uses all predictions above CONF_THRESHOLD
    ap = calculate_ap(
        recall_curve,
        precision_curve
    )

    # Precision and recall at the configured cutoff
    selected = np.asarray([
        p["confidence"] >= REPORT_CONF_THRESHOLD
        for p in class_predictions
    ])

    selected_tp = int(
        np.sum(tp[selected])
    )

    selected_fp = (
        int(np.sum(selected)) - selected_tp
    )

    precision = (
        selected_tp /
        max(selected_tp + selected_fp, 1)
    )

    recall = (
        selected_tp / total_ground_truth
    )

    return {
        "precision": float(precision),
        "recall": float(recall),
        "ap": float(ap)
    }


# ============================================================
# 8. MAIN EVALUATION FUNCTION
# ============================================================

def main():

    validate_paths()

    print("\nLoading YOLO model...")

    model = YOLO(str(MODEL_PATH))

    print("Model loaded successfully.")
    print("Model classes:", model.names)

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    }

    image_paths = sorted([
        p for p in TEST_IMAGES.iterdir()
        if p.is_file()
        and p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        raise RuntimeError(
            "No test images found."
        )

    predictions = []

    ground_truth_counts = {
        class_id: 0
        for class_id in CLASS_NAMES
    }

    total_images = 0
    total_gt = 0

    print(
        f"\nFound {len(image_paths)} test images."
    )

    # --------------------------------------------------------
    # Run inference on each image
    # --------------------------------------------------------

    for image_index, image_path in enumerate(
        image_paths,
        start=1
    ):

        label_path = (
            TEST_LABELS /
            f"{image_path.stem}.txt"
        )

        gt_boxes, gt_classes = load_ground_truth(
            label_path
        )

        for class_id in gt_classes:

            ground_truth_counts[
                int(class_id)
            ] += 1

        total_gt += len(gt_classes)

        result = model.predict(
            source=str(image_path),
            conf=CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            imgsz=IMAGE_SIZE,
            device="cpu",
            verbose=False
        )[0]

        image_height, image_width = result.orig_shape

        pred_boxes = []
        pred_classes = []
        pred_conf = []

        if result.boxes is not None:

            for box in result.boxes:

                model_class = int(
                    box.cls.item()
                )

                # Ignore classes not covered by dataset
                if model_class not in MODEL_TO_DATASET:
                    continue

                dataset_class = MODEL_TO_DATASET[
                    model_class
                ]

                x1, y1, x2, y2 = (
                    box.xyxy[0].cpu().numpy()
                )

                # Normalize coordinates
                pred_boxes.append([
                    x1 / image_width,
                    y1 / image_height,
                    x2 / image_width,
                    y2 / image_height
                ])

                pred_classes.append(
                    dataset_class
                )

                pred_conf.append(
                    float(box.conf.item())
                )

        pred_boxes = np.asarray(
            pred_boxes,
            dtype=float
        ).reshape(-1, 4)

        pred_classes = np.asarray(
            pred_classes,
            dtype=int
        )

        pred_conf = np.asarray(
            pred_conf,
            dtype=float
        )

        # Match at every IoU threshold
        tp_matrix = np.zeros(
            (
                len(pred_boxes),
                len(IOU_THRESHOLDS)
            ),
            dtype=bool
        )

        for threshold_index, threshold in enumerate(
            IOU_THRESHOLDS
        ):

            tp_matrix[:, threshold_index] = (
                match_predictions(
                    pred_boxes,
                    pred_classes,
                    pred_conf,
                    gt_boxes,
                    gt_classes,
                    float(threshold)
                )
            )

        for i in range(len(pred_boxes)):

            predictions.append({
                "class": int(pred_classes[i]),
                "confidence": float(pred_conf[i]),
                "tp": tp_matrix[i]
            })

        total_images += 1

        print(
            f"[{image_index}/{len(image_paths)}] "
            f"Processed: {image_path.name}"
        )

    # --------------------------------------------------------
    # Calculate class-wise metrics
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SITEGUARDVISION MODEL EVALUATION")
    print("=" * 70)

    class_results = {}

    for class_id, class_name in CLASS_NAMES.items():

        results_at_50 = evaluate_class(
            predictions,
            ground_truth_counts[class_id],
            class_id,
            0.50
        )

        if results_at_50 is None:

            print(
                f"{class_name:15} | "
                "No ground-truth objects"
            )

            continue

        all_aps = []

        for threshold in IOU_THRESHOLDS:

            result = evaluate_class(
                predictions,
                ground_truth_counts[class_id],
                class_id,
                float(threshold)
            )

            if result is not None:
                all_aps.append(result["ap"])

        map50 = results_at_50["ap"]

        map50_95 = (
            float(np.mean(all_aps))
            if all_aps else 0.0
        )

        class_results[class_name] = {

            "ground_truth": int(
                ground_truth_counts[class_id]
            ),

            "precision": results_at_50["precision"],

            "recall": results_at_50["recall"],

            "mAP50": map50,

            "mAP50_95": map50_95
        }

        print(
            f"{class_name:15} | "
            f"P: {results_at_50['precision']:.4f} | "
            f"R: {results_at_50['recall']:.4f} | "
            f"AP50: {map50:.4f} | "
            f"AP50-95: {map50_95:.4f}"
        )

    # --------------------------------------------------------
    # Calculate overall macro metrics
    # --------------------------------------------------------

    if not class_results:

        raise RuntimeError(
            "No classes have ground-truth objects."
        )

    def macro(results: dict) -> dict:
        return {
            "precision": float(np.mean([r["precision"] for r in results.values()])),
            "recall": float(np.mean([r["recall"] for r in results.values()])),
            "mAP50": float(np.mean([r["mAP50"] for r in results.values()])),
            "mAP50_95": float(np.mean([r["mAP50_95"] for r in results.values()])),
        }

    overall = macro(class_results)
    ppe_only = macro(
        {name: values for name, values in class_results.items() if name != "person"}
    )

    # --------------------------------------------------------
    # Save evaluation report
    # --------------------------------------------------------

    report = {

        "model": str(MODEL_PATH),

        "metric_type": (
            "custom greedy IoU matching + 101-point interpolated AP; "
            "not an official Ultralytics validation run"
        ),

        "notes": [
            "Ground truth is dataset/test with Roboflow class ids remapped to the "
            "checkpoint's class ids (0->3, 1->8, 2->10, 3->11, 4->13).",
            "The person row measures the PPE checkpoint's own Person class and is "
            "expected to be near zero; people in the app come from "
            "models/person_yolov8n.pt (see evaluation/results/person_evaluation_report.json).",
            "Presence of no-helmet / no-vest boxes in a label file is treated as "
            "annotated evidence of a violation; their absence is not proof of compliance.",
        ],

        "images_evaluated": total_images,

        "total_ground_truth_objects": total_gt,

        "confidence_threshold_for_precision_recall":
            REPORT_CONF_THRESHOLD,

        "confidence_threshold_for_AP":
            CONF_THRESHOLD,

        "iou_thresholds": IOU_THRESHOLDS.tolist(),

        "class_mapping": DATASET_TO_MODEL,

        "class_results": class_results,

        "overall_macro_metrics": overall,

        "overall_macro_metrics_ppe_only": ppe_only
    }

    report_path = (
        OUTPUT_DIR /
        "evaluation_report.json"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Display final results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("OVERALL MACRO METRICS (all five classes, including Person)")
    print("=" * 70)

    for metric, value in overall.items():

        print(
            f"{metric:15}: "
            f"{value:.4f} ({value * 100:.2f}%)"
        )

    print("\n" + "=" * 70)
    print("OVERALL MACRO METRICS (PPE classes only)")
    print("=" * 70)

    for metric, value in ppe_only.items():

        print(
            f"{metric:15}: "
            f"{value:.4f} ({value * 100:.2f}%)"
        )

    print("\nImages evaluated:", total_images)

    print("Ground-truth objects:", total_gt)

    print("\nReport saved:", report_path)


# ============================================================
# 9. EXECUTION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
