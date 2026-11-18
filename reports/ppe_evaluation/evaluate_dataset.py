"""Read-only custom PPE checkpoint evaluation and confidence sweep.

Outputs stay under reports/ppe_evaluation. Dataset IDs are explicitly mapped
to the checkpoint's IDs. This evaluates the PPE checkpoint only, not the
worker-level application assessment or dedicated COCO person detector.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get("PPE_EVAL_OUT", str(Path(__file__).resolve().parent)))
MODEL_PATH = ROOT / "models" / "yolo_ppe_core.pt"
PERSON_MODEL_PATH = ROOT / "models" / "person_yolov8n.pt"
DATASET_YAML = ROOT / "dataset" / "data.yaml"
SPLITS = ("train", "valid", "test")
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NMS_IOU = 0.70
INFERENCE_CONF = 0.001
IMAGE_SIZE = 960
IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.951, 0.05)]
REPORT_THRESHOLDS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
DATASET_TO_MODEL = {0: 3, 1: 8, 2: 10, 3: 11, 4: 13}
NAMES = {0: "Helmet", 1: "No Helmet", 2: "No Vest", 3: "Person", 4: "Safety Vest"}
PPE_CLASSES = [0, 1, 2, 4]
ALL_CLASSES = list(NAMES)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_labels(path: Path, width: int, height: int):
    boxes, classes = [], []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        v = line.split()
        if len(v) != 5:
            raise ValueError(f"{path}:{line_no}: expected 5 YOLO fields")
        cls = int(v[0])
        x, y, w, h = map(float, v[1:])
        if cls not in DATASET_TO_MODEL or not (
            0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1
        ):
            raise ValueError(f"{path}:{line_no}: invalid class/normalized box")
        boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        classes.append(cls)
    return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(classes, dtype=int)


def iou(one, many):
    if len(many) == 0:
        return np.zeros(0)
    x1 = np.maximum(one[0], many[:, 0])
    y1 = np.maximum(one[1], many[:, 1])
    x2 = np.minimum(one[2], many[:, 2])
    y2 = np.minimum(one[3], many[:, 3])
    inter = np.maximum(0, x2-x1) * np.maximum(0, y2-y1)
    a = max(0, one[2]-one[0]) * max(0, one[3]-one[1])
    b = np.maximum(0, many[:, 2]-many[:, 0]) * np.maximum(0, many[:, 3]-many[:, 1])
    return inter / np.maximum(a+b-inter, 1e-12)


def ap101(recall, precision):
    if len(recall) == 0:
        return 0.0
    r = np.concatenate(([0.0], recall, [1.0]))
    p = np.concatenate(([1.0], precision, [0.0]))
    for i in range(len(p)-2, -1, -1):
        p[i] = max(p[i], p[i+1])
    return float(np.mean([np.max(p[r >= rp]) if np.any(r >= rp) else 0 for rp in np.linspace(0, 1, 101)]))


def match_image(preds, gts, threshold):
    """Confidence-ordered, class-aware, one-to-one greedy matching per image."""
    ordered = sorted(preds, key=lambda x: x["confidence"], reverse=True)
    used = set()
    for p in ordered:
        candidates = [i for i, g in enumerate(gts) if g["class"] == p["class"] and i not in used]
        if candidates:
            overlaps = iou(np.asarray(p["bbox"]), np.asarray([gts[i]["bbox"] for i in candidates]))
            best = int(np.argmax(overlaps))
            if overlaps[best] >= threshold:
                p["tp"].append(True)
                p["matched_gt"] = candidates[best]
                used.add(candidates[best])
                continue
        p["tp"].append(False)
        p["matched_gt"] = None
    return used


def dataset_audit():
    out = {}
    for split in SPLITS:
        image_dir = ROOT / "dataset" / split / "images"
        label_dir = ROOT / "dataset" / split / "labels"
        images = sorted(p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
        labels = sorted(label_dir.glob("*.txt"))
        class_counts = Counter()
        total_objects, malformed, corrupt = 0, [], []
        for image in images:
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                malformed.append(f"missing-label:{image.name}")
                continue
            try:
                with Image.open(image) as im:
                    im.verify()
                with Image.open(image) as im:
                    parse_labels(label, im.width, im.height)
                for line in label.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        class_counts[int(line.split()[0])] += 1
                        total_objects += 1
            except Exception as exc:
                malformed.append(f"{image.name}:{type(exc).__name__}:{exc}")
        image_stems = {p.stem for p in images}
        label_stems = {p.stem for p in labels}
        orphan_labels = sorted(label_stems-image_stems)
        out[split] = {
            "images": len(images), "label_files": len(labels), "objects": total_objects,
            "class_counts": {str(k): class_counts[k] for k in sorted(class_counts)},
            "missing_labels": sorted(image_stems-label_stems),
            "orphan_labels": orphan_labels, "malformed_or_invalid": malformed,
            "corrupt_images": corrupt,
        }
    return out


def load_split(split):
    image_dir = ROOT / "dataset" / split / "images"
    label_dir = ROOT / "dataset" / split / "labels"
    paths = sorted(p for p in image_dir.glob("*") if p.is_file() and p.suffix.lower() in EXTENSIONS)
    samples = []
    for path in paths:
        with Image.open(path) as im:
            w, h = im.size
        boxes, classes = parse_labels(label_dir / f"{path.stem}.txt", w, h)
        gts = [{"class": int(c), "bbox": b.tolist()} for b, c in zip(boxes, classes)]
        samples.append((path, gts))
    return samples


def run_split(model, split):
    samples = load_split(split)
    all_predictions, all_gts, confidence_values = [], Counter(), Counter()
    image_cache = []
    elapsed = []
    for i, (path, gts) in enumerate(samples, 1):
        t0 = time.perf_counter()
        result = model.predict(
            source=str(path), conf=INFERENCE_CONF, iou=NMS_IOU, imgsz=IMAGE_SIZE,
            device="cpu", verbose=False, max_det=300, save=False,
        )[0]
        elapsed.append((time.perf_counter()-t0)*1000)
        pred = []
        for box, conf, cls in zip(result.boxes.xyxy.tolist(), result.boxes.conf.tolist(), result.boxes.cls.tolist()):
            model_cls = int(cls)
            dataset_cls = next((ds for ds, mc in DATASET_TO_MODEL.items() if mc == model_cls), None)
            if dataset_cls is None:
                continue
            pred.append({"class": dataset_cls, "confidence": float(conf), "bbox": [float(x) for x in box], "tp": []})
            confidence_values[dataset_cls] += 1
        for gt in gts:
            all_gts[gt["class"]] += 1
        image_cache.append({"image": path.name, "predictions": pred, "ground_truth": gts})
        if split == "test":
            print(f"[{i}/{len(samples)}] {path.name}", flush=True)
    for iou_thr in IOU_THRESHOLDS:
        for image in image_cache:
            match_image(image["predictions"], image["ground_truth"], iou_thr)
    for image in image_cache:
        all_predictions.extend(image["predictions"])
    return image_cache, all_gts, confidence_values, elapsed


def metrics(images, gt_counts, cls, threshold, iou_index=0):
    selected = [p for im in images for p in im["predictions"] if p["class"] == cls and p["confidence"] >= threshold]
    selected.sort(key=lambda p: p["confidence"], reverse=True)
    tp = sum(bool(p["tp"][iou_index]) for p in selected)
    fp = len(selected)-tp
    fn = max(0, gt_counts[cls]-tp)
    return {
        "predictions": len(selected), "tp": tp, "fp": fp, "fn": fn,
        "precision": tp/(tp+fp) if tp+fp else 0.0,
        "recall": tp/gt_counts[cls] if gt_counts[cls] else 0.0,
        "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0,
    }


def class_ap(images, gt_counts, cls, iou_index):
    preds = sorted((p for im in images for p in im["predictions"] if p["class"] == cls), key=lambda p:p["confidence"], reverse=True)
    if not gt_counts[cls]:
        return None
    tp = np.asarray([int(p["tp"][iou_index]) for p in preds], dtype=float)
    fp = 1.0-tp
    rc = np.cumsum(tp)/gt_counts[cls]
    pr = np.cumsum(tp)/np.maximum(np.cumsum(tp)+np.cumsum(fp), 1e-12)
    return ap101(rc, pr)


def build_confusion(images, confidence=0.25, iou_threshold=0.50):
    """Detection confusion matrix; last row/column represent background."""
    size = len(ALL_CLASSES) + 1
    matrix = np.zeros((size, size), dtype=int)
    for image in images:
        gts = image["ground_truth"]
        preds = sorted(
            (p for p in image["predictions"] if p["confidence"] >= confidence),
            key=lambda p: p["confidence"], reverse=True,
        )
        used_gt, used_pred = set(), set()
        candidates = []
        for pi, pred in enumerate(preds):
            for gi, gt in enumerate(gts):
                overlap = float(iou(np.asarray(pred["bbox"]), np.asarray([gt["bbox"]]))[0])
                if overlap >= iou_threshold:
                    candidates.append((overlap, pred["confidence"], pi, gi))
        for _, _, pi, gi in sorted(candidates, reverse=True):
            if pi in used_pred or gi in used_gt:
                continue
            matrix[gts[gi]["class"], preds[pi]["class"]] += 1
            used_gt.add(gi)
            used_pred.add(pi)
        for gi, gt in enumerate(gts):
            if gi not in used_gt:
                matrix[gt["class"], len(ALL_CLASSES)] += 1
        for pi, pred in enumerate(preds):
            if pi not in used_pred:
                matrix[len(ALL_CLASSES), pred["class"]] += 1
    return matrix


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = dataset_audit()
    (OUT / "dataset_audit.json").write_text(json.dumps({
        "dataset_yaml": DATASET_YAML.read_text(encoding="utf-8"),
        "dataset_yaml_sha256": sha256(DATASET_YAML),
        "splits": audit, "class_names": NAMES,
        "annotation_format": "YOLO normalized xywh; one class + four coordinates per non-empty row",
        "evaluation_split": "test",
    }, indent=2), encoding="utf-8")
    baseline = {
        "ppe_model": str(MODEL_PATH.relative_to(ROOT)),
        "ppe_model_sha256": sha256(MODEL_PATH),
        "ppe_model_bytes": MODEL_PATH.stat().st_size,
        "person_model": str(PERSON_MODEL_PATH.relative_to(ROOT)),
        "person_model_sha256": sha256(PERSON_MODEL_PATH),
        "dataset_yaml": str(DATASET_YAML.relative_to(ROOT)),
        "dataset_yaml_sha256": sha256(DATASET_YAML),
        "production_ppe_confidence": 0.25,
        "production_person_confidence_floor": 0.25,
        "ppe_eval_confidence_floor": INFERENCE_CONF,
        "ppe_eval_imgsz": IMAGE_SIZE,
        "ppe_eval_nms_iou": NMS_IOU,
        "association": {"minimum_ioa": 0.30, "requires_center_inside": True, "head_bottom_ratio": 0.38, "torso_top_ratio": 0.16, "torso_bottom_ratio": 0.82, "torso_side_inset": 0.03, "one_detection_per_person_slot": True},
        "class_mapping_dataset_to_model": {str(k): v for k,v in DATASET_TO_MODEL.items()},
        "class_names_dataset": {str(k): v for k,v in NAMES.items()},
        "production_person_source": "models/person_yolov8n.pt; class 0; confidence max(requested, 0.25)",
        "tracking": "No temporal/video tracking implementation found in infer_engine.py; identities P1..Pn are per-image detection order.",
        "dependencies": {},
        "baseline_preserved": True,
    }
    for pkg in ["ultralytics", "torch", "numpy", "Pillow", "opencv-python", "matplotlib"]:
        try: baseline["dependencies"][pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError: baseline["dependencies"][pkg] = None
    (OUT / "baseline_configuration.json").write_text(json.dumps(baseline, indent=2), encoding="utf-8")

    model = YOLO(str(MODEL_PATH))
    test_images, gt_counts, conf_counts, elapsed = run_split(model, "test")
    total_gt = sum(gt_counts.values())
    ap_by_class = {}
    for cls in ALL_CLASSES:
        aps = [class_ap(test_images, gt_counts, cls, j) for j in range(len(IOU_THRESHOLDS))]
        aps = [v for v in aps if v is not None]
        ap_by_class[cls] = {
            "ground_truth": int(gt_counts[cls]),
            "ap50": aps[0] if aps else None,
            "ap50_95": float(np.mean(aps)) if aps else None,
        }

    sweep_rows, class_wise = [], []
    for cls in ALL_CLASSES:
        p25 = metrics(test_images, gt_counts, cls, 0.25)
        ap50, ap5095 = ap_by_class[cls]["ap50"], ap_by_class[cls]["ap50_95"]
        class_wise.append({
            "dataset_class_id": cls, "class": NAMES[cls], "ground_truth": int(gt_counts[cls]),
            "predictions_at_0.25": p25["predictions"], "tp_at_0.25": p25["tp"],
            "fp_at_0.25": p25["fp"], "fn_at_0.25": p25["fn"],
            "precision_at_0.25_iou50": p25["precision"], "recall_at_0.25_iou50": p25["recall"],
            "f1_at_0.25_iou50": p25["f1"], "AP50": ap50, "mAP50_95": ap5095,
        })
    for conf_thr in REPORT_THRESHOLDS:
        vals = {cls: metrics(test_images, gt_counts, cls, conf_thr) for cls in ALL_CLASSES}
        for cls, m in vals.items():
            sweep_rows.append({
                "confidence_threshold": conf_thr, "dataset_class_id": cls, "class": NAMES[cls],
                **m, "AP50": ap_by_class[cls]["ap50"], "mAP50_95": ap_by_class[cls]["ap50_95"],
            })
        for macro_label, classes in [("PPE_macro_excluding_Person", PPE_CLASSES), ("all_classes_macro", ALL_CLASSES)]:
            denominators = [c for c in classes if gt_counts[c]]
            sweep_rows.append({
                "confidence_threshold": conf_thr, "dataset_class_id": "", "class": macro_label,
                "predictions": sum(vals[c]["predictions"] for c in classes),
                "tp": sum(vals[c]["tp"] for c in classes),
                "fp": sum(vals[c]["fp"] for c in classes),
                "fn": sum(vals[c]["fn"] for c in classes),
                "precision": float(np.mean([vals[c]["precision"] for c in denominators])) if denominators else 0,
                "recall": float(np.mean([vals[c]["recall"] for c in denominators])) if denominators else 0,
                "f1": float(np.mean([vals[c]["f1"] for c in denominators])) if denominators else 0,
                "AP50": float(np.mean([ap_by_class[c]["ap50"] for c in denominators])),
                "mAP50_95": float(np.mean([ap_by_class[c]["ap50_95"] for c in denominators])),
            })
    fields = ["confidence_threshold","dataset_class_id","class","predictions","tp","fp","fn","precision","recall","f1","AP50","mAP50_95"]
    write_csv(OUT / "threshold_comparison.csv", sweep_rows, fields)
    write_csv(OUT / "class_wise_metrics.csv", class_wise, list(class_wise[0]))

    # Export every prediction below the unchanged production cutoff. These are
    # diagnostic-only and do not enter the production association/assessment.
    case_rows = []
    for image in test_images:
        raw = image["predictions"]
        for gi, gt in enumerate(image["ground_truth"]):
            same = [p for p in raw if p["class"] == gt["class"]]
            prod_best = max(
                (p for p in same if p["confidence"] >= 0.25),
                key=lambda p: iou(np.asarray(gt["bbox"]), np.asarray([p["bbox"]]))[0],
                default=None,
            )
            prod_iou = float(iou(np.asarray(gt["bbox"]), np.asarray([prod_best["bbox"]]))[0]) if prod_best else 0.0
            if prod_iou < 0.5:
                best = max(same, key=lambda p: iou(np.asarray(gt["bbox"]), np.asarray([p["bbox"]]))[0], default=None)
                category = "no_candidate" if best is None else ("below_threshold_candidate" if best["confidence"] < 0.25 else "localization_or_class_mismatch")
                case_rows.append({
                    "image": image["image"], "worker_id": "", "class_id": gt["class"], "class": NAMES[gt["class"]],
                    "confidence": f"{best['confidence']:.6f}" if best else "",
                    "bbox_xyxy": json.dumps(best["bbox"]) if best else "",
                    "threshold": 0.25,
                    "association_eligibility": "not evaluated: no dedicated person-detector results in detector-only evaluation",
                    "ground_truth_correspondence": json.dumps(gt["bbox"]) if best and iou(np.asarray(gt["bbox"]), np.asarray([best["bbox"]]))[0] >= 0.5 else "",
                    "ground_truth_bbox_xyxy": json.dumps(gt["bbox"]),
                    "best_candidate_iou": f"{iou(np.asarray(gt['bbox']), np.asarray([best['bbox']]))[0]:.6f}" if best else "",
                    "failure_category": category,
                    "evidence": "Ground-truth object not matched by an accepted same-class prediction at IoU 0.50.",
                })
        for p in raw:
            if p["confidence"] < 0.25:
                same = [g for g in image["ground_truth"] if g["class"] == p["class"]]
                best_gt = max(same, key=lambda g: iou(np.asarray(p["bbox"]), np.asarray([g["bbox"]]))[0], default=None)
                match_iou = float(iou(np.asarray(p["bbox"]), np.asarray([best_gt["bbox"]]))[0]) if best_gt else 0.0
                is_ppe_person = p["class"] == 3
                case_rows.append({
                    "image": image["image"], "worker_id": "", "class_id": p["class"], "class": NAMES[p["class"]],
                    "confidence": f"{p['confidence']:.6f}", "bbox_xyxy": json.dumps(p["bbox"]),
                    "threshold": 0.25,
                    "association_eligibility": (
                        "PPE-model Person class is not used when the dedicated application person detector is available"
                        if is_ppe_person else
                        "excluded before production association because confidence is below 0.25"
                    ),
                    "ground_truth_correspondence": json.dumps(best_gt["bbox"]) if best_gt and match_iou >= 0.5 else "",
                    "ground_truth_bbox_xyxy": json.dumps(best_gt["bbox"]) if best_gt else "",
                    "best_candidate_iou": f"{match_iou:.6f}",
                    "failure_category": (
                        "ppe_person_class_unused_in_application" if is_ppe_person else
                        "below_threshold_match" if match_iou >= 0.5 else "below_threshold_unmatched_candidate"
                    ),
                    "evidence": "Diagnostic-only prediction; not treated as a production detection or assessment input.",
                })
    write_csv(OUT / "low_confidence_cases.csv", case_rows, ["image","worker_id","class_id","class","confidence","bbox_xyxy","threshold","association_eligibility","ground_truth_correspondence","ground_truth_bbox_xyxy","best_candidate_iou","failure_category","evidence"])

    # Export raw audit predictions + labels so calculations are reproducible.
    metrics_out = {
        "metric_type": "custom class-aware greedy IoU matching; not official Ultralytics val",
        "split": "test", "images_evaluated": len(test_images), "ground_truth_objects": total_gt,
        "confidence_floor_for_candidate_cache": INFERENCE_CONF, "report_threshold": 0.25,
        "imgsz": IMAGE_SIZE, "nms_iou": NMS_IOU, "iou_thresholds": IOU_THRESHOLDS,
        "class_mapping": {str(k): v for k,v in DATASET_TO_MODEL.items()},
        "class_wise": {NAMES[c]: {"ground_truth": int(gt_counts[c]), **ap_by_class[c],
            **metrics(test_images,gt_counts,c,0.25)} for c in ALL_CLASSES},
        "macro_ppe_excluding_person": {
            key: float(np.mean([(
                metrics(test_images,gt_counts,c,0.25)[key] if key in {"precision","recall","f1"}
                else ap_by_class[c]["ap50"] if key=="AP50" else ap_by_class[c]["ap50_95"]
            ) for c in PPE_CLASSES]))
            for key in ["precision","recall","f1","AP50","mAP50_95"]
        },
        "mean_wall_ms_per_image": float(np.mean(elapsed)),
        "confusion_matrix_iou50_conf025": {
            "labels": [NAMES[c] for c in ALL_CLASSES] + ["Background"],
            "matrix": build_confusion(test_images).tolist(),
        },
        "thresholds": REPORT_THRESHOLDS,
        "limitations": ["Custom evaluator and greedy class-aware IoU matching.", "PPE-checkpoint Person class is not the application's dedicated person detector.", "Metrics against dataset labels measure annotation agreement, not site compliance."],
    }
    (OUT / "model_metrics.json").write_text(json.dumps(metrics_out, indent=2), encoding="utf-8")

    # Plots: threshold precision/recall, per-class metrics, raw confidence
    # distribution, and FP/FN counts by diagnostic threshold.
    ppe_rows = [r for r in sweep_rows if r["class"] == "PPE_macro_excluding_Person"]
    plt.figure(figsize=(7,5))
    plt.plot([r["recall"] for r in ppe_rows],[r["precision"] for r in ppe_rows], marker="o")
    for r in ppe_rows: plt.annotate(f'{r["confidence_threshold"]:.2f}',(r["recall"],r["precision"]))
    plt.xlabel("Macro recall"); plt.ylabel("Macro precision"); plt.title("PPE precision–recall by confidence threshold")
    plt.grid(True,alpha=.3); plt.tight_layout(); plt.savefig(OUT/"precision_recall.png",dpi=160); plt.close()
    plt.figure(figsize=(9,5))
    x=np.arange(len(class_wise)); w=.25
    plt.bar(x-w,[r["precision_at_0.25_iou50"] for r in class_wise],w,label="Precision")
    plt.bar(x,[r["recall_at_0.25_iou50"] for r in class_wise],w,label="Recall")
    plt.bar(x+w,[r["AP50"] for r in class_wise],w,label="AP50")
    plt.xticks(x,[r["class"] for r in class_wise],rotation=20,ha="right"); plt.ylim(0,1)
    plt.ylabel("Score"); plt.title("Class-wise PPE checkpoint performance at conf=0.25"); plt.legend()
    plt.tight_layout(); plt.savefig(OUT/"class_performance.png",dpi=160); plt.close()
    plt.figure(figsize=(8,5))
    bins=np.linspace(0,1,21)
    for cls in ALL_CLASSES:
        values=[p["confidence"] for im in test_images for p in im["predictions"] if p["class"]==cls]
        if values: plt.hist(values,bins=bins,alpha=.35,label=NAMES[cls])
    plt.axvline(.25,color="black",linestyle="--",label="production cutoff 0.25")
    plt.xlabel("Confidence"); plt.ylabel("Prediction count"); plt.title("PPE checkpoint prediction confidence distribution")
    plt.legend(); plt.tight_layout(); plt.savefig(OUT/"confidence_distribution.png",dpi=160); plt.close()
    fig, ax = plt.subplots(figsize=(8,5))
    thresholds = [r["confidence_threshold"] for r in ppe_rows]
    ax.plot(thresholds,[r["fp"] for r in ppe_rows],marker="o",label="False positives")
    ax.plot(thresholds,[r["fn"] for r in ppe_rows],marker="o",label="False negatives")
    ax.set_xlabel("Confidence threshold"); ax.set_ylabel("PPE-class FP / FN count")
    ax.grid(True,alpha=.3)
    ax2 = ax.twinx()
    ax2.plot(thresholds,[r["precision"] for r in ppe_rows],color="green",marker="s",label="Macro precision")
    ax2.plot(thresholds,[r["recall"] for r in ppe_rows],color="purple",marker="^",label="Macro recall")
    ax2.set_ylabel("Macro precision / recall")
    handles, labels = ax.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(handles+handles2,labels+labels2,loc="upper center",ncol=2)
    ax.set_title("PPE threshold trade-off (diagnostic only)",pad=42)
    fig.tight_layout(); fig.savefig(OUT/"threshold_comparison.png",dpi=160); plt.close(fig)
    confusion = np.asarray(metrics_out["confusion_matrix_iou50_conf025"]["matrix"])
    labels = metrics_out["confusion_matrix_iou50_conf025"]["labels"]
    fig, ax = plt.subplots(figsize=(8,7)); ax.imshow(confusion,cmap="Blues")
    ax.set_xticks(range(len(labels)),labels,rotation=35,ha="right")
    ax.set_yticks(range(len(labels)),labels)
    for y in range(confusion.shape[0]):
        for x in range(confusion.shape[1]):
            ax.text(x,y,str(confusion[y,x]),ha="center",va="center",fontsize=8)
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
    ax.set_title("Detection confusion matrix (IoU ≥ 0.50, conf ≥ 0.25)")
    fig.tight_layout(); fig.savefig(OUT/"confusion_matrix.png",dpi=160); plt.close(fig)
    print(json.dumps({"metrics": metrics_out, "low_confidence_and_error_rows": len(case_rows), "output_dir":str(OUT)},indent=2))


if __name__ == "__main__":
    main()
