"""Validation-only PPE inference-resolution and class-threshold experiments."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "reports" / "ppe_evaluation"
OUT = BASE / "controlled_experiments"
DATA = ROOT / "dataset"
MODEL_PATH = ROOT / "models" / "yolo_ppe_core.pt"
sys.path.insert(0, str(BASE))
import evaluate_dataset as ev  # noqa: E402

RESOLUTIONS = (640, 800, 960)
THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.501, 0.05)]
PPE = ev.PPE_CLASSES
NAME_BY_LOWER = {name.lower(): cls for cls, name in ev.NAMES.items()}
IOU_THRESHOLDS = ev.IOU_THRESHOLDS
CONF_FLOOR = 0.001
PRODUCTION_CONF = 0.25
NMS_IOU = 0.70


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def infer_resolution(model, samples, imgsz):
    # Warm up using the first validation sample; exclude it from measured times.
    if samples:
        model.predict(source=str(samples[0][0]), conf=CONF_FLOOR, iou=NMS_IOU,
                      imgsz=imgsz, device="cpu", verbose=False, max_det=300,
                      save=False)
    images = []
    elapsed = []
    for index, (path, gts) in enumerate(samples, 1):
        start = time.perf_counter()
        result = model.predict(
            source=str(path), conf=CONF_FLOOR, iou=NMS_IOU, imgsz=imgsz,
            device="cpu", verbose=False, max_det=300, save=False,
        )[0]
        elapsed.append((time.perf_counter() - start) * 1000)
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
        print(f"imgsz={imgsz} [{index}/{len(samples)}] {path.name}", flush=True)
    return images, elapsed


def get_gt_counts(images):
    counts = Counter()
    for image in images:
        for gt in image["ground_truth"]:
            counts[gt["class"]] += 1
    return counts


def add_matches(images):
    for image in images:
        for pred in image["predictions"]:
            pred["tp"] = []
    for threshold in IOU_THRESHOLDS:
        for image in images:
            ev.match_image(image["predictions"], image["ground_truth"], threshold)


def compute_resolution(images, elapsed, imgsz):
    gt_counts = get_gt_counts(images)
    add_matches(images)
    class_rows = []
    aps = {}
    for cls in PPE:
        metric = ev.metrics(images, gt_counts, cls, PRODUCTION_CONF, 0)
        cls_aps = [
            ev.class_ap(images, gt_counts, cls, j)
            for j in range(len(IOU_THRESHOLDS))
        ]
        aps[cls] = cls_aps
        class_rows.append({
            "imgsz": imgsz,
            "class_id": cls,
            "class": ev.NAMES[cls],
            "ground_truth": gt_counts[cls],
            **metric,
            "AP50": cls_aps[0],
            "mAP50_95": float(np.mean(cls_aps)),
        })
    macro = {
        key: float(np.mean([row[key] for row in class_rows]))
        for key in ("precision", "recall", "f1", "AP50", "mAP50_95")
    }
    total_ms = sum(elapsed)
    summary = {
        "imgsz": imgsz,
        "images": len(images),
        "ground_truth_ppe": sum(gt_counts[c] for c in PPE),
        "inference_confidence_floor": CONF_FLOOR,
        "reported_confidence_threshold": PRODUCTION_CONF,
        "nms_iou": NMS_IOU,
        "matching": "custom class-aware greedy IoU, one-to-one; IoU 0.50 for P/R/F1",
        **macro,
        "mean_inference_ms_per_image": float(np.mean(elapsed)),
        "median_inference_ms_per_image": float(np.median(elapsed)),
        "total_inference_seconds": total_ms / 1000,
    }
    return summary, class_rows, aps


def dump_cache(path, images):
    path.write_text(json.dumps(images, separators=(",", ":")), encoding="utf-8")


def load_cache(path):
    return json.loads(path.read_text(encoding="utf-8"))


def run_threshold_sweep(images):
    gt_counts = get_gt_counts(images)
    add_matches(images)
    rows = []
    for cls in PPE:
        for threshold in THRESHOLDS:
            metric = ev.metrics(images, gt_counts, cls, threshold, 0)
            rows.append({
                "class_id": cls,
                "class": ev.NAMES[cls],
                "confidence_threshold": threshold,
                "ground_truth": gt_counts[cls],
                **metric,
                "threshold_scope": "single-class cutoff; other classes do not affect class metric",
                "prediction_cache_imgsz": 960,
            })
    return rows


def draw_box(draw, box, color, label, width=3):
    xy = tuple(round(float(v)) for v in box)
    draw.rectangle(xy, outline=color, width=width)
    x1, y1 = xy[0], xy[1]
    draw.text((x1 + 2, max(0, y1 - 14)), label, fill=color,
              stroke_width=2, stroke_fill="white")


def create_annotation_review():
    flags = {
        "ppe_0186_jpg.rf.e2e0fe8a088ba9480c8c6d4c2ccece79.jpg",
        "ppe_0828_jpg.rf.71ebe13b3ae36071d34b5ee25d1549a7.jpg",
    }
    pred_file = BASE / "error_analysis_20261115" / "all_predictions.csv"
    with pred_file.open(encoding="utf-8", newline="") as f:
        pred_rows = list(csv.DictReader(f))
    lines = [
        "# Annotation review: flagged test images",
        "",
        "These two test images are reviewed only for annotation quality. They are excluded from all controlled experiment scoring and threshold/resolution decisions. No labels were changed. Bounding boxes below come from the existing diagnostic predictions at 960px, NMS IoU 0.70, candidate floor 0.001; the production review cutoff remains 0.25. Overlays show all ground-truth boxes (green) and predicted boxes with confidence ≥0.10 (red).",
        "",
        "Classes: Helmet, No Helmet, No Vest, Safety Vest. Image coordinates are pixels in the original 640×640 image. The visual review is not a relabeling adjudication.",
        "",
    ]
    for filename in sorted(flags):
        path = DATA / "test" / "images" / filename
        label_path = DATA / "test" / "labels" / f"{path.stem}.txt"
        with Image.open(path) as im:
            width, height = im.size
            canvas = im.convert("RGB")
        truths = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = line.split()
            cls = int(values[0])
            x, y, w, h = map(float, values[1:])
            box = [(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height]
            truths.append({"class": ev.NAMES[cls], "bbox": box})
        preds = [
            row for row in pred_rows
            if row["image"] == filename and float(row["confidence"]) >= 0.10
        ]
        draw = ImageDraw.Draw(canvas)
        for gt in truths:
            draw_box(draw, gt["bbox"], (0, 160, 0), f"GT {gt['class']}")
        for pred in preds:
            box = json.loads(pred["bbox_xyxy"])
            draw_box(draw, box, (220, 0, 0),
                     f"P {pred['class']} {float(pred['confidence']):.3f}", 2)
        overlay_name = f"{path.stem}_review_overlay.jpg"
        canvas.save(OUT / overlay_name, quality=92)
        lines += [
            f"## `{filename}`",
            "",
            f"- Original: `../../../dataset/test/images/{filename}`",
            f"- Ground truth: `{label_path.relative_to(ROOT).as_posix()}`",
            f"- Overlay: [`{overlay_name}`]({overlay_name})",
            "- Ground-truth classes: " + ", ".join(
                f"{name} × {sum(g['class'] == name for g in truths)}"
                for name in ev.NAMES.values()
                if any(g["class"] == name for g in truths)
            ),
            "- Red predictions shown (confidence ≥0.10): " + (
                "; ".join(
                    f"{r['class']} {float(r['confidence']):.3f}, box {json.loads(r['bbox_xyxy'])}"
                    for r in sorted(preds, key=lambda x: float(x["confidence"]), reverse=True)
                ) or "none"
            ),
            "",
            "| GT class | GT box (xyxy) | Best same-class prediction | Confidence | IoU |",
            "|---|---|---|---:|---:|",
        ]
        for gt in truths:
            same = [r for r in pred_rows
                    if r["image"] == filename and r["class"] == gt["class"]]
            def overlap(row):
                return float(ev.iou(np.asarray(gt["bbox"]),
                                    np.asarray([json.loads(row["bbox_xyxy"])]))[0])
            best = max(same, key=overlap, default=None)
            if best:
                lines.append(
                    f"| {gt['class']} | `{[round(v,1) for v in gt['bbox']]}` | "
                    f"`{[round(v,1) for v in json.loads(best['bbox_xyxy'])]}` | "
                    f"{float(best['confidence']):.6f} | {overlap(best):.3f} |"
                )
            else:
                lines.append(
                    f"| {gt['class']} | `{[round(v,1) for v in gt['bbox']]}` | none | — | — |"
                )
        lines.append("")
    lines += [
        "## Review conclusion",
        "",
        "The existing spot-check raised possible vest-state annotation disagreements in both images; this review records the original labels and model boxes, but does not adjudicate either as an error. In particular, model confidence is not evidence that an annotation is correct or incorrect. A trained annotator should compare each person’s clothing against the project labeling policy, document the decision independently, and leave the test labels immutable for these experiments.",
        "",
    ]
    (OUT / "annotation_review.md").write_text("\n".join(lines), encoding="utf-8")


def make_graphs(summary_rows, threshold_rows):
    plt.figure(figsize=(7, 5))
    plt.plot([r["imgsz"] for r in summary_rows],
             [r["recall"] for r in summary_rows], marker="o")
    plt.xticks(RESOLUTIONS); plt.xlabel("Inference resolution (imgsz)")
    plt.ylabel("PPE macro recall"); plt.ylim(0, 1)
    plt.title("Validation recall versus inference resolution"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "resolution_vs_recall.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot([r["imgsz"] for r in summary_rows],
             [r["mAP50_95"] for r in summary_rows], marker="o", label="mAP50–95")
    plt.plot([r["imgsz"] for r in summary_rows],
             [r["AP50"] for r in summary_rows], marker="s", label="mAP50")
    plt.xticks(RESOLUTIONS); plt.xlabel("Inference resolution (imgsz)")
    plt.ylabel("PPE macro AP"); plt.ylim(0, 1); plt.legend()
    plt.title("Validation mAP versus inference resolution"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "resolution_vs_map.png", dpi=160); plt.close()

    classes = [ev.NAMES[c] for c in PPE]
    x = np.arange(len(classes)); width = .34
    fig, ax = plt.subplots(figsize=(9, 5))
    with (OUT / "resolution_class_metrics.csv").open(encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))
    group_width = .8 / len(RESOLUTIONS)
    for i, resolution in enumerate(RESOLUTIONS):
        selected = {r["class"]: r for r in all_rows if int(r["imgsz"]) == resolution}
        offset = (i - (len(RESOLUTIONS)-1)/2) * group_width
        ax.bar(x + offset - group_width*.18,
               [float(selected[c]["precision"]) for c in classes],
               group_width*.36, label=f"Precision {resolution}")
        ax.bar(x + offset + group_width*.18,
               [float(selected[c]["recall"]) for c in classes],
               group_width*.36, label=f"Recall {resolution}")
    ax.set_xticks(x, classes); ax.set_ylim(0, 1); ax.set_ylabel("Score")
    ax.set_title("Class-wise precision and recall at conf=0.25")
    ax.legend(ncols=2, fontsize=8); fig.tight_layout()
    fig.savefig(OUT / "classwise_precision_recall.png", dpi=160); plt.close(fig)

    plt.figure(figsize=(9, 5))
    for cls in PPE:
        rows = [r for r in threshold_rows if int(r["class_id"]) == cls]
        plt.plot([r["confidence_threshold"] for r in rows],
                 [r["f1"] for r in rows], marker="o", label=ev.NAMES[cls])
    plt.axvline(PRODUCTION_CONF, color="black", linestyle="--",
                label="Production cutoff 0.25")
    plt.xlabel("Class-specific confidence threshold"); plt.ylabel("F1 at IoU 0.50")
    plt.xticks(THRESHOLDS); plt.ylim(0, 1)
    plt.title("Validation class-wise F1 versus threshold"); plt.grid(alpha=.3)
    plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "class_f1_vs_threshold.png", dpi=160); plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot([r["imgsz"] for r in summary_rows],
             [r["mean_inference_ms_per_image"] for r in summary_rows], marker="o")
    plt.xticks(RESOLUTIONS); plt.xlabel("Inference resolution (imgsz)")
    plt.ylabel("Mean CPU inference time (ms/image)")
    plt.title("Inference time versus resolution"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(OUT / "inference_time_vs_resolution.png", dpi=160)
    plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    create_annotation_review()
    samples = ev.load_split("valid")
    model = YOLO(str(MODEL_PATH))
    all_summary, all_class_rows = [], []
    caches = {}
    for resolution in RESOLUTIONS:
        cache_path = OUT / f"validation_predictions_{resolution}.json"
        if cache_path.exists():
            images = load_cache(cache_path)
            elapsed_path = OUT / f"inference_times_{resolution}.json"
            elapsed = json.loads(elapsed_path.read_text(encoding="utf-8"))
        else:
            images, elapsed = infer_resolution(model, samples, resolution)
            dump_cache(cache_path, images)
            (OUT / f"inference_times_{resolution}.json").write_text(
                json.dumps(elapsed), encoding="utf-8")
        caches[resolution] = images
        summary, class_rows, _ = compute_resolution(images, elapsed, resolution)
        all_summary.append(summary)
        all_class_rows.extend(class_rows)
        print(f"RESOLUTION SUMMARY {json.dumps(summary)}", flush=True)
    write_csv(OUT / "resolution_comparison.csv", all_summary, list(all_summary[0]))
    write_csv(OUT / "resolution_class_metrics.csv", all_class_rows,
              list(all_class_rows[0]))

    threshold_rows = run_threshold_sweep(caches[960])
    write_csv(OUT / "class_threshold_comparison.csv", threshold_rows,
              list(threshold_rows[0]))
    make_graphs(all_summary, threshold_rows)

    artifact = {
        "model_sha256": sha256(MODEL_PATH),
        "split": "validation",
        "validation_images": len(samples),
        "image_size_values": list(RESOLUTIONS),
        "thresholds": THRESHOLDS,
        "candidate_floor": CONF_FLOOR,
        "production_reference_cutoff": PRODUCTION_CONF,
        "nms_iou": NMS_IOU,
        "iou_thresholds_ap": IOU_THRESHOLDS,
        "iou_threshold_for_precision_recall_f1": 0.50,
        "class_mapping_dataset_to_model": ev.DATASET_TO_MODEL,
        "timing_device": "CPU; per-image wall-clock around Ultralytics predict, first sample warm-up excluded",
        "metric_definition": "same custom class-aware greedy IoU and 101-point interpolated AP methodology as reports/ppe_evaluation/evaluate_dataset.py; PPE macro excludes Person",
        "resolution_results": all_summary,
        "threshold_prediction_resolution": 960,
        "threshold_rows": len(threshold_rows),
    }
    (OUT / "experiment_settings.json").write_text(json.dumps(artifact, indent=2),
                                                    encoding="utf-8")
    print("CONTROLLED EXPERIMENTS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
