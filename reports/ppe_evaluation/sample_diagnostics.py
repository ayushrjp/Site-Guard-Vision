"""Five-image smoke evaluation for evaluate_dataset.py (no production writes)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO

from evaluate_dataset import (
    DATASET_TO_MODEL, IMAGE_SIZE, INFERENCE_CONF, NAMES, NMS_IOU,
    ROOT, iou, parse_labels,
)

MODEL = ROOT / "models" / "yolo_ppe_core.pt"
IMAGE_DIR = ROOT / "dataset" / "test" / "images"
LABEL_DIR = ROOT / "dataset" / "test" / "labels"
OUT = Path(__file__).resolve().parent / "sample_diagnostics.json"
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def main():
    model = YOLO(str(MODEL))
    model_names = {int(k): v for k, v in model.names.items()}
    paths = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in EXTENSIONS)[:5]
    results = []
    for path in paths:
        from PIL import Image
        with Image.open(path) as im:
            width, height = im.size
        gt_boxes, gt_classes = parse_labels(LABEL_DIR / f"{path.stem}.txt", width, height)
        t0 = time.perf_counter()
        prediction = model.predict(
            source=str(path), conf=INFERENCE_CONF, iou=NMS_IOU, imgsz=IMAGE_SIZE,
            device="cpu", verbose=False, max_det=300, save=False,
        )[0]
        wall_ms = (time.perf_counter()-t0)*1000
        preds = []
        for box, conf, model_cls in zip(
            prediction.boxes.xyxy.tolist(), prediction.boxes.conf.tolist(), prediction.boxes.cls.tolist()
        ):
            dataset_cls = next((d for d, m in DATASET_TO_MODEL.items() if m == int(model_cls)), None)
            if dataset_cls is not None:
                preds.append({"class_id": dataset_cls, "class": NAMES[dataset_cls],
                              "model_class_id": int(model_cls), "model_class": model_names[int(model_cls)],
                              "confidence": float(conf), "bbox_xyxy": [round(float(v), 3) for v in box]})
        gts = [{"class_id": int(c), "class": NAMES[int(c)],
                "bbox_xyxy": [round(float(v), 3) for v in b]}
               for b, c in zip(gt_boxes, gt_classes)]
        candidates = []
        for pi, pred in enumerate(preds):
            for gi, gt in enumerate(gts):
                if pred["class_id"] == gt["class_id"]:
                    overlap = float(iou(np.asarray(pred["bbox_xyxy"]), np.asarray([gt["bbox_xyxy"]]))[0])
                    if overlap >= 0.50:
                        candidates.append((overlap, pred["confidence"], pi, gi))
        used_p, used_g, matches = set(), set(), []
        for overlap, _, pi, gi in sorted(candidates, reverse=True):
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            matches.append({"prediction_index": pi, "ground_truth_index": gi,
                            "class": preds[pi]["class"], "iou": overlap})
        results.append({
            "image": path.name, "image_size": [width, height],
            "ground_truth_count": len(gts), "ground_truth": gts,
            "prediction_count": len(preds), "predictions": preds,
            "matches_iou50_class_aware_one_to_one": matches,
            "unmatched_prediction_indices": [i for i in range(len(preds)) if i not in used_p],
            "unmatched_ground_truth_indices": [i for i in range(len(gts)) if i not in used_g],
            "wall_ms": round(wall_ms, 2),
            "inference_ms": float(prediction.speed.get("inference", 0)),
        })
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    report = {
        "model_path": str(MODEL.relative_to(ROOT)),
        "model_class_names": model_names,
        "dataset_to_model_class_mapping": DATASET_TO_MODEL,
        "device": "cpu", "cuda_available": torch.cuda.is_available(),
        "confidence_floor": INFERENCE_CONF, "production_cutoff_for_comparison": 0.25,
        "imgsz": IMAGE_SIZE, "nms_iou": NMS_IOU, "iou_match": 0.50,
        "images": results,
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"SAMPLE_REPORT={OUT}", flush=True)

if __name__ == "__main__":
    main()
