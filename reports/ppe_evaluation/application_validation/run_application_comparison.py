"""Diagnostic-only full inference-engine comparison at 960 and 640 pixels."""
from __future__ import annotations

import csv
import json
import os
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import psutil

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "reports" / "ppe_evaluation" / "application_validation"
DATA = ROOT / "dataset"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "reports" / "ppe_evaluation"))
import evaluate_dataset as ev  # noqa: E402
import infer_engine as engine  # noqa: E402

SIZES = (960, 640)
REPEATS = 2
CONF = 0.25
PERSON_CONF = 0.25
PERSON_MATCH_IOU = 0.50
PPE_MATCH_IOU = 0.50
PPE_CATEGORY_TO_DATASET = {
    "hardhat": 0,
    "no_hardhat": 1,
    "no_vest": 2,
    "vest": 4,
}
VIOLATION_DATASET_CLASSES = {1, 2}
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def overlap(one, other):
    return float(ev.iou(np.asarray(one, dtype=float),
                        np.asarray([other], dtype=float))[0])


def read_ground_truth(path):
    with Image.open(path) as image:
        width, height = image.size
    labels = []
    label_path = DATA / "test" / "labels" / f"{path.stem}.txt"
    for line_no, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"{label_path}:{line_no}: expected 5 fields")
        cls = int(values[0])
        x, y, w, h = map(float, values[1:])
        labels.append({
            "class": cls,
            "bbox": [(x-w/2)*width, (y-h/2)*height,
                     (x+w/2)*width, (y+h/2)*height],
        })
    return labels


def ground_truth_worker_states(labels):
    people = [item for item in labels if item["class"] == 3]
    ppe = []
    for item in labels:
        category = {
            0: "hardhat", 1: "no_hardhat", 2: "no_vest", 4: "vest"
        }.get(item["class"])
        if category:
            ppe.append({
                "label": ev.NAMES[item["class"]],
                "category": category,
                "confidence": 1.0,
                "bbox": item["bbox"],
                "source": "ground_truth",
                "assigned_person": None,
                "association_score": None,
            })
    ppe = engine.associate_detections(
        [person["bbox"] for person in people], ppe)
    states = {}
    for person_index in range(len(people)):
        per_slot = {"helmet": set(), "vest": set()}
        for detection in ppe:
            if detection.get("assigned_person") != person_index + 1:
                continue
            slot = "helmet" if detection["category"] in {"hardhat", "no_hardhat"} else "vest"
            per_slot[slot].add(detection["category"])
        status = {}
        for slot, cats in per_slot.items():
            positive, negative = (
                ("hardhat", "no_hardhat") if slot == "helmet"
                else ("vest", "no_vest")
            )
            if positive in cats and negative in cats:
                status[slot] = "ambiguous"
            elif positive in cats:
                status[slot] = "present"
            elif negative in cats:
                status[slot] = "missing"
            else:
                status[slot] = "unknown"
        if "missing" in status.values():
            expected = "potential_violation"
        elif status == {"helmet": "present", "vest": "present"}:
            expected = "clear"
        else:
            expected = "unscorable"
        states[person_index + 1] = {
            "helmet": status["helmet"],
            "vest": status["vest"],
            "expected_status": expected,
        }
    return people, ppe, states


def match_person_boxes(predicted_people, gt_people, threshold=PERSON_MATCH_IOU):
    candidates = []
    for pi, person in enumerate(predicted_people):
        for gi, gt in enumerate(gt_people):
            iou_value = overlap(person["bbox"], gt["bbox"])
            if iou_value >= threshold:
                candidates.append((iou_value, person["confidence"], pi, gi))
    matches = {}
    used_pred, used_gt = set(), set()
    for iou_value, _conf, pi, gi in sorted(candidates, reverse=True):
        if pi in used_pred or gi in used_gt:
            continue
        matches[pi] = {"gt_person": gi + 1, "iou": iou_value}
        used_pred.add(pi)
        used_gt.add(gi)
    return matches


def score_ppe_detections(images):
    metric_images = []
    for image in images:
        predictions = []
        for det in image["result"]["detections"]:
            cls = PPE_CATEGORY_TO_DATASET.get(det["category"])
            if cls is None:
                continue
            predictions.append({
                "class": cls,
                "confidence": float(det["confidence"]),
                "bbox": [float(v) for v in det["bbox"]],
                "tp": [],
            })
        gt = [
            {"class": item["class"], "bbox": item["bbox"]}
            for item in image["labels"] if item["class"] in PPE_CATEGORY_TO_DATASET.values()
        ]
        metric_images.append({
            "image": image["image"],
            "predictions": predictions,
            "ground_truth": gt,
        })
    gt_counts = Counter(
        item["class"] for image in metric_images for item in image["ground_truth"]
    )
    for threshold in ev.IOU_THRESHOLDS:
        for image in metric_images:
            ev.match_image(image["predictions"], image["ground_truth"], threshold)
    metrics = {}
    for cls in PPE_CATEGORY_TO_DATASET.values():
        metrics[cls] = ev.metrics(metric_images, gt_counts, cls, CONF, 0)
    return metrics


def match_prediction_to_gt_objects(detections, gt_ppe):
    """Class-aware, one-to-one IoU matching for PPE object evidence."""
    candidates = []
    for di, det in enumerate(detections):
        for gi, gt in enumerate(gt_ppe):
            if det["category"] != gt["category"]:
                continue
            iou_value = overlap(det["bbox"], gt["bbox"])
            if iou_value >= PPE_MATCH_IOU:
                candidates.append((iou_value, det["confidence"], di, gi))
    matches = {}
    used_det, used_gt = set(), set()
    for iou_value, _conf, di, gi in sorted(candidates, reverse=True):
        if di in used_det or gi in used_gt:
            continue
        matches[di] = {"gt_index": gi, "iou": iou_value}
        used_det.add(di)
        used_gt.add(gi)
    return matches, used_gt


def additional_flags(images_by_size):
    """Mark 640 detections without a same-class IoU>=.5 counterpart at 960."""
    flags = {}
    old_by_image = {
        image["image"]: [
            d for d in image["result"]["detections"]
            if d["category"] in PPE_CATEGORY_TO_DATASET
        ]
        for image in images_by_size[960]
    }
    for image in images_by_size[640]:
        old = old_by_image[image["image"]]
        current = [
            d for d in image["result"]["detections"]
            if d["category"] in PPE_CATEGORY_TO_DATASET
        ]
        candidates = []
        for ci, det in enumerate(current):
            for oi, old_det in enumerate(old):
                if old_det["category"] != det["category"]:
                    continue
                ov = overlap(det["bbox"], old_det["bbox"])
                if ov >= 0.5:
                    candidates.append((ov, min(det["confidence"], old_det["confidence"]), ci, oi))
        matched_current, matched_old = set(), set()
        for _ov, _conf, ci, oi in sorted(candidates, reverse=True):
            if ci in matched_current or oi in matched_old:
                continue
            matched_current.add(ci)
            matched_old.add(oi)
        for ci, det in enumerate(current):
            flags[(image["image"], ci)] = ci not in matched_current
    return flags


def run_size(model_info, image_paths, size, repeat, process, cpu_samples):
    sample_path = image_paths[0]
    warm = engine.analyze_image(sample_path, conf=CONF, imgsz=size)
    if warm.get("annotated_abs"):
        Path(warm["annotated_abs"]).unlink(missing_ok=True)
    if not warm.get("success"):
        raise RuntimeError(f"Warm-up failed at imgsz={size}: {warm.get('error')}")

    association_samples = []
    original_associate = engine.associate_detections

    def timed_associate(*args, **kwargs):
        started = time.perf_counter()
        try:
            return original_associate(*args, **kwargs)
        finally:
            association_samples.append((time.perf_counter() - started) * 1000)

    engine.associate_detections = timed_associate
    records = []
    rss_start = process.memory_info().rss
    rss_max = rss_start
    cpu_start = process.cpu_times()
    cpu_wall_start = time.perf_counter()
    process.cpu_percent(None)
    system_cpu_start = psutil.cpu_percent(None)
    eval_started = time.perf_counter()
    try:
        for index, image_path in enumerate(image_paths, 1):
            labels = read_ground_truth(image_path)
            people_gt, ppe_gt, worker_states = ground_truth_worker_states(labels)
            image_started = time.perf_counter()
            result = engine.analyze_image(image_path, conf=CONF, imgsz=size)
            wall_ms = (time.perf_counter() - image_started) * 1000
            if not result.get("success"):
                raise RuntimeError(
                    f"Inference failed at {size}/{repeat}/{image_path.name}: {result.get('error')}"
                )
            preview_path = result.get("annotated_abs")
            if preview_path:
                Path(preview_path).unlink(missing_ok=True)
            persons = result["persons"]
            person_matches = match_person_boxes(persons, people_gt)
            detections = result["detections"]
            gt_match, gt_used = match_prediction_to_gt_objects(detections, ppe_gt)
            records.append({
                "image": image_path.name,
                "size": size,
                "repeat": repeat,
                "result": result,
                "labels": labels,
                "gt_people": people_gt,
                "gt_ppe": ppe_gt,
                "gt_worker_states": worker_states,
                "person_matches": person_matches,
                "gt_ppe_matches": gt_match,
                "gt_ppe_used": gt_used,
                "association_ms": association_samples[-1],
                "wall_ms": wall_ms,
                "image_index": index,
            })
            rss_now = process.memory_info().rss
            rss_max = max(rss_max, rss_now)
            if index % 15 == 0:
                print(f"app imgsz={size} repeat={repeat} [{index}/{len(image_paths)}]",
                      flush=True)
    finally:
        engine.associate_detections = original_associate
    wall_total = time.perf_counter() - eval_started
    cpu_end = process.cpu_times()
    process_cpu = (cpu_end.user + cpu_end.system) - (cpu_start.user + cpu_start.system)
    process_cpu_pct = 100 * process_cpu / max(wall_total, 1e-9)
    process_cpu_end_pct = process.cpu_percent(None)
    system_cpu_end = psutil.cpu_percent(None)
    cpu_samples.append({
        "imgsz": size, "repeat": repeat,
        "process_cpu_pct_core_equivalent": process_cpu_pct,
        "process_cpu_percent_sample": process_cpu_end_pct,
        "system_cpu_percent_start_sample": system_cpu_start,
        "system_cpu_percent_end_sample": system_cpu_end,
        "rss_start_mb": rss_start / 1024**2,
        "rss_max_between_images_mb": rss_max / 1024**2,
        "rss_end_mb": process.memory_info().rss / 1024**2,
        "total_process_wall_seconds": wall_total,
    })
    return records, {
        "size": size, "repeat": repeat, "wall_total_seconds": wall_total,
        "association_samples": association_samples,
        "rss_start": rss_start, "rss_max": rss_max,
        "cpu_seconds": process_cpu, "cpu_pct": process_cpu_pct,
        "process_cpu_sample": process_cpu_end_pct,
    }


def summarize_repetition(images):
    status_counts = Counter(i["result"]["assessment_status"] for i in images)
    app_ppe_metrics = score_ppe_detections(images)
    gt_people_total = sum(len(i["gt_people"]) for i in images)
    pred_people_total = sum(i["result"]["person_count"] for i in images)
    person_abs_errors = []
    person_exact = 0
    person_tp = person_fp = person_fn = 0
    event = Counter()
    worker_slots = Counter()
    worker_status = Counter()
    image_rows = []
    worker_rows = []
    association_rows = []
    unmatched_gt_counts = Counter()
    association_quality_counts = Counter()

    for image in images:
        result = image["result"]
        name = image["image"]
        size = image["size"]
        repeat = image["repeat"]
        gt_violation = any(
            item["class"] in VIOLATION_DATASET_CLASSES for item in image["labels"]
        )
        pred_event = result["assessment_status"] == "potential_violation"
        event["tp" if gt_violation and pred_event else
              "fn" if gt_violation else
              "fp" if pred_event else "tn"] += 1
        pred_people = result["persons"]
        match_by_pred = image["person_matches"]
        by_gt_worker = {
            match["gt_person"]: pred_people[pi]
            for pi, match in match_by_pred.items()
        }
        person_abs_errors.append(abs(len(pred_people) - len(image["gt_people"])))
        if len(pred_people) == len(image["gt_people"]):
            person_exact += 1
        matched_person_count = len(match_by_pred)
        person_tp += matched_person_count
        person_fp += len(pred_people) - matched_person_count
        person_fn += len(image["gt_people"]) - matched_person_count

        img_worker_correct = img_worker_incorrect = img_worker_unconfirmed = 0
        for gt_id, gt_state in image["gt_worker_states"].items():
            pred_person = by_gt_worker.get(gt_id)
            if pred_person is None:
                result_class = "missed_person"
                pred_helmet = pred_vest = pred_worker_status = "not_assessed"
            else:
                pred_helmet = pred_person["helmet"]["state"]
                pred_vest = pred_person["vest"]["state"]
                pred_worker_status = pred_person["status"]
                expected = gt_state["expected_status"]
                if expected == "unscorable":
                    result_class = "ambiguous_ground_truth"
                elif pred_worker_status == "uncertain":
                    result_class = "unconfirmed"
                    img_worker_unconfirmed += 1
                elif (
                    (expected == "potential_violation" and pred_worker_status == "potential_violation")
                    or (expected == "clear" and pred_worker_status == "clear")
                ):
                    result_class = "correct"
                    img_worker_correct += 1
                else:
                    result_class = "incorrect"
                    img_worker_incorrect += 1
                for slot, pred_state in (("helmet", pred_helmet), ("vest", pred_vest)):
                    gt_slot = gt_state[slot]
                    if gt_slot in {"present", "missing"}:
                        worker_slots[
                            "correct" if pred_state == gt_slot else
                            "unconfirmed" if pred_state == "unknown" else "incorrect"
                        ] += 1
            worker_status[result_class] += 1
            worker_rows.append({
                "row_type": "worker",
                "image": name,
                "imgsz": size,
                "repeat": repeat,
                "gt_worker_id": gt_id,
                "pred_worker_id": pred_person["id"] if pred_person else "",
                "person_iou": (
                    next((m["iou"] for pi, m in match_by_pred.items()
                          if pred_people[pi]["id"] == pred_person["id"]), "")
                    if pred_person else ""
                ),
                "gt_helmet": gt_state["helmet"],
                "pred_helmet": pred_helmet,
                "gt_vest": gt_state["vest"],
                "pred_vest": pred_vest,
                "gt_worker_status": gt_state["expected_status"],
                "pred_worker_status": pred_worker_status,
                "comparison": result_class,
            })
        image_rows.append({
            "row_type": "image_event",
            "image": name,
            "imgsz": size,
            "repeat": repeat,
            "gt_annotated_violation": gt_violation,
            "pred_assessment_status": result["assessment_status"],
            "pred_potential_violation_event": pred_event,
            "event_outcome": "TP" if gt_violation and pred_event else
                "FN" if gt_violation else "FP" if pred_event else "TN",
            "gt_workers": len(image["gt_people"]),
            "pred_workers": len(pred_people),
            "worker_correct": img_worker_correct,
            "worker_incorrect": img_worker_incorrect,
            "worker_unconfirmed": img_worker_unconfirmed,
            "assessment_reasons": json.dumps(result["assessment_reasons"]),
        })

        gt_person_by_id = {
            i + 1: person for i, person in enumerate(image["gt_people"])
        }
        mapped_pred_worker = {
            pred_people[pi]["id"]: match["gt_person"]
            for pi, match in match_by_pred.items()
        }
        gt_ppe = image["gt_ppe"]
        matched_pred_gt = image["gt_ppe_matches"]
        detections = [
            d for d in result["detections"]
            if d["category"] in PPE_CATEGORY_TO_DATASET
        ]
        for di, det in enumerate(detections):
            assigned = det.get("assigned_person")
            mapped_gt = mapped_pred_worker.get(assigned)
            gt_info = matched_pred_gt.get(di)
            gt_obj = gt_ppe[gt_info["gt_index"]] if gt_info else None
            gt_owner = gt_obj.get("assigned_person") if gt_obj else None
            if assigned is None or mapped_gt is None:
                quality = "unassigned"
            elif gt_info and gt_owner is None:
                quality = "ambiguous"
            elif gt_info and mapped_gt == gt_owner:
                quality = "correct"
            elif gt_info:
                quality = "incorrect"
            elif mapped_gt in image["gt_worker_states"]:
                expected_slot = (
                    "helmet" if det["category"] in {"hardhat", "no_hardhat"} else "vest"
                )
                gt_state = image["gt_worker_states"][mapped_gt][expected_slot]
                quality = "ambiguous" if gt_state in {"unknown", "ambiguous"} else "incorrect"
            else:
                quality = "ambiguous"
            classification_match = (
                "same_class_iou_match" if gt_info else
                "different_class_iou_overlap" if any(
                    det["category"] != gt["category"]
                    and overlap(det["bbox"], gt["bbox"]) >= PPE_MATCH_IOU
                    for gt in gt_ppe
                ) else "no_same_class_iou_match"
            )
            association_quality_counts[(det["category"], quality)] += 1
            if gt_info:
                gt_used = image["gt_ppe_used"]
                # Track per-GT matched state for later missed-GT accounting.
                gt_used.add(gt_info["gt_index"])
            association_rows.append({
                "row_type": "prediction",
                "image": name,
                "imgsz": size,
                "repeat": repeat,
                "prediction_index": di,
                "class": det["category"],
                "confidence": det["confidence"],
                "bbox_xyxy": json.dumps(det["bbox"]),
                "assigned_pred_worker": assigned if assigned is not None else "",
                "mapped_gt_worker": mapped_gt if mapped_gt is not None else "",
                "matched_gt_class": gt_obj["category"] if gt_obj else "",
                "matched_gt_worker": gt_owner if gt_owner is not None else "",
                "gt_iou": gt_info["iou"] if gt_info else "",
                "classification_match": classification_match,
                "association_quality": quality,
                "additional_at_640_vs_960": "",
            })
        for gt_index, gt in enumerate(gt_ppe):
            if gt_index not in image["gt_ppe_used"]:
                unmatched_gt_counts[(gt["category"], size)] += 1

    return {
        "status_counts": status_counts,
        "ppe_metrics": app_ppe_metrics,
        "gt_people": gt_people_total,
        "pred_people": pred_people_total,
        "person_mae": statistics.mean(person_abs_errors) if person_abs_errors else 0,
        "person_exact_rate": person_exact / len(images) if images else 0,
        "person_tp": person_tp,
        "person_fp": person_fp,
        "person_fn": person_fn,
        "person_precision": person_tp / max(person_tp + person_fp, 1),
        "person_recall": person_tp / max(person_tp + person_fn, 1),
        "person_f1": 2 * person_tp / max(2 * person_tp + person_fp + person_fn, 1),
        "events": event,
        "event_precision": event["tp"] / max(event["tp"] + event["fp"], 1),
        "event_recall": event["tp"] / max(event["tp"] + event["fn"], 1),
        "event_f1": (
            2 * event["tp"]
            / max(2 * event["tp"] + event["fp"] + event["fn"], 1)
        ),
        "worker_slots": worker_slots,
        "worker_status": worker_status,
        "image_rows": image_rows,
        "worker_rows": worker_rows,
        "association_rows": association_rows,
        "unmatched_gt_ppe": unmatched_gt_counts,
        "association_quality_counts": association_quality_counts,
        "images": len(images),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(
        path for path in (DATA / "test" / "images").iterdir()
        if path.is_file() and path.suffix.lower() in EXTS
    )
    if len(image_paths) != 90:
        raise RuntimeError(f"Expected 90 test images, found {len(image_paths)}")
    gt_counts = Counter(
        item["class"] for path in image_paths for item in read_ground_truth(path)
    )
    ppe_sha = __import__("hashlib").sha256(
        engine.PPE_MODEL_PATH.read_bytes()).hexdigest()
    person_sha = __import__("hashlib").sha256(
        engine.PERSON_MODEL_PATH.read_bytes()).hexdigest()
    initial_rendered = {p.name for p in engine.RENDERED_DIR.iterdir() if p.is_file()}

    engine.get_ppe_model()
    person_model = engine.get_person_model()
    if person_model is None:
        raise RuntimeError("Dedicated person detector unavailable; cannot validate full pipeline.")
    process = psutil.Process()
    run_images = {}
    run_perf = []
    cpu_samples = []
    run_summaries = {}
    order = [(1, 960), (1, 640), (2, 640), (2, 960)]
    try:
        for repeat, size in order:
            images, timing = run_size(
                {"ppe_sha256": ppe_sha, "person_sha256": person_sha},
                image_paths, size, repeat, process, cpu_samples)
            run_images[(repeat, size)] = images
            run_perf.append(timing)
            run_summaries[(repeat, size)] = summarize_repetition(images)
    finally:
        # Remove only previews created by this run, leaving prior files untouched.
        for p in engine.RENDERED_DIR.iterdir():
            if p.is_file() and p.name not in initial_rendered and p.name.startswith("det_"):
                p.unlink(missing_ok=True)

    first_by_size = {size: run_images[(1, size)] for size in SIZES}
    extra_flags = additional_flags(first_by_size)
    for row in run_summaries[(1, 640)]["association_rows"]:
        row["additional_at_640_vs_960"] = extra_flags.get(
            (row["image"], row["prediction_index"]), False)
    association_rows = []
    for size in SIZES:
        summary = run_summaries[(1, size)]
        association_rows.extend(summary["association_rows"])
        for (category, quality), count in summary["association_quality_counts"].items():
            association_rows.append({
                "row_type": "summary",
                "image": "",
                "imgsz": size,
                "repeat": 1,
                "prediction_index": "",
                "class": category,
                "confidence": "",
                "bbox_xyxy": "",
                "assigned_pred_worker": "",
                "mapped_gt_worker": "",
                "matched_gt_class": "",
                "matched_gt_worker": "",
                "gt_iou": "",
                "classification_match": "",
                "association_quality": quality,
                "additional_at_640_vs_960": "",
                "count": count,
            })
        for (category, sz), count in summary["unmatched_gt_ppe"].items():
            if sz == size:
                association_rows.append({
                    "row_type": "missed_ground_truth",
                    "image": "",
                    "imgsz": size,
                    "repeat": 1,
                    "prediction_index": "",
                    "class": category,
                    "confidence": "",
                    "bbox_xyxy": "",
                    "assigned_pred_worker": "",
                    "mapped_gt_worker": "",
                    "matched_gt_class": "",
                    "matched_gt_worker": "",
                    "gt_iou": "",
                    "classification_match": "",
                    "association_quality": "missed_gt_ppe",
                    "additional_at_640_vs_960": "",
                    "count": count,
                })
    for row in association_rows:
        row.setdefault("count", "")
    assoc_fields = [
        "row_type","image","imgsz","repeat","prediction_index","class","confidence","bbox_xyxy",
        "assigned_pred_worker","mapped_gt_worker","matched_gt_class",
        "matched_gt_worker","gt_iou","classification_match","association_quality",
        "additional_at_640_vs_960","count",
    ]
    write_csv(OUT / "association_validation.csv", association_rows, assoc_fields)

    safety_rows = []
    app_rows = []
    performance_rows = []
    for size in SIZES:
        first = run_summaries[(1, size)]
        second = run_summaries[(2, size)]
        safety_rows.extend(first["image_rows"])
        safety_rows.extend(first["worker_rows"])
        detections_by_class = {}
        class_rows = first["ppe_metrics"]
        for cls, metric in class_rows.items():
            detections_by_class[ev.NAMES[cls]] = metric
            app_rows.append({
                "row_type": "ppe_detection_class",
                "imgsz": size,
                "repeat": "1 (scoring)",
                "scope": ev.NAMES[cls],
                "images": first["images"],
                "ground_truth": gt_counts[cls],
                **metric,
            })
        for repeat_summary, repeat in ((first, 1), (second, 2)):
            # Store per-repeat dashboard outcome count rows for auditing.
            app_rows.append({
                "row_type": "application_repeat",
                "imgsz": size,
                "repeat": repeat,
                "scope": "application",
                "images": repeat_summary["images"],
                "ground_truth": "",
                "predictions": "",
                "tp": repeat_summary["events"]["tp"],
                "fp": repeat_summary["events"]["fp"],
                "fn": repeat_summary["events"]["fn"],
                "tn": repeat_summary["events"]["tn"],
                "safety_event_precision": repeat_summary["event_precision"],
                "safety_event_recall": repeat_summary["event_recall"],
                "safety_event_f1": repeat_summary["event_f1"],
                "false_safety_events": repeat_summary["events"]["fp"],
                "missed_safety_events": repeat_summary["events"]["fn"],
                "potential_violation_images": repeat_summary["status_counts"]["potential_violation"],
                "no_violation_detected_images": repeat_summary["status_counts"]["no_violation_detected"],
                "insufficient_evidence_images": repeat_summary["status_counts"]["insufficient_evidence"],
                "gt_people": repeat_summary["gt_people"],
                "predicted_people": repeat_summary["pred_people"],
                "person_tp": repeat_summary["person_tp"],
                "person_fp": repeat_summary["person_fp"],
                "person_fn": repeat_summary["person_fn"],
                "person_precision": repeat_summary["person_precision"],
                "person_recall": repeat_summary["person_recall"],
                "person_f1": repeat_summary["person_f1"],
                "person_count_mae": repeat_summary["person_mae"],
                "exact_person_count_rate": repeat_summary["person_exact_rate"],
                "worker_correct": repeat_summary["worker_status"]["correct"],
                "worker_incorrect": repeat_summary["worker_status"]["incorrect"],
                "worker_unconfirmed": repeat_summary["worker_status"]["unconfirmed"],
                "ambiguous_gt_workers": repeat_summary["worker_status"]["ambiguous_ground_truth"],
                "missed_person_workers": repeat_summary["worker_status"]["missed_person"],
                "ppe_slots_correct": repeat_summary["worker_slots"]["correct"],
                "ppe_slots_unconfirmed": repeat_summary["worker_slots"]["unconfirmed"],
                "ppe_slots_incorrect": repeat_summary["worker_slots"]["incorrect"],
                "associations_correct": sum(
                    n for (category, quality), n in repeat_summary["association_quality_counts"].items()
                    if quality == "correct"
                ),
                "associations_incorrect": sum(
                    n for (category, quality), n in repeat_summary["association_quality_counts"].items()
                    if quality == "incorrect"
                ),
                "associations_ambiguous": sum(
                    n for (category, quality), n in repeat_summary["association_quality_counts"].items()
                    if quality == "ambiguous"
                ),
                "associations_unassigned": sum(
                    n for (category, quality), n in repeat_summary["association_quality_counts"].items()
                    if quality == "unassigned"
                ),
            })
        per_image_values = defaultdict(list)
        per_repeat_means = defaultdict(list)
        for repeat in (1, 2):
            images = run_images[(repeat, size)]
            for image in images:
                r = image["result"]
                values = {
                    "ppe_model_inference": r["ppe_inference_ms"],
                    "person_detection_inference": r["person_inference_ms"],
                    "combined_model_inference": r["inference_ms"],
                    "association": image["association_ms"],
                    "complete_analyze_image": image["wall_ms"],
                }
                for metric, value in values.items():
                    per_image_values[metric].append(float(value))
            for metric in per_image_values:
                # Current repeat's values are isolated by its 90-image slice.
                repeat_vals = [
                    float(img["result"]["ppe_inference_ms"] if metric == "ppe_model_inference"
                          else img["result"]["person_inference_ms"] if metric == "person_detection_inference"
                          else img["result"]["inference_ms"] if metric == "combined_model_inference"
                          else img["association_ms"] if metric == "association"
                          else img["wall_ms"])
                    for img in images
                ]
                per_repeat_means[metric].append(float(np.mean(repeat_vals)))
        for metric, values in per_image_values.items():
            performance_rows.append({
                "imgsz": size,
                "stage": metric,
                "unit": "ms/image",
                "repeats": REPEATS,
                "images_per_repeat": len(image_paths),
                "observations": len(values),
                "mean_ms": float(np.mean(values)),
                "stddev_ms": float(np.std(values, ddof=1)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
                "mean_of_repeat_means_ms": float(np.mean(per_repeat_means[metric])),
                "stddev_of_repeat_means_ms": float(np.std(per_repeat_means[metric], ddof=1)),
                "rss_start_mb": float(np.mean([r["rss_start"] for r in run_perf if r["size"] == size]) / 1024**2),
                "rss_max_sampled_mb": float(np.max([r["rss_max"] for r in run_perf if r["size"] == size]) / 1024**2),
                "rss_max_percent_system_memory": float(
                    100 * np.max([r["rss_max"] for r in run_perf if r["size"] == size])
                    / psutil.virtual_memory().total
                ),
                "process_cpu_core_equivalent_percent": float(np.mean([r["cpu_pct"] for r in run_perf if r["size"] == size])),
                "process_cpu_sample_percent_mean": float(
                    np.mean([r["process_cpu_sample"] for r in run_perf if r["size"] == size])
                ),
                "system_cpu_start_sample_percent_mean": float(
                    np.mean([r["system_cpu_percent_start_sample"] for r in run_perf if r["size"] == size])
                ),
                "system_cpu_end_sample_percent_mean": float(
                    np.mean([r["system_cpu_percent_end_sample"] for r in run_perf if r["size"] == size])
                ),
            })
    safety_fields = sorted({key for row in safety_rows for key in row})
    write_csv(OUT / "safety_assessment_comparison.csv", safety_rows, safety_fields)
    app_fields = sorted({key for row in app_rows for key in row})
    write_csv(OUT / "application_comparison.csv", app_rows, app_fields)
    write_csv(OUT / "performance_comparison.csv", performance_rows,
              list(performance_rows[0]))
    (OUT / "application_run_metadata.json").write_text(json.dumps({
        "sizes": SIZES,
        "repeats": REPEATS,
        "confidence": CONF,
        "person_confidence_used": PERSON_CONF,
        "nms_iou": 0.70,
        "ppe_model_sha256": ppe_sha,
        "person_model_sha256": person_sha,
        "test_images": len(image_paths),
        "ground_truth_counts": {ev.NAMES[c]: int(gt_counts[c]) for c in ev.ALL_CLASSES},
        "association": {
            "min_intersection_over_detection": engine.MIN_ASSOCIATION_SCORE,
            "center_inside_required": True,
            "head_bottom_ratio": engine.HEAD_BOTTOM_RATIO,
            "head_side_inset": engine.HEAD_SIDE_INSET,
            "torso_top_ratio": engine.TORSO_TOP_RATIO,
            "torso_bottom_ratio": engine.TORSO_BOTTOM_RATIO,
            "torso_side_inset": engine.TORSO_SIDE_INSET,
            "duplicate_containment": engine.DUPLICATE_CONTAINMENT,
            "one_detection_per_person_slot": True,
        },
        "person_mapping_iou": PERSON_MATCH_IOU,
        "ppe_gt_matching_iou": PPE_MATCH_IOU,
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "system_memory_gb": round(psutil.virtual_memory().total / 1024**3, 2),
            "torch_threads": __import__("torch").get_num_threads(),
            "psutil": psutil.__version__,
        },
        "cpu_utilization_runs": cpu_samples,
        "run_order": order,
        "diagnostic_only": True,
        "annotation_note": "Two test images previously flagged for possible label disagreement are included; no annotations were modified.",
    }, indent=2), encoding="utf-8")
    print("APPLICATION COMPARISON COMPLETE", flush=True)


if __name__ == "__main__":
    main()
