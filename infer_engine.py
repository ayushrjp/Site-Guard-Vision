"""
SiteGuardVision inference engine.

Two models, kept deliberately separate:

* PPE model  - models/yolo_ppe_core.pt (14 classes: hard hat, vest, NO-* etc.)
* Person model - models/person_yolov8n.pt (YOLOv8n COCO, class 0 = person)

The PPE checkpoint's own ``Person`` class produced no predictions at any
confidence in diagnostics, so people are counted with the dedicated person
detector.  The PPE checkpoint is never modified.  Neither model is allowed to
claim the other model's detections.

Every detection carries a ``source`` field and the assessment is evidence based:
missing PPE is a potential violation, absent PPE evidence is uncertainty, and
zero people is never treated as compliance.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from ultralytics import YOLO

import runtime_config

ROOT_DIR = Path(__file__).resolve().parent
MODEL_DIR = ROOT_DIR / "models"
PPE_MODEL_PATH = MODEL_DIR / "yolo_ppe_core.pt"
PERSON_MODEL_PATH = MODEL_DIR / "person_yolov8n.pt"
LEGACY_PERSON_MODEL_PATH = ROOT_DIR / "yolov8n.pt"
# Writable output location; defaults to static/rendered locally and moves to
# SITEGUARD_RENDERED_DIR (persistent disk) on a container host.
RENDERED_DIR = runtime_config.RENDERED_DIR
RENDERED_DIR.mkdir(parents=True, exist_ok=True)

MIN_CONF = 0.05
MAX_CONF = 0.95
DEFAULT_CONF = 0.25
DEFAULT_IMGSZ = 960
ALLOWED_IMGSZ = {640, 960, 1280}

# The COCO person detector is not driven below this floor: small false
# positives from a 0.10 slider setting would corrupt every downstream count.
PERSON_CONF_FLOOR = 0.25

# Geometry used to attach PPE boxes to the worker they most likely belong to.
HEAD_BOTTOM_RATIO = 0.38
HEAD_SIDE_INSET = 0.05
TORSO_TOP_RATIO = 0.16
TORSO_BOTTOM_RATIO = 0.82
TORSO_SIDE_INSET = 0.03
MIN_ASSOCIATION_SCORE = 0.30

# Two boxes of the same PPE category are duplicate detections of one physical
# object when the smaller box is almost entirely inside the larger one.  The
# lower-confidence copy is ignored so it cannot take a slot from the confident
# detection.  Opposite categories (Hardhat vs NO-Hardhat) are never merged.
DUPLICATE_CONTAINMENT = 0.85

PPE_PERSON_LABEL = "Person"

logger = logging.getLogger(__name__)

_predict_lock = threading.RLock()
_load_lock = threading.RLock()
_ppe_model: YOLO | None = None
_person_model: YOLO | None = None
_ppe_load_ms: float | None = None
_person_load_ms: float | None = None
_person_model_checked = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_yolo(path: Path) -> tuple[YOLO, float]:
    started = time.perf_counter()
    model = YOLO(str(path))
    elapsed = (time.perf_counter() - started) * 1000.0
    return model, round(elapsed, 1)


def get_ppe_model() -> YOLO:
    global _ppe_model, _ppe_load_ms
    with _load_lock:
        if _ppe_model is None:
            if not PPE_MODEL_PATH.is_file():
                raise FileNotFoundError(f"PPE model not found: {PPE_MODEL_PATH}")
            _ppe_model, _ppe_load_ms = _load_yolo(PPE_MODEL_PATH)
            logger.info("PPE model loaded in %.1f ms", _ppe_load_ms)
    return _ppe_model


def get_person_model() -> YOLO | None:
    """Return the dedicated COCO person detector, or None when unavailable."""
    global _person_model, _person_load_ms, _person_model_checked

    with _load_lock:
        if _person_model_checked:
            return _person_model

        _person_model_checked = True
        candidate = (
            PERSON_MODEL_PATH
            if PERSON_MODEL_PATH.is_file()
            else LEGACY_PERSON_MODEL_PATH
        )

        if not candidate.is_file():
            logger.warning("Person detector not found; falling back to PPE Person class.")
            return None

        try:
            _person_model, _person_load_ms = _load_yolo(candidate)
            logger.info("Person model loaded in %.1f ms", _person_load_ms)
        except Exception:  # noqa: BLE001 - degraded mode is better than a crash
            logger.exception("Could not load the person detector")
            _person_model = None

    return _person_model


def model_status() -> dict[str, Any]:
    """Small, side-effect-light description of the configured models."""
    person = get_person_model()
    return {
        "ppe_model": PPE_MODEL_PATH.name,
        "ppe_load_ms": _ppe_load_ms,
        "person_model": (
            PERSON_MODEL_PATH.name
            if PERSON_MODEL_PATH.is_file()
            else LEGACY_PERSON_MODEL_PATH.name
        ),
        "person_load_ms": _person_load_ms,
        "person_model_available": person is not None,
    }


def prediction_lock() -> threading.RLock:
    """Shared lock so image and video inference never hit the models at once.

    The image pipeline already uses this lock internally.  Video analysis calls
    this accessor so it can serialise short per-frame predictions without
    duplicating the inference engine.
    """
    return _predict_lock


# ---------------------------------------------------------------------------
# Label and geometry helpers (pure functions, unit-testable)
# ---------------------------------------------------------------------------


def classify_label(label: str) -> str:
    normalized = "".join(ch.lower() for ch in label if ch.isalnum())
    mapping = {
        "person": "person",
        "hardhat": "hardhat",
        "helmet": "hardhat",
        "nohardhat": "no_hardhat",
        "nohelmet": "no_hardhat",
        "safetyvest": "vest",
        "vest": "vest",
        "nosafetyvest": "no_vest",
        "novest": "no_vest",
    }
    return mapping.get(normalized, "other")


def _area(box: Iterable[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _intersection(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0.0, min(ay2, by2) - max(ay1, by1))
    return width * height


def _intersection_over_detection(det: Iterable[float], region: Iterable[float]) -> float:
    area = _area(det)
    if area <= 0:
        return 0.0
    return _intersection(det, region) / area


def _center(box: Iterable[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _center_inside(box: Iterable[float], region: Iterable[float]) -> bool:
    cx, cy = _center(box)
    x1, y1, x2, y2 = region
    return x1 <= cx <= x2 and y1 <= cy <= y2


def _head_region(person_box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1
    return [
        x1 + width * HEAD_SIDE_INSET,
        y1,
        x2 - width * HEAD_SIDE_INSET,
        y1 + height * HEAD_BOTTOM_RATIO,
    ]


def _torso_region(person_box: Iterable[float]) -> list[float]:
    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1
    return [
        x1 + width * TORSO_SIDE_INSET,
        y1 + height * TORSO_TOP_RATIO,
        x2 - width * TORSO_SIDE_INSET,
        y1 + height * TORSO_BOTTOM_RATIO,
    ]


def _duplicate_detection_indices(detections: list[dict[str, Any]]) -> set[int]:
    """
    Return the indices of lower-confidence duplicate PPE boxes.

    A duplicate must have the same category as the kept box and be at least
    ``DUPLICATE_CONTAINMENT`` covered by it.  Opposite categories are kept:
    a NO-Hardhat box overlapping a Hardhat box is contradicting evidence, not a
    duplicate, and must still be able to flag a violation.
    """
    order = sorted(
        range(len(detections)),
        key=lambda index: detections[index]["confidence"],
        reverse=True,
    )
    duplicates: set[int] = set()

    for position, high_index in enumerate(order):
        if high_index in duplicates:
            continue
        high = detections[high_index]
        category = high.get("category")
        if category not in {"hardhat", "no_hardhat", "vest", "no_vest"}:
            continue
        high_box = high["bbox"]
        high_area = _area(high_box)
        if high_area <= 0:
            continue

        for low_index in order[position + 1:]:
            if low_index in duplicates:
                continue
            low = detections[low_index]
            if low.get("category") != category:
                continue
            low_box = low["bbox"]
            low_area = _area(low_box)
            if low_area <= 0:
                continue
            overlap = _intersection(high_box, low_box)
            if overlap / min(high_area, low_area) >= DUPLICATE_CONTAINMENT:
                duplicates.add(low_index)

    return duplicates


def associate_detections(
    person_boxes: list[list[float]],
    detections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Attach each PPE detection to at most one worker.

    A detection is attached when its box is mostly covered by the worker's head
    region (hard hat / no hard hat) or torso region (vest / no vest) and its
    centre falls inside that region.  Greedy assignment by overlap score keeps
    one PPE object from being counted for several workers.  Duplicate boxes of
    the same category are removed first so a low-confidence copy cannot
    outrank the confident detection for the same slot.
    """
    slots = {
        "hardhat": "helmet",
        "no_hardhat": "helmet",
        "vest": "vest",
        "no_vest": "vest",
    }

    # Callers can pass plain detection dictionaries; always expose the fields.
    for detection in detections:
        detection.setdefault("assigned_person", None)
        detection.setdefault("association_score", None)

    regions: dict[tuple[int, str], list[float]] = {}
    for index, person_box in enumerate(person_boxes):
        regions[(index, "helmet")] = _head_region(person_box)
        regions[(index, "vest")] = _torso_region(person_box)

    candidates: list[tuple[float, int, int, str]] = []
    duplicates = _duplicate_detection_indices(detections)
    for det_index, detection in enumerate(detections):
        if det_index in duplicates:
            continue
        slot = slots.get(detection.get("category", ""))
        if slot is None:
            continue
        bbox = detection["bbox"]
        for person_index in range(len(person_boxes)):
            region = regions[(person_index, slot)]
            score = _intersection_over_detection(bbox, region)
            if score < MIN_ASSOCIATION_SCORE:
                continue
            if not _center_inside(bbox, region):
                continue
            candidates.append((score, det_index, person_index, slot))

    candidates.sort(reverse=True)

    assigned_detections: set[int] = set()
    assigned_slots: set[tuple[int, str]] = set()

    for score, det_index, person_index, slot in candidates:
        if det_index in assigned_detections:
            continue
        if (person_index, slot) in assigned_slots:
            continue
        detection = detections[det_index]
        detection["assigned_person"] = person_index + 1
        detection["association_score"] = round(float(score), 4)
        assigned_detections.add(det_index)
        assigned_slots.add((person_index, slot))

    return detections


def build_person_records(
    person_boxes: list[list[float]],
    person_confidences: list[float],
    detections: list[dict[str, Any]],
    source: str,
) -> list[dict[str, Any]]:
    """Describe each worker and the PPE evidence attached to them."""
    records: list[dict[str, Any]] = []

    for index, (box, confidence) in enumerate(zip(person_boxes, person_confidences), start=1):
        helmet_detection = None
        vest_detection = None

        for detection in detections:
            if detection.get("assigned_person") != index:
                continue
            category = detection.get("category")
            if category in {"hardhat", "no_hardhat"}:
                if helmet_detection is None or detection["confidence"] > helmet_detection["confidence"]:
                    helmet_detection = detection
            elif category in {"vest", "no_vest"}:
                if vest_detection is None or detection["confidence"] > vest_detection["confidence"]:
                    vest_detection = detection

        def state_for(detection: dict[str, Any] | None, positive: str) -> dict[str, Any]:
            if detection is None:
                return {"state": "unknown", "confidence": None, "label": None}
            state = "present" if detection["category"] == positive else "missing"
            return {
                "state": state,
                "confidence": detection["confidence"],
                "label": detection["label"],
            }

        helmet = state_for(helmet_detection, "hardhat")
        vest = state_for(vest_detection, "vest")

        if helmet["state"] == "missing" or vest["state"] == "missing":
            worker_status = "potential_violation"
        elif helmet["state"] == "present" and vest["state"] == "present":
            worker_status = "clear"
        else:
            worker_status = "uncertain"

        records.append(
            {
                "id": index,
                "bbox": [round(float(value), 1) for value in box],
                "confidence": round(float(confidence), 4),
                "source": source,
                "helmet": helmet,
                "vest": vest,
                "status": worker_status,
            }
        )

    return records


def assess(
    persons: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    counts: dict[str, int],
    person_detector_available: bool = True,
) -> dict[str, Any]:
    """
    Build the three-state assessment from actual evidence.

    potential_violation     explicit NO-Hardhat / NO-Safety Vest detections
    no_violation_detected   every worker has positive hard-hat and vest evidence
    insufficient_evidence   anything else (no people, missing evidence, unknowns)
    """
    hardhat_count = sum(counts.get(label, 0) for label in counts if classify_label(label) == "hardhat")
    no_hardhat_count = sum(
        counts.get(label, 0) for label in counts if classify_label(label) == "no_hardhat"
    )
    vest_count = sum(counts.get(label, 0) for label in counts if classify_label(label) == "vest")
    no_vest_count = sum(counts.get(label, 0) for label in counts if classify_label(label) == "no_vest")

    missing_detections = [
        detection
        for detection in detections
        if detection.get("category") in {"no_hardhat", "no_vest"}
    ]
    unassigned_missing = [
        detection for detection in missing_detections if detection.get("assigned_person") is None
    ]

    reasons: list[str] = []

    if not persons:
        reasons.append("No people were detected in the image.")

    if hardhat_count + no_hardhat_count == 0:
        reasons.append("No hard-hat evidence was detected (no Hardhat or NO-Hardhat boxes).")

    if vest_count + no_vest_count == 0:
        reasons.append("No safety-vest evidence was detected (no Safety Vest or NO-Safety Vest boxes).")

    unknown_helmet = [person for person in persons if person["helmet"]["state"] == "unknown"]
    unknown_vest = [person for person in persons if person["vest"]["state"] == "unknown"]

    if unknown_helmet:
        workers = ", ".join(f"P{person['id']}" for person in unknown_helmet)
        reasons.append(
            f"Hard-hat status could not be confirmed for {workers} (no hard-hat box overlaps the head region)."
        )

    if unknown_vest:
        workers = ", ".join(f"P{person['id']}" for person in unknown_vest)
        reasons.append(
            f"Safety-vest status could not be confirmed for {workers} (no vest box overlaps the torso region)."
        )

    if unassigned_missing:
        reasons.append(
            f"{len(unassigned_missing)} missing-PPE detection(s) could not be matched to a detected worker."
        )

    if not person_detector_available:
        reasons.append(
            "The dedicated person detector is unavailable; people may be undercounted."
        )

    violation_count = len(missing_detections)

    if violation_count:
        return {
            "status": "potential_violation",
            "title": "Potential PPE violation detected",
            "message": (
                f"{violation_count} explicit missing-PPE detection(s) require manual review. "
                "Check the flagged workers in the photo before acting."
            ),
            "reasons": reasons,
        }

    if not persons or unknown_helmet or unknown_vest or not person_detector_available:
        return {
            "status": "insufficient_evidence",
            "title": "Unable to assess fully",
            "message": (
                "The model did not collect enough reliable evidence to clear this image. "
                "Review the reasons below and inspect the photo manually."
            ),
            "reasons": reasons,
        }

    if hardhat_count + no_hardhat_count == 0 or vest_count + no_vest_count == 0:
        return {
            "status": "insufficient_evidence",
            "title": "Unable to assess fully",
            "message": (
                "People were detected, but at least one required PPE category has no evidence. "
                "Inspect the photo manually."
            ),
            "reasons": reasons,
        }

    return {
        "status": "no_violation_detected",
        "title": "No potential violations detected",
        "message": (
            "Every detected worker has positive hard-hat and safety-vest evidence, and no "
            "missing-PPE detections passed the filter. This is not proof of compliance."
        ),
        "reasons": [],
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("The image could not be decoded.")
    return image


def _write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise RuntimeError("Could not encode the annotated image.")
    encoded.tofile(str(path))


def _draw_box(
    image: np.ndarray,
    box: Iterable[float],
    color: tuple[int, int, int],
    label: str,
) -> None:
    height, width = image.shape[:2]
    max_dim = max(height, width)
    thickness = max(1, round(max_dim / 700))
    font_scale = max(0.45, min(1.1, max_dim / 1500))
    font = cv2.FONT_HERSHEY_SIMPLEX

    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)

    (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
    text_y = max(text_height + 6, y1)
    cv2.rectangle(
        image,
        (x1, text_y - text_height - 6),
        (x1 + text_width + 8, text_y + baseline),
        color,
        cv2.FILLED,
    )
    cv2.putText(
        image,
        label,
        (x1 + 4, text_y - 4),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _annotate(
    image: np.ndarray,
    persons: list[dict[str, Any]],
    detections: list[dict[str, Any]],
) -> np.ndarray:
    canvas = image.copy()

    person_color = (168, 82, 91)  # indigo (BGR)
    safe_color = (79, 123, 15)  # green (BGR)
    danger_color = (24, 35, 180)  # red (BGR)
    other_color = (128, 114, 107)  # neutral (BGR)

    # People first so PPE boxes stay visible on top.
    for person in persons:
        _draw_box(
            canvas,
            person["bbox"],
            person_color,
            f"P{person['id']} Person {person['confidence']:.2f}",
        )

    for detection in detections:
        category = detection.get("category")
        if category == "person":
            continue
        if category in {"no_hardhat", "no_vest"}:
            color = danger_color
        elif category in {"hardhat", "vest"}:
            color = safe_color
        else:
            color = other_color

        suffix = ""
        if detection.get("assigned_person"):
            suffix = f" · P{detection['assigned_person']}"
        elif category in {"no_hardhat", "no_vest", "hardhat", "vest"}:
            suffix = " · unassigned"

        _draw_box(
            canvas,
            detection["bbox"],
            color,
            f"{detection['label']} {detection['confidence']:.2f}{suffix}",
        )

    return canvas


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _empty_result() -> dict[str, Any]:
    return {
        "success": False,
        "annotated_rel": None,
        "annotated_abs": None,
        "counts": {},
        "counts_by_source": [],
        "person_count": 0,
        "hardhat_count": 0,
        "no_hardhat_count": 0,
        "vest_count": 0,
        "no_vest_count": 0,
        "potential_no_hardhat": [],
        "potential_no_vest": [],
        "persons": [],
        "detections": [],
        "assessment_status": "insufficient_evidence",
        "assessment_title": "Unable to assess fully",
        "assessment_message": "",
        "assessment_reasons": [],
        "confidence_used": None,
        "person_confidence_used": None,
        "inference_ms": 0.0,
        "ppe_inference_ms": 0.0,
        "person_inference_ms": 0.0,
        "image_size": [0, 0],
        "model_info": {},
        "error": None,
    }


def analyze_image(
    image_path: str | Path,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
) -> dict[str, Any]:
    """Run PPE + person inference on one image and build the assessment."""
    result = _empty_result()
    started = time.perf_counter()
    image_path = Path(image_path)

    if not image_path.is_file():
        result["error"] = f"Image not found: {image_path.name}"
        return result

    try:
        conf = min(MAX_CONF, max(MIN_CONF, float(conf)))
        imgsz = int(imgsz)
        if imgsz not in ALLOWED_IMGSZ:
            imgsz = DEFAULT_IMGSZ

        result["confidence_used"] = round(conf, 2)

        ppe_model = get_ppe_model()
        person_model = get_person_model()

        with _predict_lock:
            ppe_prediction = ppe_model.predict(
                source=str(image_path),
                conf=conf,
                imgsz=imgsz,
                device="cpu",
                save=False,
                verbose=False,
                max_det=300,
            )[0]

            person_conf = max(conf, PERSON_CONF_FLOOR)
            result["person_confidence_used"] = round(person_conf, 2)

            person_prediction = None
            if person_model is not None:
                person_prediction = person_model.predict(
                    source=str(image_path),
                    conf=person_conf,
                    imgsz=imgsz,
                    device="cpu",
                    classes=[0],
                    save=False,
                    verbose=False,
                    max_det=300,
                )[0]

        ppe_inference = float(ppe_prediction.speed.get("inference", 0.0))
        person_inference = (
            float(person_prediction.speed.get("inference", 0.0))
            if person_prediction is not None
            else 0.0
        )
        result["ppe_inference_ms"] = round(ppe_inference, 1)
        result["person_inference_ms"] = round(person_inference, 1)
        result["inference_ms"] = round(ppe_inference + person_inference, 1)

        names = ppe_prediction.names if hasattr(ppe_prediction, "names") else {}

        detections: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        potential_no_hardhat: list[dict[str, Any]] = []
        potential_no_vest: list[dict[str, Any]] = []

        boxes = getattr(ppe_prediction, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            for bbox, confidence, class_id in zip(
                boxes.xyxy.tolist(),
                boxes.conf.tolist(),
                boxes.cls.tolist(),
            ):
                label = str(names.get(int(class_id), str(int(class_id))))
                category = classify_label(label)

                # The PPE checkpoint's Person class is not used when the
                # dedicated detector is available; this avoids duplicate boxes.
                if category == "person" and person_model is not None:
                    continue

                detection = {
                    "label": label,
                    "category": category,
                    "confidence": round(float(confidence), 4),
                    "bbox": [round(float(value), 1) for value in bbox],
                    "source": "ppe",
                    "assigned_person": None,
                    "association_score": None,
                }
                detections.append(detection)

                if category != "person":
                    counts[label] = counts.get(label, 0) + 1

        person_boxes: list[list[float]] = []
        person_confidences: list[float] = []
        person_source = ""

        if person_prediction is not None:
            person_source = "YOLOv8n COCO person detector"
            person_boxes_raw = getattr(person_prediction, "boxes", None)
            if person_boxes_raw is not None and len(person_boxes_raw) > 0:
                for bbox, confidence, class_id in zip(
                    person_boxes_raw.xyxy.tolist(),
                    person_boxes_raw.conf.tolist(),
                    person_boxes_raw.cls.tolist(),
                ):
                    if int(class_id) != 0:
                        continue
                    person_boxes.append([round(float(value), 1) for value in bbox])
                    person_confidences.append(round(float(confidence), 4))
        else:
            # Degraded fallback: use the PPE model's own Person boxes.
            person_source = "PPE model Person class (fallback)"
            for detection in detections:
                if detection["category"] != "person":
                    continue
                person_boxes.append(detection["bbox"])
                person_confidences.append(detection["confidence"])

        # Remove PPE-model person boxes from the PPE detection list; they are
        # represented by the person records instead.
        detections = [detection for detection in detections if detection["category"] != "person"]

        detections = associate_detections(person_boxes, detections)
        persons = build_person_records(
            person_boxes,
            person_confidences,
            detections,
            person_source,
        )

        for detection in detections:
            if detection["category"] == "no_hardhat":
                potential_no_hardhat.append(
                    {
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"],
                        "assigned_person": detection["assigned_person"],
                    }
                )
            elif detection["category"] == "no_vest":
                potential_no_vest.append(
                    {
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"],
                        "assigned_person": detection["assigned_person"],
                    }
                )

        assessment = assess(
            persons,
            detections,
            counts,
            person_detector_available=person_model is not None,
        )

        counts_by_source = [
            {
                "label": label,
                "source": "PPE model",
                "count": count,
            }
            for label, count in sorted(counts.items())
        ]
        if persons:
            counts_by_source.append(
                {
                    "label": "Person",
                    "source": person_source,
                    "count": len(persons),
                }
            )

        result.update(
            {
                "counts": counts,
                "counts_by_source": counts_by_source,
                "person_count": len(persons),
                "hardhat_count": sum(
                    count for label, count in counts.items() if classify_label(label) == "hardhat"
                ),
                "no_hardhat_count": sum(
                    count for label, count in counts.items() if classify_label(label) == "no_hardhat"
                ),
                "vest_count": sum(
                    count for label, count in counts.items() if classify_label(label) == "vest"
                ),
                "no_vest_count": sum(
                    count for label, count in counts.items() if classify_label(label) == "no_vest"
                ),
                "potential_no_hardhat": potential_no_hardhat,
                "potential_no_vest": potential_no_vest,
                "persons": persons,
                "detections": detections,
                "assessment_status": assessment["status"],
                "assessment_title": assessment["title"],
                "assessment_message": assessment["message"],
                "assessment_reasons": assessment["reasons"],
            }
        )

        image = _read_image(image_path)
        annotated = _annotate(image, persons, detections)

        annotated_name = f"det_{uuid.uuid4().hex[:10]}.jpg"
        annotated_path = RENDERED_DIR / annotated_name
        _write_image(annotated_path, annotated)

        result["annotated_abs"] = str(annotated_path)
        result["annotated_rel"] = f"rendered/{annotated_name}"
        result["image_size"] = [int(image.shape[1]), int(image.shape[0])]
        result["model_info"] = {
            "ppe_model": PPE_MODEL_PATH.name,
            "ppe_load_ms": _ppe_load_ms,
            "person_model": (
                "yolov8n.pt (COCO)" if person_model is not None else None
            ),
            "person_load_ms": _person_load_ms,
            "person_source": person_source,
            "person_model_available": person_model is not None,
        }
        result["total_ms"] = round((time.perf_counter() - started) * 1000.0, 1)
        result["success"] = True
        return result

    except Exception as exc:  # noqa: BLE001 - returned to the caller, not raised
        logger.exception("Inference failed for %s", image_path)
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
