"""Reusable video analysis engine.

Pipeline
--------
1. Probe the container with OpenCV and plan frame sampling.
2. Track people with the dedicated person checkpoint and Ultralytics ByteTrack.
3. Run the existing PPE checkpoint on the same sampled frames.
4. Reuse :func:`infer_engine.associate_detections` for hard hat / vest and the
   same geometry helpers for gloves, goggles, and masks.
5. Accumulate per-worker evidence over time and only emit a safety event when
   explicit negative PPE detections persist across sampled frames.
6. Write an annotated MP4 that keeps the source aspect ratio.

The image pipeline is untouched: this module never changes model weights,
image thresholds, labels, or assessment rules.
"""

from __future__ import annotations

import math
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from ultralytics import YOLO

import infer_engine as engine
from .sample_library import probe_video

# ---------------------------------------------------------------------------
# Label mapping for the 14 PPE classes
# ---------------------------------------------------------------------------

VIDEO_CATEGORY_BY_NORMALIZED_LABEL = {
    "person": "person",
    "hardhat": "hardhat",
    "helmet": "hardhat",
    "nohardhat": "no_hardhat",
    "nohelmet": "no_hardhat",
    "safetyvest": "vest",
    "vest": "vest",
    "nosafetyvest": "no_vest",
    "novest": "no_vest",
    "gloves": "gloves",
    "nogloves": "no_gloves",
    "goggles": "goggles",
    "nogoggles": "no_goggles",
    "mask": "mask",
    "nomask": "no_mask",
    "falldetected": "fall_detected",
    "ladder": "ladder",
    "safetycone": "safety_cone",
}

SLOT_FOR_CATEGORY = {
    "hardhat": "helmet",
    "no_hardhat": "helmet",
    "vest": "vest",
    "no_vest": "vest",
    "gloves": "gloves",
    "no_gloves": "gloves",
    "goggles": "goggles",
    "no_goggles": "goggles",
    "mask": "mask",
    "no_mask": "mask",
}

POLARITY = {
    "hardhat": 1,
    "vest": 1,
    "gloves": 1,
    "goggles": 1,
    "mask": 1,
    "no_hardhat": -1,
    "no_vest": -1,
    "no_gloves": -1,
    "no_goggles": -1,
    "no_mask": -1,
}

CATEGORY_LABELS = {
    "helmet": "Missing hard hat",
    "vest": "Missing safety vest",
    "gloves": "Missing gloves",
    "goggles": "Missing goggles",
    "mask": "Missing mask",
}

CATEGORY_ORDER = ("helmet", "vest", "gloves", "goggles", "mask")

# Extra region geometry for classes the image pipeline does not assess.
HANDS_TOP_RATIO = 0.40
HANDS_BOTTOM_RATIO = 0.98
HANDS_SIDE_INSET = 0.02

VIDEO_FILE_NAME = "annotated.mp4"


class VideoProcessingError(RuntimeError):
    """A user-facing video processing failure."""


def classify_video_label(label: str) -> str:
    normalized = "".join(ch.lower() for ch in str(label or "") if ch.isalnum())
    return VIDEO_CATEGORY_BY_NORMALIZED_LABEL.get(normalized, "other")


# ---------------------------------------------------------------------------
# Settings and sampling
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoSettings:
    sample_fps: float = 2.0
    max_sampled_frames: int = 180
    imgsz: int = 640
    conf: float = 0.25
    max_output_width: int = 1280
    min_observations: int = 3
    min_ratio: float = 0.5
    cooldown_s: float = 8.0

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None = None) -> "VideoSettings":
        payload = payload or {}

        def number(key: str, default: float, low: float, high: float) -> float:
            try:
                value = float(payload.get(key, default))
            except (TypeError, ValueError):
                value = default
            return min(high, max(low, value))

        imgsz = payload.get("imgsz", cls.imgsz)
        try:
            imgsz = int(imgsz)
        except (TypeError, ValueError):
            imgsz = cls.imgsz
        if imgsz not in engine.ALLOWED_IMGSZ:
            imgsz = cls.imgsz

        try:
            max_frames = int(payload.get("max_sampled_frames", cls.max_sampled_frames))
        except (TypeError, ValueError):
            max_frames = cls.max_sampled_frames
        max_frames = min(600, max(6, max_frames))

        try:
            min_obs = int(payload.get("min_observations", cls.min_observations))
        except (TypeError, ValueError):
            min_obs = cls.min_observations
        min_obs = min(30, max(2, min_obs))

        return cls(
            sample_fps=round(number("sample_fps", cls.sample_fps, 0.25, 10.0), 2),
            max_sampled_frames=max_frames,
            imgsz=imgsz,
            conf=round(number("conf", cls.conf, engine.MIN_CONF, engine.MAX_CONF), 2),
            max_output_width=int(number("max_output_width", cls.max_output_width, 320, 3840)),
            min_observations=min_obs,
            min_ratio=round(number("min_ratio", cls.min_ratio, 0.2, 1.0), 2),
            cooldown_s=round(number("cooldown_s", cls.cooldown_s, 0.0, 120.0), 1),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_fps": self.sample_fps,
            "max_sampled_frames": self.max_sampled_frames,
            "imgsz": self.imgsz,
            "conf": self.conf,
            "max_output_width": self.max_output_width,
            "min_observations": self.min_observations,
            "min_ratio": self.min_ratio,
            "cooldown_s": self.cooldown_s,
        }


def plan_sampling(fps: float, frame_count: int, settings: VideoSettings) -> dict[str, Any]:
    """Choose a stride that caps work and preserves the source duration."""
    fps = float(fps or 0.0)
    if fps <= 0:
        raise VideoProcessingError("The video does not report a usable frame rate.")

    target_sample_fps = max(0.25, min(float(settings.sample_fps), fps))
    stride = max(1, int(round(fps / target_sample_fps)))

    estimated = (
        int(math.ceil(frame_count / stride)) if frame_count and frame_count > 0 else 0
    )
    if estimated > settings.max_sampled_frames and frame_count > 0:
        stride = max(stride, int(math.ceil(frame_count / settings.max_sampled_frames)))
        estimated = int(math.ceil(frame_count / stride))

    if estimated <= 0:
        estimated = settings.max_sampled_frames

    output_fps = fps / stride
    return {
        "stride": stride,
        "sample_fps": round(fps / stride, 3),
        "output_fps": round(output_fps, 3),
        "estimated_frames": min(estimated, settings.max_sampled_frames),
    }


# ---------------------------------------------------------------------------
# Point-in-time tracker (dedicated model instance so image analysis cannot
# reset ByteTrack state mid-job)
# ---------------------------------------------------------------------------

_track_model: YOLO | None = None
_track_model_lock = threading.Lock()


def _person_checkpoint() -> Path:
    return (
        engine.PERSON_MODEL_PATH
        if engine.PERSON_MODEL_PATH.is_file()
        else engine.LEGACY_PERSON_MODEL_PATH
    )


def _get_track_model() -> YOLO:
    global _track_model
    with _track_model_lock:
        if _track_model is None:
            path = _person_checkpoint()
            if not path.is_file():
                raise VideoProcessingError("The person detector checkpoint is missing.")
            _track_model = YOLO(str(path))
        return _track_model


def reset_tracker() -> None:
    """Reset ByteTrack state so worker IDs never leak between jobs.

    Ultralytics' ``persist=True`` mode expects ``predictor.trackers`` to stay a
    list.  Deleting or replacing it with ``None`` makes the next track call
    fail, so each existing tracker is reset in place instead.
    """
    global _track_model
    with _track_model_lock:
        model = _track_model
    predictor = getattr(model, "predictor", None) if model is not None else None
    if predictor is None:
        return
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            try:
                tracker.reset()
            except Exception:  # noqa: BLE001 - best effort reset
                pass
        try:
            predictor.vid_path = [None] * len(trackers)
        except Exception:  # noqa: BLE001 - best effort reset
            pass


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class _FallbackIds:
    """Maintains IDs when the tracker returns none (e.g. a single-frame clip)."""

    def __init__(self) -> None:
        self.next_id = 1
        self.previous: list[dict[str, Any]] = []

    def assign(self, boxes: list[list[float]], confidences: list[float]) -> list[int]:
        ids: list[int] = []
        used: set[int] = set()
        for box in boxes:
            best_id = None
            best_iou = 0.3
            for item in self.previous:
                if item["id"] in used:
                    continue
                score = _iou(box, item["box"])
                if score > best_iou:
                    best_iou = score
                    best_id = item["id"]
            if best_id is None:
                best_id = self.next_id
                self.next_id += 1
            used.add(best_id)
            ids.append(best_id)
        self.previous = [
            {"id": track_id, "box": box}
            for track_id, box in zip(ids, boxes)
        ]
        return ids


def _track_people(
    frame: np.ndarray,
    settings: VideoSettings,
    fallback: _FallbackIds,
) -> list[dict[str, Any]]:
    model = _get_track_model()
    try:
        with engine.prediction_lock():
            results = model.track(
                source=frame,
                persist=True,
                conf=max(settings.conf, engine.PERSON_CONF_FLOOR),
                imgsz=settings.imgsz,
                device="cpu",
                classes=[0],
                tracker="bytetrack.yaml",
                verbose=False,
                max_det=300,
            )
    except Exception as exc:  # noqa: BLE001 - tracker failure is reported by the job
        raise VideoProcessingError(f"Worker tracking failed: {type(exc).__name__}: {exc}") from exc

    if not results:
        return []
    boxes = getattr(results[0], "boxes", None)
    if boxes is None or len(boxes) == 0:
        fallback.previous = []
        return []

    coordinates = boxes.xyxy.tolist()
    confidences = boxes.conf.tolist()
    class_ids = boxes.cls.tolist()
    raw_ids = boxes.id.int().tolist() if getattr(boxes, "id", None) is not None else None

    people: list[dict[str, Any]] = []
    for index, (box, confidence, class_id) in enumerate(
        zip(coordinates, confidences, class_ids)
    ):
        if int(class_id) != 0:
            continue
        people.append(
            {
                "bbox": [round(float(value), 1) for value in box],
                "confidence": round(float(confidence), 4),
                "track_id": int(raw_ids[index]) if raw_ids is not None else None,
            }
        )

    if not people:
        fallback.previous = []
        return []

    if raw_ids is None or any(person["track_id"] is None for person in people):
        assigned = fallback.assign(
            [person["bbox"] for person in people],
            [person["confidence"] for person in people],
        )
        for person, track_id in zip(people, assigned):
            person["track_id"] = int(track_id)
    else:
        # ByteTrack guarantees unique IDs per frame; enforce it defensively.
        seen: set[int] = set()
        for person in people:
            track_id = int(person["track_id"])
            if track_id in seen:
                track_id = fallback.next_id
                fallback.next_id += 1
                person["track_id"] = track_id
            seen.add(track_id)
        fallback.previous = [
            {"id": person["track_id"], "box": person["bbox"]} for person in people
        ]

    return people


# ---------------------------------------------------------------------------
# PPE detection and association
# ---------------------------------------------------------------------------


def detect_ppe(frame: np.ndarray, settings: VideoSettings) -> list[dict[str, Any]]:
    """Run the existing PPE checkpoint and return normalised detections."""
    model = engine.get_ppe_model()
    with engine.prediction_lock():
        prediction = model.predict(
            source=frame,
            conf=settings.conf,
            imgsz=settings.imgsz,
            device="cpu",
            save=False,
            verbose=False,
            max_det=300,
        )[0]

    names = prediction.names if hasattr(prediction, "names") else {}
    detections: list[dict[str, Any]] = []
    boxes = getattr(prediction, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return detections

    for bbox, confidence, class_id in zip(
        boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()
    ):
        label = str(names.get(int(class_id), str(int(class_id))))
        category = classify_video_label(label)
        # The dedicated person detector owns people; the PPE checkpoint's own
        # Person boxes are dropped so they cannot be counted twice.
        if category == "person":
            continue
        detections.append(
            {
                "label": label,
                "category": category,
                "confidence": round(float(confidence), 4),
                "bbox": [round(float(value), 1) for value in bbox],
                "source": "ppe",
                "assigned_person": None,
                "association_score": None,
            }
        )
    return detections


def _hands_region(person_box: list[float]) -> list[float]:
    x1, y1, x2, y2 = person_box
    width = x2 - x1
    height = y2 - y1
    return [
        x1 + width * HANDS_SIDE_INSET,
        y1 + height * HANDS_TOP_RATIO,
        x2 - width * HANDS_SIDE_INSET,
        y1 + height * HANDS_BOTTOM_RATIO,
    ]


_EXTRA_SLOTS = {
    "gloves": "gloves",
    "no_gloves": "gloves",
    "goggles": "goggles",
    "no_goggles": "goggles",
    "mask": "mask",
    "no_mask": "mask",
}

_EXTRA_REGIONS = {
    "gloves": "hands",
    "goggles": "head",
    "mask": "head",
}


def _associate_extra(
    person_boxes: list[list[float]], detections: list[dict[str, Any]]
) -> None:
    """Attach gloves/goggles/mask with the same geometry rules as hard hats.

    Positives and negatives for the same slot are allowed to attach to the same
    worker: that contradiction is real conflict evidence and must reach the
    worker state, not be silently dropped.
    """
    regions: dict[tuple[int, str], list[float]] = {}
    for index, box in enumerate(person_boxes):
        regions[(index, "head")] = engine._head_region(box)
        regions[(index, "hands")] = _hands_region(box)

    candidates: list[tuple[float, int, int, str, int]] = []
    for det_index, detection in enumerate(detections):
        if detection.get("assigned_person") is not None:
            continue
        category = detection.get("category", "")
        slot = _EXTRA_SLOTS.get(category)
        if slot is None:
            continue
        polarity = POLARITY.get(category, 0)
        region_name = _EXTRA_REGIONS[slot]
        bbox = detection["bbox"]
        for person_index in range(len(person_boxes)):
            region = regions[(person_index, region_name)]
            score = engine._intersection_over_detection(bbox, region)
            if score < engine.MIN_ASSOCIATION_SCORE:
                continue
            if not engine._center_inside(bbox, region):
                continue
            candidates.append((score, det_index, person_index, slot, polarity))

    candidates.sort(reverse=True)
    used_detections: set[int] = set()
    used_slots: set[tuple[int, str, int]] = set()
    for score, det_index, person_index, slot, polarity in candidates:
        key = (person_index, slot, polarity)
        if det_index in used_detections or key in used_slots:
            continue
        detection = detections[det_index]
        detection["assigned_person"] = person_index + 1
        detection["association_score"] = round(float(score), 4)
        used_detections.add(det_index)
        used_slots.add(key)


def _associate_contradictions(
    person_boxes: list[list[float]], detections: list[dict[str, Any]]
) -> None:
    """Attach a negative hat/vest box next to an already attached positive one.

    :func:`infer_engine.associate_detections` intentionally assigns only one
    box per worker slot, which is correct for the image dashboard.  For video
    evidence, a NO-Hardhat box on a worker who also has a Hardhat box is a
    contradiction that must be visible as such.  Only the opposite polarity is
    added here, so duplicate boxes of the same class stay suppressed.
    """
    used: set[tuple[int, str, int]] = set()
    for detection in detections:
        owner = detection.get("assigned_person")
        slot = SLOT_FOR_CATEGORY.get(detection.get("category", ""))
        polarity = POLARITY.get(detection.get("category", ""))
        if owner and slot in {"helmet", "vest"} and polarity is not None:
            used.add((int(owner), slot, polarity))

    candidates: list[tuple[float, int, int, str, int]] = []
    for det_index, detection in enumerate(detections):
        if detection.get("assigned_person") is not None:
            continue
        slot = SLOT_FOR_CATEGORY.get(detection.get("category", ""))
        polarity = POLARITY.get(detection.get("category", ""))
        if slot not in {"helmet", "vest"} or polarity is None:
            continue
        region_for = (
            engine._head_region if slot == "helmet" else engine._torso_region
        )
        bbox = detection["bbox"]
        for person_index, person_box in enumerate(person_boxes):
            person_id = person_index + 1
            if (person_id, slot, polarity) in used:
                continue
            if (person_id, slot, -polarity) not in used:
                continue
            region = region_for(person_box)
            score = engine._intersection_over_detection(bbox, region)
            if score < engine.MIN_ASSOCIATION_SCORE:
                continue
            if not engine._center_inside(bbox, region):
                continue
            candidates.append((score, det_index, person_id, slot, polarity))

    candidates.sort(reverse=True)
    for score, det_index, person_id, slot, polarity in candidates:
        key = (person_id, slot, polarity)
        if key in used or detections[det_index].get("assigned_person") is not None:
            continue
        detections[det_index]["assigned_person"] = person_id
        detections[det_index]["association_score"] = round(float(score), 4)
        used.add(key)


def associate_frame(
    person_boxes: list[list[float]], detections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Reuse the image association for hat/vest, then extend to other PPE."""
    detections = engine.associate_detections(person_boxes, detections)
    _associate_contradictions(person_boxes, detections)
    _associate_extra(person_boxes, detections)
    return detections


def worker_states(person_index: int, detections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Summarise one worker's per-slot evidence for a single frame."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detection in detections:
        if detection.get("assigned_person") != person_index:
            continue
        slot = SLOT_FOR_CATEGORY.get(detection.get("category", ""))
        if slot:
            buckets[slot].append(detection)

    states: dict[str, dict[str, Any]] = {}
    for slot in CATEGORY_ORDER:
        items = buckets.get(slot, [])
        positives = [item for item in items if POLARITY.get(item["category"]) == 1]
        negatives = [item for item in items if POLARITY.get(item["category"]) == -1]
        if positives and negatives:
            best = max(items, key=lambda item: item["confidence"])
            states[slot] = {
                "state": "conflict",
                "confidence": best["confidence"],
                "label": best["label"],
            }
        elif negatives:
            best = max(negatives, key=lambda item: item["confidence"])
            states[slot] = {
                "state": "missing",
                "confidence": best["confidence"],
                "label": best["label"],
            }
        elif positives:
            best = max(positives, key=lambda item: item["confidence"])
            states[slot] = {
                "state": "present",
                "confidence": best["confidence"],
                "label": best["label"],
            }
        else:
            states[slot] = {"state": "unknown", "confidence": None, "label": None}
    return states


def frame_status(states: dict[str, dict[str, Any]]) -> str:
    if any(state["state"] == "missing" for state in states.values()):
        return "potential_violation"
    if (
        states["helmet"]["state"] == "present"
        and states["vest"]["state"] == "present"
        and not any(state["state"] == "conflict" for state in states.values())
    ):
        return "clear"
    return "uncertain"


# ---------------------------------------------------------------------------
# Temporal verification
# ---------------------------------------------------------------------------


@dataclass
class _Observation:
    timestamp: float
    frame_index: int
    state: str
    confidence: float | None


class TemporalEventEngine:
    """Turns per-frame evidence into cautious, persistent safety events."""

    def __init__(self, settings: VideoSettings) -> None:
        self.settings = settings
        self.frames: dict[int, list[dict[str, Any]]] = defaultdict(list)
        self.observations: dict[tuple[int, str], list[_Observation]] = defaultdict(list)

    def observe(
        self,
        track_id: int,
        timestamp: float,
        frame_index: int,
        states: dict[str, dict[str, Any]],
    ) -> None:
        status = frame_status(states)
        self.frames[int(track_id)].append(
            {
                "timestamp": float(timestamp),
                "frame_index": int(frame_index),
                "status": status,
                "states": states,
            }
        )
        for slot, info in states.items():
            self.observations[(int(track_id), slot)].append(
                _Observation(
                    timestamp=float(timestamp),
                    frame_index=int(frame_index),
                    state=str(info.get("state") or "unknown"),
                    confidence=info.get("confidence"),
                )
            )

    def _sample_gap(self, observations: list[_Observation]) -> float:
        if len(observations) < 2:
            return 1.0 / max(self.settings.sample_fps, 0.25)
        deltas = [
            later.timestamp - earlier.timestamp
            for earlier, later in zip(observations, observations[1:])
            if later.timestamp > earlier.timestamp
        ]
        if not deltas:
            return 1.0 / max(self.settings.sample_fps, 0.25)
        deltas.sort()
        return max(0.2, deltas[len(deltas) // 2] * 2.5)

    def _runs(self, observations: list[_Observation]) -> list[list[_Observation]]:
        gap_limit = self._sample_gap(observations)
        runs: list[list[_Observation]] = []
        current: list[_Observation] = []
        for observation in observations:
            if observation.state != "missing":
                if current:
                    runs.append(current)
                    current = []
                continue
            if current and (observation.timestamp - current[-1].timestamp) > gap_limit:
                runs.append(current)
                current = []
            current.append(observation)
        if current:
            runs.append(current)
        return runs

    def _events_for(
        self, track_id: int, slot: str, observations: list[_Observation]
    ) -> list[dict[str, Any]]:
        confirmed: list[dict[str, Any]] = []
        for run in self._runs(observations):
            if len(run) < self.settings.min_observations:
                continue

            first = run[0].timestamp
            last = run[-1].timestamp
            before = [item for item in observations if item.timestamp < first]
            after = [item for item in observations if item.timestamp > last]
            window = (
                ([before[-1]] if before else [])
                + run
                + ([after[0]] if after else [])
            )
            missing = [item for item in window if item.state == "missing"]
            ratio = len(missing) / max(1, len(window))
            if ratio < self.settings.min_ratio:
                continue

            confidences = [
                item.confidence for item in run if item.confidence is not None
            ]
            confirmed.append(
                {
                    "worker_id": int(track_id),
                    "category": slot,
                    "label": CATEGORY_LABELS.get(slot, slot),
                    "status": "potential_violation",
                    "confidence": round(max(confidences), 4) if confidences else None,
                    "first_seen_s": round(first, 2),
                    "last_seen_s": round(last, 2),
                    "first_frame": int(run[0].frame_index),
                    "last_frame": int(run[-1].frame_index),
                    "observations": len(run),
                    "evidence": (
                        f"Explicit negative PPE evidence in {len(run)} sampled "
                        f"frame(s); {len(missing)}/{len(window)} observations in "
                        "the surrounding window support it."
                    ),
                }
            )

        # Cooldown suppresses duplicate alerts, but never manufactures evidence:
        # each run above had to qualify on its own first.
        merged: list[dict[str, Any]] = []
        for event in confirmed:
            if (
                merged
                and (event["first_seen_s"] - merged[-1]["last_seen_s"])
                < self.settings.cooldown_s
            ):
                previous = merged[-1]
                previous["last_seen_s"] = event["last_seen_s"]
                previous["last_frame"] = event["last_frame"]
                previous["observations"] += event["observations"]
                previous["merged_runs"] = previous.get("merged_runs", 1) + 1
                # Keep the rendered evidence consistent with the merged totals.
                # The per-run ratio is intentionally not reused here because it
                # describes only one of the merged runs.
                previous["evidence"] = (
                    f"Explicit negative PPE evidence in {previous['observations']} "
                    f"sampled frame(s) across {previous['merged_runs']} verified runs "
                    f"within the {self.settings.cooldown_s:g} s cooldown window."
                )
                if event["confidence"] is not None:
                    previous["confidence"] = max(
                        previous["confidence"] or 0.0, event["confidence"]
                    )
            else:
                merged.append(dict(event))
        return merged

    def finalize(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        events: list[dict[str, Any]] = []
        workers: list[dict[str, Any]] = []

        for track_id in sorted(self.frames):
            frames = self.frames[track_id]
            track_events = []
            for slot in CATEGORY_ORDER:
                observations = self.observations.get((track_id, slot), [])
                track_events.extend(self._events_for(track_id, slot, observations))
            track_events.sort(key=lambda item: (item["first_seen_s"], item["category"]))
            events.extend(track_events)

            verified = {event["category"] for event in track_events}
            states: dict[str, str] = {}
            confidences: dict[str, float | None] = {}
            for slot in CATEGORY_ORDER:
                observations = self.observations.get((track_id, slot), [])
                counts = Counter(item.state for item in observations)
                total = max(1, sum(counts.values()))
                if slot in verified:
                    states[slot] = "missing"
                elif counts["conflict"]:
                    states[slot] = "conflict"
                elif (
                    counts["present"] >= self.settings.min_observations
                    and counts["present"] / total >= self.settings.min_ratio
                ):
                    states[slot] = "present"
                else:
                    states[slot] = "unconfirmed"
                confidence_values = [
                    item.confidence
                    for item in observations
                    if item.confidence is not None
                ]
                confidences[slot] = (
                    round(max(confidence_values), 4) if confidence_values else None
                )

            if track_events:
                status = "potential_violation"
            elif states["helmet"] == "present" and states["vest"] == "present":
                status = "compliant"
            else:
                status = "unconfirmed"

            workers.append(
                {
                    "id": int(track_id),
                    "first_seen_s": round(frames[0]["timestamp"], 2),
                    "last_seen_s": round(frames[-1]["timestamp"], 2),
                    "frames_observed": len(frames),
                    "status": status,
                    "states": states,
                    "confidence": confidences,
                    "events": [event["label"] for event in track_events],
                }
            )

        events.sort(
            key=lambda item: (item["first_seen_s"], item["worker_id"], item["category"])
        )
        return events, workers


def summarize_workers(
    workers: list[dict[str, Any]], events: list[dict[str, Any]]
) -> dict[str, Any]:
    by_category = Counter(event["category"] for event in events)
    return {
        "worker_count": len(workers),
        "compliant_workers": sum(1 for worker in workers if worker["status"] == "compliant"),
        "violation_workers": sum(
            1 for worker in workers if worker["status"] == "potential_violation"
        ),
        "unconfirmed_workers": sum(
            1 for worker in workers if worker["status"] == "unconfirmed"
        ),
        "event_count": len(events),
        "events_by_category": dict(by_category),
    }


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

COLOR_BY_STATUS = {
    "clear": (79, 123, 15),          # green
    "potential_violation": (36, 52, 210),  # red
    "uncertain": (0, 146, 255),      # orange
}
COLOR_POSITIVE = (79, 123, 15)
COLOR_NEGATIVE = (36, 52, 210)
COLOR_OTHER = (128, 114, 107)
COLOR_PANEL = (18, 38, 58)
COLOR_TEXT = (245, 247, 251)
COLOR_ACCENT = (216, 201, 32)  # cyan-ish


def _draw_label(
    image: np.ndarray,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, scale, thickness)
    top = max(0, y - text_height - baseline - 4)
    right = min(image.shape[1] - 1, x + text_width + 8)
    cv2.rectangle(image, (max(0, x), top), (right, y + 2), color, cv2.FILLED)
    cv2.putText(
        image,
        text,
        (max(0, x) + 4, max(text_height + 1, y - baseline)),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def _draw_box(
    image: np.ndarray,
    box: list[float],
    color: tuple[int, int, int],
    label: str,
    scale: float,
    thickness: int,
) -> None:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(x1, width - 1))
    x2 = max(0, min(x2, width - 1))
    y1 = max(0, min(y1, height - 1))
    y2 = max(0, min(y2, height - 1))
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    if label:
        _draw_label(image, x1, max(y1, 16), label, color, scale, max(1, thickness - 1))


def _status_text(status: str) -> str:
    return {
        "clear": "CLEAR",
        "potential_violation": "REVIEW",
        "uncertain": "UNCONFIRMED",
    }.get(status, status.upper())


def _state_text(value: str) -> str:
    return {
        "present": "OK",
        "missing": "MISSING",
        "conflict": "CONFLICT",
        "unconfirmed": "?",
        "unknown": "?",
    }.get(value, "?")


def annotate_frame(
    frame: np.ndarray,
    timestamp: float,
    people: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    states_by_track: dict[int, dict[str, dict[str, Any]]],
    status_by_track: dict[int, str],
    scale_factor: float,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    base = max(1, round(max(height, width) / 900))
    scale = max(0.42, min(1.0, max(height, width) / 1600.0))

    for person in people:
        track_id = int(person["track_id"])
        status = status_by_track.get(track_id, "uncertain")
        color = COLOR_BY_STATUS.get(status, COLOR_BY_STATUS["uncertain"])
        scaled = [value * scale_factor for value in person["bbox"]]
        states = states_by_track.get(track_id, {})
        summary = " ".join(
            f"{short}:{_state_text(states.get(slot, {}).get('state', 'unknown'))}"
            for slot, short in (
                ("helmet", "H"),
                ("vest", "V"),
                ("gloves", "G"),
                ("goggles", "E"),
                ("mask", "M"),
            )
        )
        _draw_box(
            canvas,
            scaled,
            color,
            f"#{track_id} {_status_text(status)} | {summary}",
            scale,
            base + 1,
        )

    for detection in detections:
        scaled = [value * scale_factor for value in detection["bbox"]]
        category = detection.get("category")
        if POLARITY.get(category) == -1:
            color = COLOR_NEGATIVE
        elif POLARITY.get(category) == 1:
            color = COLOR_POSITIVE
        else:
            color = COLOR_OTHER
        owner = detection.get("assigned_person")
        suffix = f" -> #{owner}" if owner else ""
        _draw_box(
            canvas,
            scaled,
            color,
            f"{detection['label']} {detection['confidence']:.2f}{suffix}",
            max(0.36, scale * 0.8),
            max(1, base - 1),
        )

    stamp = f"{int(timestamp // 60):02d}:{timestamp % 60:04.1f}"
    _draw_label(canvas, 10, 26, f"SiteGuardVision | {stamp}", COLOR_PANEL, 0.62, 2)
    cv2.putText(
        canvas,
        stamp,
        (16, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        COLOR_TEXT,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Decision support - verify every flag",
        (16, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLOR_ACCENT,
        1,
        cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _output_dimensions(width: int, height: int, max_width: int) -> tuple[int, int, float]:
    if width <= 0 or height <= 0:
        raise VideoProcessingError("The video reports invalid dimensions.")
    scale = min(1.0, float(max_width) / float(max(width, height)))
    out_width = max(2, int(round(width * scale)))
    out_height = max(2, int(round(height * scale)))
    if out_width % 2:
        out_width -= 1
    if out_height % 2:
        out_height -= 1
    return out_width, out_height, out_width / float(width)


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> tuple[cv2.VideoWriter, str]:
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*codec), fps, size
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
    raise VideoProcessingError("Could not create the annotated video encoder.")


def process_video(
    source_path: str | Path,
    output_path: str | Path,
    settings: VideoSettings | None = None,
    progress: Callable[[float, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Analyse one video and write an annotated MP4.

    Returns a serialisable result containing real metadata, tracked workers,
    timestamped events, summary counts, and processing settings.
    """
    settings = settings or VideoSettings()
    source_path = Path(source_path)
    output_path = Path(output_path)

    probe = probe_video(source_path)
    if not probe["ok"]:
        raise VideoProcessingError(probe.get("error") or "The video could not be opened.")

    plan = plan_sampling(probe["fps"], probe["frame_count"], settings)
    stride = int(plan["stride"])
    expected = int(plan["estimated_frames"])
    fps = float(probe["fps"])

    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise VideoProcessingError("OpenCV could not open the uploaded video.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    codec_used = ""
    temporal = TemporalEventEngine(settings)
    fallback = _FallbackIds()
    track_id_map: dict[int, int] = {}
    reset_tracker()

    started = time.perf_counter()
    frame_index = 0
    processed = 0
    corrupt_frames = 0
    output_size: tuple[int, int] | None = None
    output_scale = 1.0
    last_reported = -1
    notes: list[str] = []

    def report(percent: float, message: str) -> None:
        nonlocal last_reported
        if progress is None:
            return
        rounded = int(max(0, min(99, percent)))
        if rounded == last_reported:
            return
        last_reported = rounded
        progress(percent, processed, expected, message)

    try:
        report(0.0, "Opening video and planning frame sampling…")
        while True:
            if not capture.grab():
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            ok, frame = capture.retrieve()
            if not ok or frame is None:
                corrupt_frames += 1
                frame_index += 1
                continue

            if output_size is None:
                out_w, out_h, output_scale = _output_dimensions(
                    frame.shape[1], frame.shape[0], settings.max_output_width
                )
                output_size = (out_w, out_h)
                writer, codec_used = _open_writer(output_path, plan["output_fps"], output_size)

            people = _track_people(frame, settings, fallback)
            # Normalise tracker IDs to a per-job sequence.  ByteTrack may keep
            # increasing its internal counter across jobs; the dashboard must
            # show stable worker IDs starting at #1 for every video.
            for person in people:
                raw_id = int(person["track_id"])
                if raw_id not in track_id_map:
                    track_id_map[raw_id] = len(track_id_map) + 1
                person["track_id"] = track_id_map[raw_id]

            detections = detect_ppe(frame, settings)
            person_boxes = [person["bbox"] for person in people]
            detections = associate_frame(person_boxes, detections)

            states_by_track: dict[int, dict[str, dict[str, Any]]] = {}
            status_by_track: dict[int, str] = {}
            timestamp = frame_index / fps
            for person_index, person in enumerate(people, start=1):
                track_id = int(person["track_id"])
                # worker_states expects the 1-based association index.
                states = worker_states(person_index, detections)
                states_by_track[track_id] = states
                status_by_track[track_id] = frame_status(states)
                temporal.observe(track_id, timestamp, frame_index, states)

            resized = cv2.resize(frame, output_size, interpolation=cv2.INTER_AREA)
            annotated = annotate_frame(
                resized,
                timestamp,
                people,
                detections,
                states_by_track,
                status_by_track,
                output_scale,
            )
            writer.write(annotated)
            processed += 1
            percent = (processed / expected * 100.0) if expected else 0.0
            report(percent, f"Processed {processed} sampled frame(s)…")

            if processed >= settings.max_sampled_frames:
                notes.append(
                    f"Stopped after the configured limit of {settings.max_sampled_frames} sampled frames."
                )
                break

            frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    if writer is None or output_size is None:
        raise VideoProcessingError("No readable frames were found in the video.")

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise VideoProcessingError("The annotated video could not be written.")

    # Verify the file is actually decodable before reporting success.
    check = cv2.VideoCapture(str(output_path))
    try:
        output_frames = int(check.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        readable = bool(check.read()[0])
    finally:
        check.release()
    if not readable:
        raise VideoProcessingError("The annotated video was written but cannot be decoded.")

    events, workers = temporal.finalize()
    summary = summarize_workers(workers, events)
    if not workers:
        notes.append("No workers were tracked in the sampled frames.")
    if corrupt_frames:
        notes.append(f"{corrupt_frames} corrupt frame(s) were skipped.")
    if any(
        worker["states"].get(slot) == "unconfirmed"
        for worker in workers
        for slot in ("gloves", "goggles", "mask")
    ):
        notes.append(
            "Gloves, goggles, and masks are assessed only when the checkpoint emits "
            "matching boxes; otherwise they stay Unconfirmed."
        )

    elapsed = round(time.perf_counter() - started, 2)
    result = {
        "video": {
            "filename": source_path.name,
            "width": probe["width"],
            "height": probe["height"],
            "fps": probe["fps"],
            "frame_count": probe["frame_count"],
            "duration_s": probe["duration_s"],
        },
        "processing": {
            **plan,
            "imgsz": settings.imgsz,
            "conf": settings.conf,
            "max_output_width": settings.max_output_width,
            "processed_frames": processed,
            "corrupt_frames": corrupt_frames,
            "elapsed_s": elapsed,
            "codec": codec_used,
            "output_fps": plan["output_fps"],
        },
        "annotated": {
            "filename": output_path.name,
            "width": output_size[0],
            "height": output_size[1],
            "fps": plan["output_fps"],
            "frame_count": output_frames,
            "size_bytes": int(output_path.stat().st_size),
        },
        "workers": workers,
        "events": events,
        "summary": summary,
        "settings": settings.as_dict(),
        "notes": notes,
    }
    report(100.0, "Video analysis complete.")
    return result
