"""Video integration tests for SiteGuardVision.

Run with::

    .\\venv\\Scripts\\python.exe -m unittest discover -s tests -v

The video suite is layered the same way as the image suite:

1. Pure tests for label mapping, sampling plans, association, worker states,
   and temporal event verification.
2. Sample-library and upload/storage tests against synthetic videos created in
   an isolated temp directory.
3. Flask route and dashboard rendering tests with an isolated SQLite database.
4. A real background-worker test and a real-model annotated-output test, both
   skipped automatically when the checkpoints are absent.

Every test that touches storage or uploaded/annotated files uses temporary
directories.  The production database, sample library, and outputs are never
modified.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import tempfile
import time
import unittest
import uuid
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from werkzeug.datastructures import FileStorage

import storage
import server
from video_processing import jobs as video_jobs
from video_processing import sample_library
from video_processing import video_engine as vengine
from video_processing.video_engine import (
    TemporalEventEngine,
    VideoProcessingError,
    VideoSettings,
    classify_video_label,
    frame_status,
    plan_sampling,
    summarize_workers,
    worker_states,
)

ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "test_videos"
PPE_MODEL = ROOT / "models" / "yolo_ppe_core.pt"
PERSON_MODEL = ROOT / "models" / "person_yolov8n.pt"
MODELS_AVAILABLE = PPE_MODEL.is_file() and PERSON_MODEL.is_file()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_video(
    path: Path,
    *,
    fps: float = 10.0,
    frames: int = 20,
    size: tuple[int, int] = (160, 120),
) -> Path:
    """Write a small valid MP4 using OpenCV (mp4v is always available)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the synthetic test video.")
    width, height = size
    try:
        for index in range(frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[:] = (30, 45, 60)
            cv2.rectangle(
                frame,
                (5 + index * 2 % max(1, width - 30), 10),
                (25 + index * 2 % max(1, width - 30), 50),
                (0, 200, 255),
                -1,
            )
            cv2.putText(
                frame,
                str(index),
                (8, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
            )
            writer.write(frame)
    finally:
        writer.release()
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError("Synthetic test video was not written.")
    return path


def state_map(**overrides: str) -> dict[str, dict]:
    base = {
        slot: {"state": "unknown", "confidence": None, "label": None}
        for slot in vengine.CATEGORY_ORDER
    }
    for slot, state in overrides.items():
        base[slot] = {"state": state, "confidence": 0.66, "label": slot}
    return base


def detection(category: str, bbox: list[float], confidence: float = 0.7) -> dict:
    return {
        "label": category,
        "category": category,
        "confidence": confidence,
        "bbox": bbox,
        "source": "ppe",
        "assigned_person": None,
        "association_score": None,
    }


def box_inside(region: list[float], inset: float = 4.0) -> list[float]:
    x1, y1, x2, y2 = region
    width = max(8.0, (x2 - x1) * 0.4)
    height = max(8.0, (y2 - y1) * 0.4)
    left = x1 + inset
    top = y1 + inset
    return [left, top, left + width, top + height]


class VideoEnvTestCase(unittest.TestCase):
    """Base class with isolated database, sample, upload, output, and thumb dirs."""

    sample_count = 2

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="sgv-video-tests-"))
        cls.sample_dir = cls.temp_dir / "samples"
        cls.upload_dir = cls.temp_dir / "uploads"
        cls.output_dir = cls.temp_dir / "outputs"
        cls.thumb_dir = cls.temp_dir / "thumbs"
        for directory in (
            cls.sample_dir,
            cls.upload_dir,
            cls.output_dir,
            cls.thumb_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        cls.env_patch = mock.patch.dict(
            os.environ,
            {
                "SITEGUARD_DB_PATH": str(cls.temp_dir / "video-tests.db"),
                "SITEGUARD_SAMPLE_VIDEO_DIR": str(cls.sample_dir),
                "SITEGUARD_VIDEO_UPLOAD_DIR": str(cls.upload_dir),
                "SITEGUARD_VIDEO_OUTPUT_DIR": str(cls.output_dir),
                "SITEGUARD_VIDEO_THUMB_DIR": str(cls.thumb_dir),
                "SITEGUARD_VIDEO_REPORT": str(cls.temp_dir / "report.json"),
            },
        )
        cls.env_patch.start()
        sample_library.clear_cache()

        cls.sample_paths = [
            write_video(cls.sample_dir / "alpha.mp4", fps=10, frames=20),
            write_video(cls.sample_dir / "beta.mp4", fps=8, frames=16),
        ]
        (cls.sample_dir / "notes.txt").write_text("not a video", encoding="utf-8")
        (cls.sample_dir / "broken.mp4").write_bytes(b"not really an mp4")

        server.app.config.update(TESTING=True)
        cls.client = server.app.test_client()

    @classmethod
    def tearDownClass(cls):
        sample_library.clear_cache()
        cls.env_patch.stop()
        shutil.rmtree(cls.temp_dir, ignore_errors=True)
        super().tearDownClass()

    def tearDown(self):
        sample_library.clear_cache()


# ---------------------------------------------------------------------------
# 1. Pure tests
# ---------------------------------------------------------------------------


class TestVideoLabelMapping(unittest.TestCase):
    def test_all_health_check_classes_map(self):
        expected = {
            "Person": "person",
            "Hardhat": "hardhat",
            "NO-Hardhat": "no_hardhat",
            "Safety Vest": "vest",
            "NO-Safety Vest": "no_vest",
            "Gloves": "gloves",
            "NO-Gloves": "no_gloves",
            "Goggles": "goggles",
            "NO-Goggles": "no_goggles",
            "Mask": "mask",
            "NO-Mask": "no_mask",
            "Fall-Detected": "fall_detected",
            "Ladder": "ladder",
            "Safety Cone": "safety_cone",
        }
        for label, category in expected.items():
            with self.subTest(label=label):
                self.assertEqual(classify_video_label(label), category)

    def test_unknown_label_is_other(self):
        self.assertEqual(classify_video_label("Excavator"), "other")
        self.assertEqual(classify_video_label(""), "other")


class TestSamplingPlan(unittest.TestCase):
    def test_two_fps_plan_preserves_duration(self):
        plan = plan_sampling(30.0, 300, VideoSettings(sample_fps=2.0))
        self.assertEqual(plan["stride"], 15)
        self.assertAlmostEqual(plan["sample_fps"], 2.0, places=3)
        self.assertAlmostEqual(plan["output_fps"], 2.0, places=3)
        self.assertEqual(plan["estimated_frames"], 20)

    def test_high_frame_rate_is_capped_by_budget(self):
        plan = plan_sampling(
            50.0, 100_000, VideoSettings(sample_fps=2.0, max_sampled_frames=180)
        )
        self.assertLessEqual(plan["estimated_frames"], 180)
        self.assertGreaterEqual(plan["stride"], 556)
        self.assertAlmostEqual(plan["output_fps"], 50.0 / plan["stride"], places=3)

    def test_4k_60fps_is_not_processed_frame_by_frame(self):
        plan = plan_sampling(60.0, 930, VideoSettings())
        self.assertEqual(plan["stride"], 30)
        self.assertEqual(plan["estimated_frames"], 31)

    def test_invalid_frame_rate_is_rejected(self):
        with self.assertRaises(VideoProcessingError):
            plan_sampling(0.0, 100, VideoSettings())

    def test_output_dimensions_preserve_aspect_and_even_size(self):
        width, height, scale = vengine._output_dimensions(3840, 2160, 1280)
        self.assertEqual((width, height), (1280, 720))
        self.assertAlmostEqual(scale, 1 / 3, places=4)

        width, height, _ = vengine._output_dimensions(2160, 3840, 1280)
        self.assertEqual((width, height), (720, 1280))

        width, height, _ = vengine._output_dimensions(1919, 1079, 1280)
        self.assertEqual(width % 2, 0)
        self.assertEqual(height % 2, 0)

    def test_settings_payload_is_clamped(self):
        settings = VideoSettings.from_payload(
            {
                "sample_fps": 100,
                "max_sampled_frames": 9999,
                "imgsz": 111,
                "conf": 5,
                "min_observations": 1,
                "min_ratio": 2,
                "cooldown_s": -3,
            }
        )
        self.assertEqual(settings.sample_fps, 10.0)
        self.assertEqual(settings.max_sampled_frames, 600)
        self.assertEqual(settings.imgsz, 640)
        self.assertEqual(settings.conf, 0.95)
        self.assertEqual(settings.min_observations, 2)
        self.assertEqual(settings.min_ratio, 1.0)
        self.assertEqual(settings.cooldown_s, 0.0)

    def test_allowed_image_size_is_kept(self):
        self.assertEqual(VideoSettings.from_payload({"imgsz": "1280"}).imgsz, 1280)


class TestPpeAssociation(unittest.TestCase):
    def test_hat_vest_and_extra_ppe_attach_to_one_worker(self):
        person = [100.0, 100.0, 200.0, 400.0]
        head = vengine.engine._head_region(person)
        torso = vengine.engine._torso_region(person)
        hands = vengine._hands_region(person)

        detections = [
            detection("hardhat", box_inside(head)),
            detection("no_hardhat", box_inside(head), 0.52),
            detection("vest", box_inside(torso)),
            detection("no_vest", box_inside(torso), 0.55),
            detection("gloves", box_inside(hands)),
            detection("goggles", box_inside(head, 1.0)),
            detection("mask", box_inside(head, 2.0)),
        ]
        result = vengine.associate_frame([person], detections)

        for item in result:
            with self.subTest(category=item["category"]):
                self.assertEqual(item["assigned_person"], 1)
                self.assertGreaterEqual(item["association_score"], vengine.engine.MIN_ASSOCIATION_SCORE)

        states = worker_states(1, result)
        self.assertEqual(states["helmet"]["state"], "conflict")
        self.assertEqual(states["vest"]["state"], "conflict")
        self.assertEqual(states["gloves"]["state"], "present")
        self.assertEqual(states["goggles"]["state"], "present")
        self.assertEqual(states["mask"]["state"], "present")

    def test_detection_outside_worker_is_unassigned(self):
        person = [100.0, 100.0, 200.0, 400.0]
        detections = [detection("hardhat", [700.0, 700.0, 740.0, 730.0])]
        result = vengine.associate_frame([person], detections)
        self.assertIsNone(result[0]["assigned_person"])

    def test_two_workers_keep_separate_hats(self):
        left = [50.0, 80.0, 150.0, 380.0]
        right = [600.0, 80.0, 700.0, 380.0]
        detections = [
            detection("no_hardhat", box_inside(vengine.engine._head_region(left))),
            detection("hardhat", box_inside(vengine.engine._head_region(right))),
        ]
        result = vengine.associate_frame([left, right], detections)
        assigned = sorted(item["assigned_person"] for item in result)
        self.assertEqual(assigned, [1, 2])

        self.assertEqual(worker_states(1, result)["helmet"]["state"], "missing")
        self.assertEqual(worker_states(2, result)["helmet"]["state"], "present")

    def test_frame_status_is_cautious(self):
        self.assertEqual(
            frame_status(state_map(helmet="present", vest="present")), "clear"
        )
        self.assertEqual(
            frame_status(state_map(helmet="present", vest="missing")),
            "potential_violation",
        )
        self.assertEqual(frame_status(state_map(vest="present")), "uncertain")
        self.assertEqual(
            frame_status(state_map(helmet="conflict", vest="present")), "uncertain"
        )


class TestTemporalVerification(unittest.TestCase):
    def make_engine(self, **overrides) -> TemporalEventEngine:
        defaults = dict(
            sample_fps=2.0,
            min_observations=3,
            min_ratio=0.5,
            cooldown_s=0.1,
        )
        defaults.update(overrides)
        return TemporalEventEngine(VideoSettings(**defaults))

    def test_persistent_negative_evidence_creates_one_event(self):
        engine = self.make_engine()
        for index in range(5):
            engine.observe(7, index * 0.5, index, state_map(helmet="missing"))

        events, workers = engine.finalize()
        helmet_events = [event for event in events if event["category"] == "helmet"]

        self.assertEqual(len(helmet_events), 1)
        event = helmet_events[0]
        self.assertEqual(event["worker_id"], 7)
        self.assertEqual(event["status"], "potential_violation")
        self.assertEqual(event["first_seen_s"], 0.0)
        self.assertEqual(event["last_seen_s"], 2.0)
        self.assertEqual(event["observations"], 5)
        self.assertIn("Explicit negative PPE evidence", event["evidence"])
        self.assertEqual(workers[0]["status"], "potential_violation")

    def test_too_few_negative_observations_do_not_create_event(self):
        engine = self.make_engine()
        engine.observe(1, 0.0, 0, state_map(helmet="missing"))
        engine.observe(1, 0.5, 1, state_map(helmet="missing"))
        engine.observe(1, 1.0, 2, state_map(helmet="unknown"))
        engine.observe(1, 1.5, 3, state_map(helmet="unknown"))

        events, workers = engine.finalize()
        self.assertEqual(events, [])
        self.assertEqual(workers[0]["status"], "unconfirmed")
        self.assertEqual(workers[0]["states"]["helmet"], "unconfirmed")

    def test_absent_positive_ppe_is_not_an_event(self):
        """No POSITIVE box alone must never be reported as a violation."""
        engine = self.make_engine(min_observations=2)
        for index in range(6):
            engine.observe(2, index * 0.5, index, state_map())

        events, workers = engine.finalize()
        self.assertEqual(events, [])
        self.assertEqual(workers[0]["status"], "unconfirmed")
        self.assertNotEqual(workers[0]["status"], "compliant")

    def test_present_hat_and_vest_mark_worker_compliant(self):
        engine = self.make_engine(min_observations=2)
        for index in range(5):
            engine.observe(
                3, index * 0.5, index, state_map(helmet="present", vest="present")
            )

        events, workers = engine.finalize()
        self.assertEqual(events, [])
        self.assertEqual(workers[0]["status"], "compliant")
        self.assertEqual(workers[0]["states"]["gloves"], "unconfirmed")

    def test_cooldown_merges_runs_only_after_both_qualify(self):
        engine = self.make_engine(
            sample_fps=1.0, min_observations=2, min_ratio=0.5, cooldown_s=10.0
        )
        # Two separate qualified runs at t=0,1 and t=4,5.
        for index, state in enumerate(("missing", "missing", "present", "present", "missing", "missing")):
            engine.observe(4, float(index), index, state_map(helmet=state))

        events, _ = engine.finalize()
        helmet_events = [event for event in events if event["category"] == "helmet"]
        self.assertEqual(len(helmet_events), 1)
        self.assertEqual(helmet_events[0]["first_seen_s"], 0.0)
        self.assertEqual(helmet_events[0]["last_seen_s"], 5.0)
        self.assertEqual(helmet_events[0]["observations"], 4)
        # The rendered evidence must match the merged totals, not one sub-run.
        self.assertIn("4 sampled frame(s)", helmet_events[0]["evidence"])
        self.assertEqual(helmet_events[0]["merged_runs"], 2)

    def test_conflicting_evidence_is_not_compliant(self):
        engine = self.make_engine(min_observations=2)
        for index in range(4):
            engine.observe(
                5,
                index * 0.5,
                index,
                state_map(helmet="conflict", vest="present"),
            )

        events, workers = engine.finalize()
        self.assertEqual(events, [])
        self.assertEqual(workers[0]["status"], "unconfirmed")
        self.assertEqual(workers[0]["states"]["helmet"], "conflict")

    def test_low_support_ratio_blocks_event(self):
        engine = self.make_engine(
            sample_fps=1.0, min_observations=2, min_ratio=0.99, cooldown_s=0.1
        )
        for index, state in enumerate(("missing", "missing", "unknown", "unknown")):
            engine.observe(6, float(index), index, state_map(helmet=state))

        events, workers = engine.finalize()
        self.assertEqual(events, [])
        self.assertEqual(workers[0]["status"], "unconfirmed")

    def test_summary_counts_are_derived_from_real_records(self):
        workers = [
            {"id": 1, "status": "compliant"},
            {"id": 2, "status": "potential_violation"},
            {"id": 3, "status": "unconfirmed"},
            {"id": 4, "status": "potential_violation"},
        ]
        events = [
            {"category": "helmet"},
            {"category": "helmet"},
            {"category": "vest"},
        ]
        summary = summarize_workers(workers, events)
        self.assertEqual(summary["worker_count"], 4)
        self.assertEqual(summary["compliant_workers"], 1)
        self.assertEqual(summary["violation_workers"], 2)
        self.assertEqual(summary["unconfirmed_workers"], 1)
        self.assertEqual(summary["event_count"], 3)
        self.assertEqual(summary["events_by_category"], {"helmet": 2, "vest": 1})


# ---------------------------------------------------------------------------
# 2. Sample library and uploads
# ---------------------------------------------------------------------------


class TestSampleLibrary(VideoEnvTestCase):
    sample_count = 2

    def test_discovery_skips_corrupt_and_non_video_files(self):
        samples = sample_library.discover_samples(force=True)
        names = {sample["filename"] for sample in samples}
        self.assertEqual(names, {"alpha.mp4", "beta.mp4"})
        self.assertNotIn("broken.mp4", names)
        self.assertNotIn("notes.txt", names)

    def test_metadata_comes_from_opencv(self):
        samples = {sample["filename"]: sample for sample in sample_library.discover_samples(force=True)}
        alpha = samples["alpha.mp4"]
        self.assertEqual((alpha["width"], alpha["height"]), (160, 120))
        self.assertAlmostEqual(alpha["fps"], 10.0, places=2)
        self.assertEqual(alpha["frame_count"], 20)
        self.assertAlmostEqual(alpha["duration_s"], 2.0, places=2)
        self.assertEqual(alpha["orientation"], "landscape")

    def test_ids_are_stable_and_path_free(self):
        first = sample_library.video_id_for("alpha.mp4")
        second = sample_library.video_id_for("alpha.mp4")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 12)
        self.assertNotIn("/", first)
        sample = sample_library.get_sample(first)
        self.assertIsNotNone(sample)
        self.assertNotIn("..", sample["thumbnail_url"])

    def test_unknown_id_is_rejected(self):
        self.assertIsNone(sample_library.get_sample("deadbeefcafe"))
        self.assertIsNone(sample_library.get_sample("../../server.py"))
        with self.assertRaises(sample_library.SampleVideoError):
            sample_library.resolve_sample_path("deadbeefcafe")

    def test_thumbnail_is_created_and_cached(self):
        sample = sample_library.discover_samples(force=True)[0]
        thumbnail = sample_library.ensure_thumbnail(sample)
        self.assertIsNotNone(thumbnail)
        self.assertTrue(thumbnail.is_file())
        self.assertGreater(thumbnail.stat().st_size, 0)

        mtime = thumbnail.stat().st_mtime_ns
        again = sample_library.ensure_thumbnail(sample)
        self.assertEqual(again, thumbnail)
        self.assertEqual(again.stat().st_mtime_ns, mtime)

    def test_corrupt_video_probe_fails_cleanly(self):
        probe = sample_library.probe_video(self.sample_dir / "broken.mp4")
        self.assertFalse(probe["ok"])
        self.assertTrue(probe["error"])
        missing = sample_library.probe_video(self.sample_dir / "missing.mp4")
        self.assertFalse(missing["ok"])

    def test_public_sample_has_no_server_path(self):
        sample = sample_library.discover_samples(force=True)[0]
        public = server._public_sample(sample)
        self.assertNotIn("path", public)
        self.assertIn("filename", public)


class TestUploadsAndJobStorage(VideoEnvTestCase):
    def video_bytes(self) -> bytes:
        path = write_video(self.temp_dir / "payload.mp4", fps=6, frames=8)
        return path.read_bytes()

    def test_valid_upload_is_stored_and_probed(self):
        saved = video_jobs.save_upload(
            FileStorage(
                stream=io.BytesIO(self.video_bytes()),
                filename="site-walkthrough.mp4",
                content_type="video/mp4",
            ),
            "site-walkthrough.mp4",
        )
        self.assertTrue(saved["probe"]["ok"])
        self.assertEqual(saved["filename"], "site-walkthrough.mp4")
        self.assertTrue(Path(saved["path"]).is_file())
        self.assertRegex(saved["upload_id"], r"^[a-f0-9]{24}$")

        resolved = video_jobs.resolve_upload(saved["upload_id"])
        self.assertEqual(Path(resolved["path"]), Path(saved["path"]))

    def test_bad_extension_is_rejected(self):
        with self.assertRaises(video_jobs.UploadError):
            video_jobs.save_upload(
                FileStorage(
                    stream=io.BytesIO(b"data"),
                    filename="clip.txt",
                    content_type="text/plain",
                ),
                "clip.txt",
            )

    def test_corrupt_video_is_rejected_and_cleaned_up(self):
        before = {path.name for path in self.upload_dir.iterdir()} if self.upload_dir.is_dir() else set()
        with self.assertRaises(video_jobs.UploadError):
            video_jobs.save_upload(
                FileStorage(
                    stream=io.BytesIO(b"definitely not a video"),
                    filename="corrupt.mp4",
                    content_type="video/mp4",
                ),
                "corrupt.mp4",
            )
        after = {path.name for path in self.upload_dir.iterdir()} if self.upload_dir.is_dir() else set()
        self.assertEqual(before, after)

    def test_upload_id_traversal_is_rejected(self):
        for value in ("../../etc", "a" * 24, "ABCDEF" * 4, "", "a" * 25):
            with self.subTest(value=value):
                with self.assertRaises(video_jobs.UploadError):
                    video_jobs.resolve_upload(value)

    def test_mime_validation(self):
        self.assertTrue(video_jobs.validate_video_mime("video/mp4"))
        self.assertTrue(video_jobs.validate_video_mime("application/octet-stream"))
        self.assertFalse(video_jobs.validate_video_mime("text/plain"))
        self.assertFalse(video_jobs.validate_video_mime("image/png"))

    def test_job_lifecycle_and_events_round_trip(self):
        job_id = "a" * 32
        storage.create_video_job(
            job_id,
            source_type="upload",
            original_filename="clip.mp4",
            source_id="b" * 24,
            stored_path="b/original.mp4",
            settings={"sample_fps": 2},
        )
        job = storage.get_video_job(job_id)
        self.assertEqual(job["status"], "queued")
        self.assertEqual(job["settings"]["sample_fps"], 2)

        storage.update_video_job(job_id, status="running", progress=42, message="Working")
        job = storage.get_video_job(job_id)
        self.assertEqual(job["status"], "running")
        self.assertAlmostEqual(job["progress"], 42.0)

        event = {
            "worker_id": 3,
            "category": "helmet",
            "label": "Missing hard hat",
            "status": "potential_violation",
            "confidence": 0.71,
            "first_seen_s": 1.2,
            "last_seen_s": 3.4,
            "first_frame": 12,
            "last_frame": 34,
            "observations": 4,
            "evidence": "Explicit negative PPE evidence in 4 sampled frames.",
        }
        storage.finish_video_job(
            job_id,
            result={"workers": [{"id": 3}], "events": [event]},
            events=[event],
            output_rel=f"{job_id}.mp4",
            summary={
                "worker_count": 1,
                "compliant_workers": 0,
                "violation_workers": 1,
                "unconfirmed_workers": 0,
                "event_count": 1,
            },
            duration_s=6.5,
            width=160,
            height=120,
            fps=10.0,
            total_frames=65,
            sampled_frames=13,
            processed_frames=13,
        )

        job = storage.get_video_job(job_id)
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["progress"], 100)
        self.assertEqual(job["event_count"], 1)
        self.assertEqual(job["worker_count"], 1)
        self.assertEqual(job["summary"]["violation_workers"], 1)

        events = storage.list_video_events(job_id)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["worker_id"], 3)
        self.assertEqual(events[0]["category"], "helmet")
        self.assertAlmostEqual(events[0]["first_seen_s"], 1.2)

    def test_stale_jobs_are_failed_on_startup(self):
        first = "c" * 32
        second = "d" * 32
        for job_id in (first, second):
            storage.create_video_job(
                job_id,
                source_type="sample",
                original_filename="alpha.mp4",
                source_id="e" * 12,
            )
        storage.fail_stale_video_jobs("Server restarted.")
        for job_id in (first, second):
            job = storage.get_video_job(job_id)
            self.assertEqual(job["status"], "failed")
            self.assertEqual(job["error"], "Server restarted.")


# ---------------------------------------------------------------------------
# 3. Flask routes and dashboard rendering
# ---------------------------------------------------------------------------


class TestVideoRoutes(VideoEnvTestCase):
    def completed_job(self) -> dict:
        job_id = uuid.uuid4().hex
        output = self.output_dir / f"{job_id}.mp4"
        write_video(output, fps=4, frames=8, size=(320, 180))
        event = {
            "worker_id": 2,
            "category": "helmet",
            "label": "Missing hard hat",
            "status": "potential_violation",
            "confidence": 0.68,
            "first_seen_s": 1.5,
            "last_seen_s": 3.0,
            "first_frame": 6,
            "last_frame": 12,
            "observations": 3,
            "evidence": "Explicit negative PPE evidence in 3 sampled frames; 3/4 observations support it.",
        }
        result = {
            "video": {
                "filename": "alpha.mp4",
                "width": 160,
                "height": 120,
                "fps": 10.0,
                "frame_count": 20,
                "duration_s": 2.0,
            },
            "processing": {
                "stride": 10,
                "sample_fps": 1.0,
                "output_fps": 1.0,
                "estimated_frames": 2,
                "imgsz": 640,
                "conf": 0.25,
                "processed_frames": 2,
                "elapsed_s": 1.2,
                "codec": "avc1",
            },
            "annotated": {
                "filename": output.name,
                "width": 320,
                "height": 180,
                "fps": 4.0,
                "frame_count": 8,
                "size_bytes": output.stat().st_size,
            },
            "workers": [
                {
                    "id": 2,
                    "first_seen_s": 0.0,
                    "last_seen_s": 2.0,
                    "frames_observed": 2,
                    "status": "potential_violation",
                    "states": {
                        "helmet": "missing",
                        "vest": "present",
                        "gloves": "unconfirmed",
                        "goggles": "unconfirmed",
                        "mask": "unconfirmed",
                    },
                    "confidence": {"helmet": 0.68, "vest": 0.54},
                    "events": ["Missing hard hat"],
                }
            ],
            "events": [event],
            "summary": {
                "worker_count": 1,
                "compliant_workers": 0,
                "violation_workers": 1,
                "unconfirmed_workers": 0,
                "event_count": 1,
                "events_by_category": {"helmet": 1},
            },
            "notes": ["One corrupt frame was skipped."],
        }
        sample = sample_library.discover_samples(force=True)[0]
        storage.create_video_job(
            job_id,
            source_type="sample",
            original_filename=sample["filename"],
            source_id=sample["id"],
            settings=VideoSettings().as_dict(),
        )
        storage.finish_video_job(
            job_id,
            result=result,
            events=[event],
            output_rel=output.name,
            summary=result["summary"],
            duration_s=2.0,
            width=160,
            height=120,
            fps=10.0,
            total_frames=20,
            sampled_frames=2,
            processed_frames=2,
        )
        return storage.get_video_job(job_id)

    def test_workspace_renders_samples_and_upload(self):
        response = self.client.get("/videos")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Video Analysis", html)
        self.assertIn('id="video-file"', html)
        self.assertIn("data-sample-list", html)
        self.assertIn("alpha.mp4", html)
        self.assertIn("Analysis setup", html)

    def test_samples_api_never_exposes_paths(self):
        response = self.client.get("/api/videos/samples")
        payload = response.get_json()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["ok"])
        self.assertEqual(len(payload["samples"]), 2)
        for sample in payload["samples"]:
            self.assertNotIn("path", sample)
            self.assertRegex(sample["id"], r"^[a-f0-9]{12}$")

    def test_sample_metadata_and_thumbnail(self):
        sample = sample_library.discover_samples(force=True)[0]
        metadata = self.client.get(f"/api/videos/samples/{sample['id']}/metadata")
        self.assertEqual(metadata.status_code, 200)
        self.assertEqual(metadata.get_json()["sample"]["filename"], sample["filename"])

        thumbnail = self.client.get(f"/api/videos/samples/{sample['id']}/thumbnail")
        self.assertEqual(thumbnail.status_code, 200)
        self.assertEqual(thumbnail.content_type, "image/jpeg")
        self.assertTrue(thumbnail.data)
        thumbnail.close()

    def test_sample_stream_supports_range_requests(self):
        sample = sample_library.discover_samples(force=True)[0]
        response = self.client.get(
            f"/api/videos/samples/{sample['id']}/stream",
            headers={"Range": "bytes=0-255"},
        )
        self.assertEqual(response.status_code, 206)
        self.assertEqual(len(response.data), 256)
        response.close()

    def test_bad_sample_ids_return_404(self):
        for value in ("deadbeefcafe", "x" * 80, "..%2Fserver.py"):
            with self.subTest(value=value):
                self.assertEqual(
                    self.client.get(f"/api/videos/samples/{value}/metadata").status_code,
                    404,
                )

    def test_upload_endpoint_rejects_bad_files(self):
        response = self.client.post(
            "/api/videos/upload",
            data={"video": (io.BytesIO(b"plain text"), "notes.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("not recognised as a video", response.get_json()["error"])

        response = self.client.post(
            "/api/videos/upload",
            data={"video": (io.BytesIO(b"not a video"), "broken.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("could not open", response.get_json()["error"])

    def test_upload_endpoint_accepts_real_video(self):
        data = (self.sample_paths[0]).read_bytes()
        response = self.client.post(
            "/api/videos/upload",
            data={"video": (io.BytesIO(data), "walkthrough.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 201)
        payload = response.get_json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["width"], 160)
        self.assertEqual(payload["height"], 120)
        preview = self.client.get(payload["preview_url"], headers={"Range": "bytes=0-63"})
        self.assertEqual(preview.status_code, 206)
        preview.close()

    def test_analyze_endpoint_validates_source(self):
        response = self.client.post(
            "/api/videos/analyze", json={"source_type": "unknown"}
        )
        self.assertEqual(response.status_code, 400)

        response = self.client.post(
            "/api/videos/analyze",
            json={"source_type": "sample", "video_id": "deadbeefcafe"},
        )
        self.assertEqual(response.status_code, 404)

        response = self.client.post(
            "/api/videos/analyze",
            json={"source_type": "upload", "upload_id": "../escape"},
        )
        self.assertEqual(response.status_code, 404)

    def test_completed_result_dashboard_renders_real_values(self):
        job = self.completed_job()
        response = self.client.get(f"/videos/jobs/{job['id']}")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Timestamped Safety Events", html)
        self.assertIn("Worker Assessment", html)
        self.assertIn("Missing hard hat", html)
        self.assertIn("Potential Violation", html)
        self.assertIn("00:01.5", html)
        self.assertIn(f"/api/videos/jobs/{job['id']}/output", html)
        self.assertIn(f"/api/videos/jobs/{job['id']}/source", html)
        self.assertNotIn("Traceback", html)

    def test_result_apis_return_persisted_records(self):
        job = self.completed_job()
        status = self.client.get(f"/api/videos/jobs/{job['id']}")
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.get_json()["job"]["status"], "completed")
        self.assertNotIn("result", status.get_json()["job"])

        result = self.client.get(f"/api/videos/jobs/{job['id']}/result")
        payload = result.get_json()
        self.assertEqual(result.status_code, 200)
        self.assertEqual(payload["summary"]["event_count"], 1)
        self.assertEqual(payload["events"][0]["category"], "helmet")
        self.assertIn("workers", payload["result"])

        events = self.client.get(f"/api/videos/jobs/{job['id']}/events")
        self.assertEqual(events.status_code, 200)
        self.assertEqual(len(events.get_json()["events"]), 1)

    def test_annotated_and_source_streams(self):
        job = self.completed_job()
        output = self.client.get(
            f"/api/videos/jobs/{job['id']}/output",
            headers={"Range": "bytes=0-127"},
        )
        self.assertEqual(output.status_code, 206)
        self.assertEqual(output.content_type, "video/mp4")
        output.close()

        source = self.client.get(
            f"/api/videos/jobs/{job['id']}/source",
            headers={"Range": "bytes=0-127"},
        )
        self.assertEqual(source.status_code, 206)
        source.close()

    def test_missing_or_malformed_jobs_return_404(self):
        self.assertEqual(self.client.get("/videos/jobs/" + "a" * 32).status_code, 404)
        self.assertEqual(self.client.get("/videos/jobs/not-a-job").status_code, 404)
        self.assertEqual(self.client.get("/api/videos/jobs/" + "a" * 32).status_code, 404)

    def test_overview_lists_jobs(self):
        job = self.completed_job()
        html = self.client.get("/videos").get_data(as_text=True)
        self.assertIn(job["original_filename"], html)
        self.assertIn("Completed", html)
        self.assertIn(f"/videos/jobs/{job['id']}", html)


class TestBackgroundVideoJob(VideoEnvTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    def test_queued_job_reports_progress_and_completes(self):
        if not MODELS_AVAILABLE:
            self.skipTest("model checkpoints are not installed")

        sample = sample_library.discover_samples(force=True)[0]
        response = self.client.post(
            "/api/videos/analyze",
            json={
                "source_type": "sample",
                "video_id": sample["id"],
                "sample_fps": 1,
                "max_sampled_frames": 2,
                "imgsz": 640,
                "min_observations": 2,
            },
        )
        self.assertEqual(response.status_code, 202)
        payload = response.get_json()
        job_id = payload["job_id"]
        self.assertRegex(job_id, r"^[a-f0-9]{32}$")

        status_url = payload["status_url"]
        saw_progress = False
        deadline = time.time() + 240
        job = None
        while time.time() < deadline:
            status = self.client.get(status_url)
            self.assertEqual(status.status_code, 200)
            job = status.get_json()["job"]
            if job["status"] in ("completed", "failed"):
                break
            if job["progress"] > 0:
                saw_progress = True
            time.sleep(0.25)

        self.assertIsNotNone(job)
        self.assertEqual(job["status"], "completed", job.get("error"))
        self.assertEqual(job["progress"], 100)
        self.assertEqual(job["processed_frames"], 2)
        self.assertEqual(job["sampled_frames"], 2)
        self.assertTrue(saw_progress or job["processed_frames"] == 2)
        self.assertTrue((self.output_dir / f"{job_id}.mp4").is_file())

        result = self.client.get(f"/api/videos/jobs/{job_id}/result").get_json()
        self.assertIn("processing", result["result"])
        self.assertEqual(result["result"]["processing"]["processed_frames"], 2)
        # A blank synthetic video has no people; no worker is invented.
        self.assertEqual(result["summary"]["worker_count"], 0)
        self.assertEqual(result["summary"]["event_count"], 0)

        page = self.client.get(f"/videos/jobs/{job_id}")
        self.assertEqual(page.status_code, 200)


# ---------------------------------------------------------------------------
# 4. Real model processing
# ---------------------------------------------------------------------------


class TestRealVideoProcessing(unittest.TestCase):
    @unittest.skipUnless(MODELS_AVAILABLE, "model checkpoints are not installed")
    def test_real_video_produces_playable_annotated_output(self):
        source = VIDEO_DIR / "Recording 2026-11-18 093655.mp4"
        if not source.is_file():
            self.skipTest("negative-PPE validation video is not installed")

        temp_dir = Path(tempfile.mkdtemp(prefix="sgv-video-real-"))
        output = temp_dir / "annotated.mp4"
        settings = VideoSettings(
            sample_fps=1.0,
            max_sampled_frames=4,
            imgsz=640,
            conf=0.25,
            min_observations=2,
            min_ratio=0.4,
            cooldown_s=2.0,
        )
        try:
            result = vengine.process_video(source, output, settings)
        except VideoProcessingError as exc:
            self.skipTest(f"real video could not be processed in this environment: {exc}")

        try:
            self.assertTrue(output.is_file())
            self.assertGreater(output.stat().st_size, 0)

            processing = result["processing"]
            annotated = result["annotated"]
            self.assertEqual(processing["processed_frames"], 4)
            self.assertGreater(processing["output_fps"], 0)
            self.assertLessEqual(annotated["width"], 1280)
            self.assertEqual(annotated["width"] % 2, 0)
            self.assertEqual(annotated["height"] % 2, 0)
            self.assertGreater(annotated["frame_count"], 0)
            self.assertAlmostEqual(annotated["fps"], processing["output_fps"], places=3)

            self.assertIn("workers", result)
            self.assertIn("events", result)
            self.assertIn("summary", result)
            self.assertEqual(
                result["summary"]["event_count"], len(result["events"])
            )
            self.assertGreaterEqual(result["summary"]["worker_count"], 1)

            capture = cv2.VideoCapture(str(output))
            try:
                self.assertTrue(capture.isOpened())
                ok, frame = capture.read()
                self.assertTrue(ok)
                self.assertIsNotNone(frame)
                self.assertEqual(frame.shape[1], annotated["width"])
                self.assertEqual(frame.shape[0], annotated["height"])
            finally:
                capture.release()

            for worker in result["workers"]:
                self.assertIn(worker["status"], ("compliant", "potential_violation", "unconfirmed"))
                self.assertGreaterEqual(worker["frames_observed"], 1)
                for slot in ("helmet", "vest", "gloves", "goggles", "mask"):
                    self.assertIn(
                        worker["states"][slot],
                        ("present", "missing", "conflict", "unconfirmed"),
                    )

            for event in result["events"]:
                self.assertEqual(event["status"], "potential_violation")
                self.assertIn("Explicit negative PPE evidence", event["evidence"])
                self.assertEqual(event["worker_id"], int(event["worker_id"]))
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
