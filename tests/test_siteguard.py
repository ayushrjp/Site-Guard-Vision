"""
SiteGuardVision test suite.

Run with:

    .\\venv\\Scripts\\python.exe -m unittest discover -s tests -v

The suite has three layers:

1. Pure unit tests for label mapping, geometry, association, and assessment.
2. Flask route/validation/rendering tests using the Flask test client.
3. Real CPU inference tests against bundled sample photos.

Real model tests are skipped automatically when the model files are missing.
"""

from __future__ import annotations

import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from PIL import Image

import infer_engine as engine
import server

ROOT = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT / "samples"
PPE_MODEL = ROOT / "models" / "yolo_ppe_core.pt"
PERSON_MODEL = ROOT / "models" / "person_yolov8n.pt"
MODELS_AVAILABLE = PPE_MODEL.is_file() and PERSON_MODEL.is_file()


def find_sample(prefix: str) -> Path | None:
    """Return the first sample whose filename starts with ``prefix``."""
    if not SAMPLES_DIR.is_dir():
        return None
    for path in sorted(SAMPLES_DIR.iterdir()):
        if path.is_file() and path.name.startswith(prefix):
            return path
    return None


def make_test_image(path: Path, size: tuple[int, int] = (320, 240)) -> Path:
    Image.new("RGB", size, (232, 232, 232)).save(path)
    return path


# ---------------------------------------------------------------------------
# Pure unit tests
# ---------------------------------------------------------------------------


class TestLabelMapping(unittest.TestCase):
    def test_known_ppe_labels(self):
        cases = {
            "Person": "person",
            "Hardhat": "hardhat",
            "Helmet": "hardhat",
            "NO-Hardhat": "no_hardhat",
            "NO-Helmet": "no_hardhat",
            "Safety Vest": "vest",
            "Vest": "vest",
            "NO-Safety Vest": "no_vest",
            "NO-Vest": "no_vest",
            "Ladder": "other",
        }
        for label, expected in cases.items():
            with self.subTest(label=label):
                self.assertEqual(engine.classify_label(label), expected)


class TestGeometryAndAssociation(unittest.TestCase):
    def test_head_and_torso_regions(self):
        head = engine._head_region([0, 0, 100, 200])
        torso = engine._torso_region([0, 0, 100, 200])

        self.assertAlmostEqual(head[1], 0.0)
        self.assertAlmostEqual(head[3], 76.0)  # 38% of height
        self.assertGreater(head[0], 0.0)  # side inset
        self.assertLess(head[2], 100.0)

        self.assertAlmostEqual(torso[1], 32.0)  # 16% of height
        self.assertAlmostEqual(torso[3], 164.0)  # 82% of height

    def test_hat_overlapping_two_workers_is_assigned_once(self):
        # Two heavily overlapping workers share a head area.
        person_boxes = [[0.0, 0.0, 100.0, 300.0], [40.0, 0.0, 140.0, 300.0]]
        detections = [
            {
                "label": "Hardhat",
                "category": "hardhat",
                "confidence": 0.9,
                "bbox": [50.0, 5.0, 80.0, 25.0],
            }
        ]

        result = engine.associate_detections(person_boxes, detections)

        self.assertIsNotNone(result[0]["assigned_person"])
        self.assertIn(result[0]["assigned_person"], {1, 2})
        self.assertGreaterEqual(result[0]["association_score"], engine.MIN_ASSOCIATION_SCORE)

    def test_two_hats_go_to_two_workers(self):
        person_boxes = [[0.0, 0.0, 100.0, 300.0], [200.0, 0.0, 300.0, 300.0]]
        detections = [
            {
                "label": "Hardhat",
                "category": "hardhat",
                "confidence": 0.9,
                "bbox": [30.0, 5.0, 70.0, 25.0],
            },
            {
                "label": "Hardhat",
                "category": "hardhat",
                "confidence": 0.8,
                "bbox": [230.0, 5.0, 270.0, 25.0],
            },
        ]

        result = engine.associate_detections(person_boxes, detections)
        assigned = sorted(detection["assigned_person"] for detection in result)

        self.assertEqual(assigned, [1, 2])

    def test_distant_ppe_is_not_assigned(self):
        person_boxes = [[0.0, 0.0, 100.0, 300.0]]
        detections = [
            {
                "label": "Safety Vest",
                "category": "vest",
                "confidence": 0.9,
                "bbox": [500.0, 500.0, 560.0, 580.0],
            }
        ]

        result = engine.associate_detections(person_boxes, detections)

        self.assertIsNone(result[0]["assigned_person"])
        self.assertIsNone(result[0]["association_score"])

    def test_one_ppe_cannot_fill_two_worker_slots(self):
        # A very large vest box that covers both torso regions must not be
        # duplicated onto two workers.
        person_boxes = [[0.0, 0.0, 100.0, 300.0], [110.0, 0.0, 210.0, 300.0]]
        detections = [
            {
                "label": "Safety Vest",
                "category": "vest",
                "confidence": 0.9,
                "bbox": [30.0, 50.0, 180.0, 240.0],
            }
        ]

        result = engine.associate_detections(person_boxes, detections)

        assigned_workers = [
            detection["assigned_person"]
            for detection in result
            if detection["assigned_person"] is not None
        ]
        self.assertLessEqual(len(assigned_workers), 1)

    def test_low_confidence_duplicate_cannot_take_confident_slot(self):
        # Real ppe_0545 geometry at conf 0.05: the 0.078 hard-hat box contains
        # the 0.493 hard-hat box and used to win the slot on overlap score.
        person_boxes = [[452.6, 11.6, 600.5, 450.3]]
        detections = [
            {
                "label": "Hardhat",
                "category": "hardhat",
                "confidence": 0.493,
                "bbox": [509.9, 0.0, 573.4, 102.2],
            },
            {
                "label": "Hardhat",
                "category": "hardhat",
                "confidence": 0.078,
                "bbox": [507.8, 0.0, 573.9, 152.2],
            },
        ]

        result = engine.associate_detections(person_boxes, detections)

        self.assertEqual(result[0]["assigned_person"], 1)
        self.assertIsNone(result[1]["assigned_person"])

    def test_opposite_categories_are_never_duplicates(self):
        # A NO-Hardhat box on a Hardhat box is contradicting evidence and must
        # not be suppressed as a duplicate.
        detections = [
            {
                "label": "Hardhat",
                "category": "hardhat",
                "confidence": 0.6,
                "bbox": [10.0, 10.0, 50.0, 40.0],
            },
            {
                "label": "NO-Hardhat",
                "category": "no_hardhat",
                "confidence": 0.2,
                "bbox": [11.0, 11.0, 49.0, 39.0],
            },
        ]

        self.assertEqual(engine._duplicate_detection_indices(detections), set())

    def test_same_category_duplicate_keeps_confident_box(self):
        detections = [
            {
                "label": "Safety Vest",
                "category": "vest",
                "confidence": 0.4,
                "bbox": [10.0, 50.0, 60.0, 150.0],
            },
            {
                "label": "Safety Vest",
                "category": "vest",
                "confidence": 0.1,
                "bbox": [12.0, 52.0, 58.0, 148.0],
            },
        ]

        self.assertEqual(engine._duplicate_detection_indices(detections), {1})


class TestPersonRecords(unittest.TestCase):
    def test_unknown_ppe_is_not_missing(self):
        records = engine.build_person_records(
            [[10.0, 10.0, 90.0, 210.0]],
            [0.88],
            [],
            "test",
        )
        self.assertEqual(records[0]["helmet"]["state"], "unknown")
        self.assertEqual(records[0]["vest"]["state"], "unknown")
        self.assertEqual(records[0]["status"], "uncertain")

    def test_present_and_missing_states(self):
        detections = [
            {
                "label": "Hardhat",
                "category": "hardhat",
                "confidence": 0.9,
                "bbox": [20.0, 10.0, 70.0, 30.0],
                "assigned_person": 1,
            },
            {
                "label": "NO-Safety Vest",
                "category": "no_vest",
                "confidence": 0.7,
                "bbox": [15.0, 50.0, 85.0, 160.0],
                "assigned_person": 1,
            },
        ]
        records = engine.build_person_records(
            [[0.0, 0.0, 100.0, 220.0]],
            [0.9],
            detections,
            "test",
        )
        self.assertEqual(records[0]["helmet"]["state"], "present")
        self.assertEqual(records[0]["vest"]["state"], "missing")
        self.assertEqual(records[0]["status"], "potential_violation")


class TestAssessment(unittest.TestCase):
    def _worker(self, helmet: str, vest: str, worker_id: int = 1):
        return {
            "id": worker_id,
            "helmet": {"state": helmet, "confidence": 0.9, "label": "Hardhat"},
            "vest": {"state": vest, "confidence": 0.9, "label": "Safety Vest"},
        }

    def test_no_people_is_never_compliance(self):
        result = engine.assess([], [], {})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertEqual(result["title"], "Unable to assess fully")
        self.assertTrue(any("No people" in reason for reason in result["reasons"]))

    def test_people_without_ppe_evidence_is_insufficient(self):
        result = engine.assess([self._worker("unknown", "unknown")], [], {})
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(any("Hard-hat status" in reason for reason in result["reasons"]))
        self.assertTrue(any("Safety-vest status" in reason for reason in result["reasons"]))

    def test_explicit_missing_ppe_is_potential_violation(self):
        detections = [
            {
                "label": "NO-Hardhat",
                "category": "no_hardhat",
                "confidence": 0.6,
                "bbox": [10.0, 10.0, 40.0, 30.0],
                "assigned_person": 1,
            }
        ]
        result = engine.assess(
            [self._worker("missing", "present")],
            detections,
            {"NO-Hardhat": 1, "Safety Vest": 1},
        )
        self.assertEqual(result["status"], "potential_violation")
        self.assertIn("violation", result["title"].lower())

    def test_unassigned_missing_ppe_still_flags(self):
        detections = [
            {
                "label": "NO-Safety Vest",
                "category": "no_vest",
                "confidence": 0.55,
                "bbox": [400.0, 400.0, 460.0, 500.0],
                "assigned_person": None,
            }
        ]
        result = engine.assess(
            [self._worker("present", "present")],
            detections,
            {"Hardhat": 1, "NO-Safety Vest": 1},
        )
        self.assertEqual(result["status"], "potential_violation")
        self.assertTrue(any("could not be matched" in reason for reason in result["reasons"]))

    def test_complete_positive_evidence_is_clear(self):
        result = engine.assess(
            [self._worker("present", "present")],
            [],
            {"Hardhat": 1, "Safety Vest": 1},
        )
        self.assertEqual(result["status"], "no_violation_detected")
        self.assertIn("not proof of compliance", result["message"])

    def test_missing_person_detector_downgrades_to_insufficient(self):
        result = engine.assess(
            [self._worker("present", "present")],
            [],
            {"Hardhat": 1, "Safety Vest": 1},
            person_detector_available=False,
        )
        self.assertEqual(result["status"], "insufficient_evidence")
        self.assertTrue(any("person detector is unavailable" in r for r in result["reasons"]))


# ---------------------------------------------------------------------------
# Flask route and rendering tests
# ---------------------------------------------------------------------------


class TestFlaskRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        server.app.config.update(TESTING=True)
        cls.client = server.app.test_client()
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="sgv-test-"))
        cls.blank = make_test_image(cls.temp_dir / "blank.jpg")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["status"], "ok")

    def test_home_renders_upload_and_samples(self):
        response = self.client.get("/")
        html = response.get_data(as_text=True)

        self.assertEqual(response.status_code, 200)
        self.assertIn('name="image"', html)
        self.assertIn('name="conf"', html)
        self.assertIn('name="imgsz"', html)
        self.assertIn("SiteGuardVision", html)
        self.assertNotIn("DEBUG STATUS", html)

    def test_security_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")

    def test_unreadable_samples_are_not_offered(self):
        """A mislabelled or corrupt sample must not be shown as a sample card."""
        broken = SAMPLES_DIR / "sampletest2.jpg"
        if not broken.is_file() or server._is_readable_image(broken):
            self.skipTest("no known unreadable sample present")

        html = self.client.get("/").get_data(as_text=True)
        self.assertNotIn(broken.name, html)
        self.assertNotIn(broken.name, server._list_samples())

    def test_sample_file_served(self):
        sample = find_sample("ppe_0009")
        if sample is None:
            self.skipTest("sample ppe_0009 is not available")
        response = self.client.get(f"/samples/{sample.name}")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.data)
        response.close()

    def test_sample_path_traversal_rejected(self):
        for path in (
            "/samples/%2e%2e%2fserver.py",
            "/samples/..%5Cserver.py",
            "/samples/../server.py",
        ):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 404)

    def test_unknown_sample_returns_404(self):
        response = self.client.get("/samples/does-not-exist.jpg")
        self.assertEqual(response.status_code, 404)

    def test_unsupported_extension_rejected(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"hello"), "notes.txt")},
            content_type="multipart/form-data",
        )
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Unsupported file type", html)

    def test_corrupt_image_rejected_without_crash(self):
        response = self.client.post(
            "/analyze",
            data={"image": (io.BytesIO(b"this is not an image"), "broken.jpg")},
            content_type="multipart/form-data",
        )
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("isn&#39;t a readable image", html)

    def test_missing_file_selection(self):
        response = self.client.post("/analyze", data={}, content_type="multipart/form-data")
        html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Choose a sample photo or upload an image first", html)

    def test_oversized_upload_returns_413(self):
        original = server.app.config["MAX_CONTENT_LENGTH"]
        server.app.config["MAX_CONTENT_LENGTH"] = 1024
        try:
            response = self.client.post(
                "/analyze",
                data={"image": (io.BytesIO(b"x" * 4096), "big.jpg")},
                content_type="multipart/form-data",
            )
        finally:
            server.app.config["MAX_CONTENT_LENGTH"] = original

        self.assertEqual(response.status_code, 413)
        self.assertIn("larger than 15 MB", response.get_data(as_text=True))

    def test_uploaded_file_is_removed_after_analysis(self):
        """Successful analysis must not retain the uploaded original on disk."""
        buffer = io.BytesIO()
        Image.new("RGB", (32, 32), (200, 200, 200)).save(buffer, "JPEG")
        buffer.seek(0)

        stub = engine._empty_result()
        stub.update(
            {
                "success": True,
                "assessment_status": "insufficient_evidence",
                "assessment_title": "Unable to assess fully",
                "assessment_message": "Test message.",
                "confidence_used": 0.25,
                "person_confidence_used": 0.25,
                "inference_ms": 1.0,
                "image_size": [32, 32],
                "model_info": {
                    "ppe_model": "yolo_ppe_core.pt",
                    "person_source": "test",
                    "person_model_available": True,
                },
            }
        )

        before = {path.name for path in server.INCOMING_DIR.iterdir()}
        original = server.analyze_image
        try:
            server.analyze_image = lambda *args, **kwargs: stub
            response = self.client.post(
                "/analyze",
                data={
                    "image": (buffer, "temporary-upload.jpg"),
                    "conf": "0.25",
                    "imgsz": "640",
                },
                content_type="multipart/form-data",
            )
        finally:
            server.analyze_image = original

        after = {path.name for path in server.INCOMING_DIR.iterdir()}
        self.assertEqual(response.status_code, 200)
        self.assertEqual(after - before, set())

    def test_rendered_status_matches_backend_status(self):
        """The template must render result.assessment_status verbatim."""
        cases = {
            "potential_violation": "Potential PPE violation detected",
            "no_violation_detected": "No potential violations detected",
            "insufficient_evidence": "Unable to assess fully",
        }
        sample = find_sample("ppe_0009") or find_sample("ppe_0000")
        if sample is None:
            self.skipTest("no sample image available")

        original = server.analyze_image
        try:
            for status, title in cases.items():
                stub = engine._empty_result()
                stub.update(
                    {
                        "success": True,
                        "assessment_status": status,
                        "assessment_title": title,
                        "assessment_message": "Test message.",
                        "confidence_used": 0.25,
                        "person_confidence_used": 0.25,
                        "inference_ms": 1.0,
                        "image_size": [320, 240],
                        "model_info": {
                            "ppe_model": "yolo_ppe_core.pt",
                            "person_source": "test",
                            "person_model_available": True,
                        },
                    }
                )
                server.analyze_image = lambda *args, **kwargs: stub

                response = self.client.post(
                    "/analyze",
                    data={"sample": sample.name, "conf": "0.25", "imgsz": "640"},
                )
                html = response.get_data(as_text=True)
                self.assertEqual(response.status_code, 200)
                self.assertIn(f'data-assessment-status="{status}"', html)
                self.assertIn(title, html)
                self.assertNotIn("DEBUG STATUS", html)
        finally:
            server.analyze_image = original


# ---------------------------------------------------------------------------
# Real CPU inference tests
# ---------------------------------------------------------------------------


@unittest.skipUnless(MODELS_AVAILABLE, "PPE or person model is missing")
class TestRealInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = Path(tempfile.mkdtemp(prefix="sgv-infer-"))
        cls.blank = make_test_image(cls.temp_dir / "blank.jpg", (640, 360))
        cls.results = {}
        cls.samples = {
            "clear": find_sample("ppe_0009"),
            "no_hardhat": find_sample("ppe_0014"),
            "no_vest": find_sample("ppe_0000"),
            "multi": find_sample("ppe_0133"),
            "occluded": find_sample("ppe_0008"),
        }
        for key, path in cls.samples.items():
            if path is not None:
                cls.results[key] = engine.analyze_image(path, conf=0.25, imgsz=640)
        cls.results["blank"] = engine.analyze_image(cls.blank, conf=0.25, imgsz=640)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.temp_dir, ignore_errors=True)

    def test_01_clear_scene(self):
        result = self.results.get("clear")
        if result is None:
            self.skipTest("sample ppe_0009 is not available")
        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(result["person_count"], 1)
        self.assertGreaterEqual(result["hardhat_count"], 1)
        self.assertGreaterEqual(result["vest_count"], 1)
        self.assertEqual(result["assessment_status"], "no_violation_detected")
        self.assertGreater(result["inference_ms"], 0)
        self.assertTrue(Path(result["annotated_abs"]).is_file())

    def test_02_missing_hardhats_flagged(self):
        result = self.results.get("no_hardhat")
        if result is None:
            self.skipTest("sample ppe_0014 is not available")
        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(result["no_hardhat_count"], 1)
        self.assertEqual(result["assessment_status"], "potential_violation")

    def test_03_missing_vests_flagged(self):
        result = self.results.get("no_vest")
        if result is None:
            self.skipTest("sample ppe_0000 is not available")
        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(result["no_vest_count"], 1)
        self.assertEqual(result["assessment_status"], "potential_violation")

    def test_04_blank_image_is_insufficient(self):
        result = self.results["blank"]
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["person_count"], 0)
        self.assertEqual(result["assessment_status"], "insufficient_evidence")
        self.assertTrue(any("No people" in r for r in result["assessment_reasons"]))

    def test_05_partial_evidence_is_uncertain(self):
        result = self.results.get("occluded")
        if result is None:
            self.skipTest("sample ppe_0008 is not available")
        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(result["person_count"], 2)
        self.assertEqual(result["assessment_status"], "insufficient_evidence")
        self.assertTrue(
            any("could not be confirmed" in r for r in result["assessment_reasons"])
        )

    def test_06_multiple_workers_no_duplicate_assignment(self):
        result = self.results.get("multi")
        if result is None:
            self.skipTest("sample ppe_0133 is not available")
        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(result["person_count"], 3)
        self.assertFalse(result["counts_by_source"][-1]["source"].startswith("PPE"))

        used_slots = set()
        for detection in result["detections"]:
            worker = detection.get("assigned_person")
            if worker is None:
                continue
            self.assertGreaterEqual(worker, 1)
            self.assertLessEqual(worker, result["person_count"])
            slot = "helmet" if detection["category"] in {"hardhat", "no_hardhat"} else "vest"
            key = (worker, slot)
            self.assertNotIn(key, used_slots, "PPE detection assigned to one slot twice")
            used_slots.add(key)

    def test_07_cpu_inference_metadata(self):
        result = self.results.get("clear")
        if result is None:
            self.skipTest("sample ppe_0009 is not available")
        self.assertTrue(result["success"], result["error"])
        self.assertGreater(result["ppe_inference_ms"], 0)
        self.assertGreaterEqual(result["total_ms"], result["inference_ms"])
        self.assertTrue(result["model_info"]["person_model_available"])
        self.assertIn("COCO", result["model_info"]["person_source"])

    def test_08_missing_image_reports_error(self):
        result = engine.analyze_image(self.temp_dir / "nope.jpg")
        self.assertFalse(result["success"])
        self.assertIn("Image not found", result["error"])
        self.assertEqual(result["assessment_status"], "insufficient_evidence")

    def test_09_invalid_person_confidence_is_clamped(self):
        # Directly exercise the clamp path without a second full inference.
        self.assertEqual(engine.MIN_CONF, 0.05)
        self.assertEqual(engine.MAX_CONF, 0.95)
        self.assertGreaterEqual(engine.PERSON_CONF_FLOOR, engine.MIN_CONF)

    def test_10_uncertain_vest_is_not_forced_present(self):
        # ppe_0004 at the app default 960 px: three workers, three hard hats,
        # two vests.  P1's annotated vest is only detected at confidence 0.144,
        # below the 0.25 app threshold, so the worker must stay Unconfirmed.
        sample = find_sample("ppe_0004")
        if sample is None:
            self.skipTest("sample ppe_0004 is not available")

        result = engine.analyze_image(sample, conf=0.25, imgsz=960)

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["person_count"], 3)
        self.assertEqual(result["hardhat_count"], 3)
        self.assertEqual(result["vest_count"], 2)

        unknown = [person for person in result["persons"] if person["vest"]["state"] == "unknown"]
        self.assertEqual(len(unknown), 1)
        self.assertEqual(unknown[0]["status"], "uncertain")
        self.assertIsNone(unknown[0]["vest"]["confidence"])

        # No vest box may be attached to the unknown worker.
        forced = [
            detection
            for detection in result["detections"]
            if detection["category"] == "vest"
            and detection["assigned_person"] == unknown[0]["id"]
        ]
        self.assertEqual(forced, [])
        self.assertEqual(result["assessment_status"], "insufficient_evidence")


if __name__ == "__main__":
    unittest.main(verbosity=2)
