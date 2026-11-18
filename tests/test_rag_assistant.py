"""Unit tests for document parsing, grounding, persistence and assistant APIs."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from rag.document_loader import chunk_text, clean_text
from rag.llm_service import OllamaService, _model_present
from rag.retriever import RetrievedChunk
from rag.safety_assistant import (
    INSUFFICIENT_MESSAGE,
    NO_ANALYSIS_MESSAGE,
    OFFLINE_MESSAGE,
    SYSTEM_PROMPT,
    SafetyAssistant,
    _PRODUCT_FACTS,
    build_messages,
    format_analysis_facts,
    is_image_question,
    is_siteguardvision_question,
    summarize_analysis,
)
from rag.vector_store import VectorStore
import storage
import server


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.posts = []

    def get(self, url, timeout=None):
        return FakeResponse(
            {
                "models": [
                    {"name": "qwen2.5:3b"},
                    {"name": "nomic-embed-text:latest"},
                ]
            }
        )

    def post(self, url, json=None, timeout=None):
        self.posts.append((url, json, timeout))
        if url.endswith("/api/chat"):
            return FakeResponse({"message": {"content": "Use the site policy [1]."}})
        if url.endswith("/api/embed"):
            return FakeResponse({"embeddings": [[0.1, 0.2] for _ in json["input"]]})
        return FakeResponse({"embedding": [0.1, 0.2]})


class FakeEmbeddings:
    model = "test-embed"

    def embed(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeStore:
    def __init__(self, matches=None):
        self.matches = matches or []

    def count(self):
        return len(self.matches)

    def query(self, embedding, top_k):
        return self.matches[:top_k]


class FakeRetriever:
    def __init__(self, chunks):
        self.chunks = chunks

    def retrieve(self, question):
        return self.chunks


class TestDocumentText(unittest.TestCase):
    def test_clean_text_removes_control_chars_and_collapses_blank_lines(self):
        self.assertEqual(clean_text(" a  b\r\n\r\n\r\nc\x00 "), "a b\n\nc")

    def test_chunk_text_respects_chunk_limit_and_overlap(self):
        chunks = chunk_text(("Safety instructions. " * 80), size=300, overlap=40)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 300 for chunk in chunks))


class TestGroundingHelpers(unittest.TestCase):
    def test_image_question_detection(self):
        self.assertTrue(is_image_question("How many workers were detected?"))
        self.assertFalse(is_image_question("What is the general PPE guidance?"))

    def test_analysis_facts_use_structured_worker_data(self):
        facts = format_analysis_facts(
            {
                "assessment_status": "insufficient_evidence",
                "assessment_title": "Unable to assess fully",
                "assessment_message": "Evidence is incomplete.",
                "person_count": 1,
                "hardhat_count": 1,
                "no_hardhat_count": 0,
                "vest_count": 0,
                "no_vest_count": 0,
                "persons": [
                    {
                        "id": 2,
                        "confidence": 0.87,
                        "helmet": {"state": "present", "confidence": 0.92},
                        "vest": {"state": "unknown", "confidence": None},
                        "status": "uncertain",
                    }
                ],
            }
        )
        self.assertIn("Worker P2", facts)
        self.assertIn("Unconfirmed", facts)
        self.assertIn("Unable to assess fully", facts)

    def test_prompt_separates_document_context_and_image_facts(self):
        chunk = RetrievedChunk(
            text="Workers must follow the posted PPE rules.",
            filename="site-policy.pdf",
            page=3,
            chunk_index=0,
            similarity=0.72,
        )
        messages = build_messages("Explain this result", [chunk], "Assessment status: insufficient_evidence")
        self.assertIn("CONTEXT (retrieved safety documents)", messages[1]["content"])
        self.assertIn("ANALYSIS FACTS", messages[1]["content"])
        self.assertIn("site-policy.pdf, page 3", messages[1]["content"])

    def test_insufficient_fallback_is_exact(self):
        self.assertEqual(
            INSUFFICIENT_MESSAGE,
            "I could not find sufficient supporting information in the available safety documents. Please consult the applicable site safety policy.",
        )

    def test_system_prompt_covers_conversational_and_architecture_rules(self):
        for phrase in (
            "conversational",
            "never require an uploaded or indexed document",
            "YOLO-based person detector",
            "Unconfirmed",
            "No Violation Detected",
            "Potential Violation",
            "Insufficient Evidence",
            "not legal advice",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase.lower(), SYSTEM_PROMPT.lower())

    def test_siteguardvision_question_detection(self):
        self.assertTrue(is_siteguardvision_question("How does PPE detection work?"))
        self.assertTrue(is_siteguardvision_question("Explain the SiteGuardVision architecture"))
        self.assertFalse(is_siteguardvision_question("What is a safety vest?"))

    def test_product_facts_included_for_architecture_question(self):
        messages = build_messages(
            "How does PPE detection work?", [], "", product_facts=_PRODUCT_FACTS
        )
        self.assertIn("SITEGUARDVISION PRODUCT FACTS", messages[1]["content"])
        self.assertIn("does not process video", messages[1]["content"])

    def test_history_is_included_before_the_current_question(self):
        messages = build_messages(
            "Why is it important?",
            [],
            "",
            [
                {"role": "user", "content": "What is a safety vest?"},
                {"role": "assistant", "content": "A high-visibility garment."},
            ],
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("What is a safety vest?", messages[1]["content"])
        self.assertEqual(messages[2]["role"], "assistant")
        self.assertEqual(messages[3]["role"], "user")
        self.assertIn("Why is it important?", messages[3]["content"])

    def test_offline_image_question_gets_backend_only_summary(self):
        summary = summarize_analysis(
            {
                "assessment_title": "Unable to assess fully",
                "person_count": 1,
                "hardhat_count": 1,
                "no_hardhat_count": 0,
                "vest_count": 0,
                "no_vest_count": 0,
                "persons": [
                    {
                        "id": 1,
                        "helmet": {"state": "present"},
                        "vest": {"state": "unknown"},
                        "status": "uncertain",
                    }
                ],
            }
        )
        self.assertIn("detected 1 workers", summary)
        self.assertIn("P1", summary)
        self.assertIn("Manually verify", summary)


class TestOllamaService(unittest.TestCase):
    def test_model_matching_allows_implicit_latest(self):
        self.assertTrue(_model_present(["nomic-embed-text:latest"], "nomic-embed-text"))
        self.assertFalse(_model_present(["qwen2.5:3b"], "qwen2.5:7b"))

    def test_status_and_chat_use_installed_local_model(self):
        session = FakeSession()
        service = OllamaService(session=session)
        status = service.status(force=True)
        self.assertTrue(status["reachable"])
        self.assertTrue(status["chat_model_available"])
        answer, elapsed = service.chat([{"role": "user", "content": "Question"}])
        self.assertIn("[1]", answer)
        self.assertGreaterEqual(elapsed, 0)


class TestAssistantWithoutNetwork(unittest.TestCase):
    @staticmethod
    def _mock_llm(answer="A general conversational answer.", llm_ms=18.0):
        return type(
            "MockLLM",
            (),
            {
                "chat_model": "qwen2.5:3b",
                "status": lambda self, **kwargs: {
                    "reachable": True,
                    "chat_model_available": True,
                    "embed_model_available": True,
                },
                "chat": lambda self, messages, **kwargs: (answer, llm_ms),
            },
        )()

    def test_general_question_without_documents_uses_llm(self):
        assistant = SafetyAssistant(
            llm=self._mock_llm(),
            embeddings=object(),
            store=object(),
            retriever=object(),
        )
        assistant.status = lambda force=False: {
            "mode": "ready",
            "ready": True,
            "chat_model": "qwen2.5:3b",
            "rag": {"chunks": 0, "retrieval_available": False},
        }
        response = assistant.answer("What PPE is required on a construction site?")
        self.assertEqual(response["answer"], "A general conversational answer.")
        self.assertEqual(response["mode"], "chat")
        self.assertEqual(response["basis"], "general")
        self.assertEqual(response["sources"], [])

    def test_answer_contains_sources_and_timings(self):
        chunk = RetrievedChunk(
            text="Hard hats must be worn in the designated area.",
            filename="policy.txt",
            page=None,
            chunk_index=0,
            similarity=0.8,
        )
        assistant = SafetyAssistant(
            llm=self._mock_llm("Wear the hard hat [1].", 24.0),
            embeddings=object(),
            store=object(),
            retriever=FakeRetriever([chunk]),
        )
        assistant.status = lambda force=False: {
            "mode": "ready",
            "ready": True,
            "chat_model": "qwen2.5:3b",
            "rag": {"chunks": 1, "retrieval_available": True},
        }
        response = assistant.answer("What does the policy say?")
        self.assertEqual(response["answer"], "Wear the hard hat [1].")
        self.assertEqual(response["sources"][0]["filename"], "policy.txt")
        self.assertEqual(response["basis"], "documents")
        self.assertIn("total_ms", response["timings"])
        self.assertEqual(response["timings"]["llm_ms"], 24.0)

    def test_follow_up_question_includes_conversation_history(self):
        captured = {}

        def chat(messages, **kwargs):
            captured["messages"] = messages
            return "Follow-up answer.", 12.0

        llm = self._mock_llm()
        llm.chat = chat
        assistant = SafetyAssistant(
            llm=llm,
            embeddings=object(),
            store=object(),
            retriever=object(),
        )
        assistant.status = lambda force=False: {
            "mode": "ready",
            "ready": True,
            "chat_model": "qwen2.5:3b",
            "rag": {"chunks": 0, "retrieval_available": False},
        }
        history = [{"role": "user", "content": "What is a safety vest?"}]
        response = assistant.answer(
            "Why is it important?", history=history, session_id="test-session"
        )
        self.assertEqual(response["answer"], "Follow-up answer.")
        roles = [message["role"] for message in captured["messages"]]
        self.assertEqual(roles, ["system", "user", "user"])
        self.assertIn("What is a safety vest?", captured["messages"][1]["content"])
        self.assertIn("Why is it important?", captured["messages"][2]["content"])
        self.assertEqual(
            assistant.history("test-session"),
            [
                {"role": "user", "content": "Why is it important?"},
                {"role": "assistant", "content": "Follow-up answer."},
            ],
        )

    def test_image_question_without_analysis_is_explained(self):
        assistant = SafetyAssistant(
            llm=self._mock_llm(),
            embeddings=object(),
            store=object(),
            retriever=object(),
        )
        response = assistant.answer("How many workers were detected?")
        self.assertEqual(response["mode"], "no_analysis")
        self.assertEqual(response["answer"], NO_ANALYSIS_MESSAGE)

    def test_offline_blocks_chat_but_keeps_image_facts(self):
        assistant = SafetyAssistant(
            llm=self._mock_llm(),
            embeddings=object(),
            store=object(),
            retriever=object(),
        )
        assistant.status = lambda force=False: {
            "mode": "offline",
            "ready": False,
            "chat_model": "qwen2.5:3b",
            "rag": {"chunks": 0, "retrieval_available": False},
        }
        response = assistant.answer(
            "Explain this image",
            {"assessment_title": "Unable to assess fully", "person_count": 1},
        )
        self.assertFalse(response["ok"])
        self.assertEqual(response["mode"], "offline")
        self.assertTrue(response["answer"].startswith(OFFLINE_MESSAGE))
        self.assertIn("Unable to assess fully", response["answer"])


class TestStorage(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("SITEGUARD_DB_PATH")
        os.environ["SITEGUARD_DB_PATH"] = str(Path(self.temp.name) / "test.db")
        storage._schema_ready.clear()

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("SITEGUARD_DB_PATH", None)
        else:
            os.environ["SITEGUARD_DB_PATH"] = self.previous
        storage._schema_ready.clear()
        self.temp.cleanup()

    def test_save_analysis_and_stats(self):
        result = {
            "success": True,
            "annotated_rel": "rendered/example.jpg",
            "assessment_status": "insufficient_evidence",
            "assessment_title": "Unable to assess fully",
            "assessment_message": "Test.",
            "person_count": 2,
            "hardhat_count": 1,
            "no_hardhat_count": 0,
            "vest_count": 1,
            "no_vest_count": 1,
            "confidence_used": 0.25,
            "person_confidence_used": 0.25,
            "inference_ms": 120.0,
            "image_size": [640, 480],
            "model_info": {"ppe_model": "yolo_ppe_core.pt", "person_source": "test"},
        }
        record_id = storage.save_analysis(result, source_label="site.jpg", source_name="x.jpg", imgsz=640)
        self.assertIsNotNone(record_id)
        self.assertEqual(storage.stats()["total"], 1)
        stored = storage.get_analysis(record_id)
        self.assertEqual(stored["result"]["person_count"], 2)

    def test_settings_validate_and_persist(self):
        values, errors = storage.save_settings({"conf": "0.45", "imgsz": "1280"})
        self.assertFalse(errors)
        self.assertEqual(values["conf"], "0.45")
        self.assertEqual(values["imgsz"], "1280")
        values, errors = storage.save_settings({"imgsz": "777"})
        self.assertIn("imgsz", errors)
        self.assertEqual(values["imgsz"], "1280")


class TestNewRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = server.app.test_client()

    def test_navigation_pages_render(self):
        for path in ("/assistant", "/reports", "/evaluation", "/settings"):
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200)

    def test_chat_rejects_empty_input(self):
        response = self.client.post("/api/assistant/chat", json={"message": " "})
        self.assertEqual(response.status_code, 400)

    def test_document_upload_rejects_wrong_extension(self):
        response = self.client.post(
            "/api/assistant/documents",
            data={"file": (io.BytesIO(b"not a document"), "notes.exe")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Only PDF", response.get_json()["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
