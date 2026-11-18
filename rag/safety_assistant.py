"""Conversational Safety Assistant backed by a local Ollama LLM.

Design rules enforced here:

* General questions are answered from the pretrained knowledge of the
  configured Ollama chat model. No indexed document is required.
* Document retrieval is an optional enhancement: it is used only when an
  index exists, the embedding model is available, and relevant passages are
  found. Answers that use documents cite them; answers that do not say so.
* Image facts come only from the structured computer-vision result. The
  assistant never invents detections, workers, confidence values, or
  compliance conclusions.
* When Ollama is offline the rest of the app keeps working; the assistant says
  so clearly and can still show a deterministic fact sheet for the analysed
  image.
* Conversation context is kept per session so follow-up questions work.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Any

from . import config
from .embeddings import OllamaEmbeddings, OllamaError
from .llm_service import OllamaError as ChatError
from .llm_service import OllamaService
from .retriever import Retriever, RetrievedChunk
from .vector_store import (
    RagDependencyError,
    RagError,
    VectorStore,
    list_documents,
    pending_documents,
    read_manifest,
)

# Retained for backwards compatibility with callers/tests. It is no longer a
# blocker: general questions are answered by the chat model without documents.
INSUFFICIENT_MESSAGE = (
    "I could not find sufficient supporting information in the available safety documents. "
    "Please consult the applicable site safety policy."
)

OFFLINE_MESSAGE = (
    "The local Ollama service is not reachable, so I can't answer questions right now. "
    "Image analysis still works. Start Ollama and make sure the conversational model is "
    "installed with `ollama pull qwen2.5:3b`, then try again."
)

NO_ANALYSIS_MESSAGE = (
    "No image analysis result is available in this session yet. Upload a photo or "
    "choose a sample on the Analysis page, then ask again."
)

SYSTEM_PROMPT = """You are the SiteGuardVision Construction Safety and PPE Guidance Assistant.

You are a helpful, professional conversational assistant. Users ask questions about construction-site safety, personal protective equipment (PPE), risk assessment, and the SiteGuardVision application. Answer directly, in plain English, with complete and useful explanations.

Core behaviour:
1. Answer general questions from your own pretrained knowledge. You never require an uploaded or indexed document to answer.
2. When the user asks about an uploaded image or analysis, use only the supplied ANALYSIS FACTS. Never invent detections, workers, confidence values, bounding boxes, or compliance conclusions.
3. When a CONTEXT section is supplied, it contains retrieved safety-document passages. Use it to support the answer and cite the relevant passages with [1], [2], and so on, matching the numbered context. If no context is supplied, do not invent citations.
4. Clearly distinguish three kinds of information: (a) general guidance from your knowledge, (b) document-supported guidance from the CONTEXT, and (c) facts from SiteGuardVision's computer-vision backend. Do not mix them up.
5. Explain technical concepts when asked, and give practical examples where helpful.
6. Avoid unnecessary length. About 150-300 words is usually enough; stay concise and conversational.

SiteGuardVision architecture:
- SiteGuardVision processes uploaded still images only. It does not process video, live cameras, or streams. Never claim video or live-camera support.
- It uses a YOLO-based person detector to find workers in the uploaded photo.
- It uses a separate YOLO-based PPE detector. The worker-association step uses hard-hat, no-hard-hat, safety-vest, and no-safety-vest evidence. The checkpoint also contains other classes, but do not claim it detects safety footwear, eye protection, gloves, or respiratory equipment for worker assessment. The PPE checkpoint's own person class is not used when the dedicated person detector is available.
- A worker-to-PPE association step attaches PPE boxes to workers using head and torso regions. A detection must overlap the relevant region and its center must fall inside it, and each worker slot receives at most one detection. Same-category duplicate boxes are merged; opposite present/missing categories are kept as conflicting evidence.
- Each worker receives a hard-hat state and a vest state: Present, Missing, or Unconfirmed. Unconfirmed means the system found no matching PPE evidence, not that PPE is definitely absent.
- The image-level assessment is one of: No Violation Detected, Potential Violation, or Insufficient Evidence. A clear result is not a compliance certificate.
- The Flask backend renders an analysis dashboard, stores analysis history in SQLite, and can optionally augment answers with retrieved safety documents.

Safety guidance rules:
- General safety guidance is not legal advice. Do not invent specific laws, regulations, fine amounts, or penalties.
- Do not claim that any recommendation is a universally binding legal requirement. Recommend checking applicable local regulations and site-specific risk assessments.
- When you are uncertain, say so plainly rather than guessing.
- Never declare a worker compliant when the only available evidence is missing; distinguish an explicit missing-PPE detection from an unconfirmed state.
"""


_SITEGUARDVISION_MARKERS = (
    "siteguardvision",
    "site guard vision",
    "site-guard vision",
    "ppe detection",
    "how does ppe",
    "worker-to-ppe",
    "worker to ppe",
    "identify workers",
    "analysis dashboard",
    "sqlite",
    "flask backend",
    "image-based processing",
)

_PRODUCT_FACTS = """SITEGUARDVISION PRODUCT FACTS (authoritative — use these instead of guessing):
- SiteGuardVision analyzes uploaded still images. It does not process video or live camera streams.
- Person detection: a YOLO-based COCO person detector (models/person_yolov8n.pt) finds workers.
- PPE detection: a separate YOLO checkpoint (models/yolo_ppe_core.pt) emits hard-hat, no-hard-hat, safety-vest, and no-safety-vest evidence. Worker assessment uses these four. The application does not assess safety footwear, eye protection, gloves, or respiratory equipment.
- Association: PPE boxes are attached to a worker's head region (hard hat / no hard hat) or torso region (vest / no vest). A candidate must overlap the region (at least 0.30 intersection-over-detection) and its center must be inside the region. Each worker slot receives at most one detection. Same-category duplicate boxes are merged; opposite present/missing categories are kept as conflicting evidence.
- Worker states: hard hat and vest are each Present, Missing, or Unconfirmed. Unconfirmed means no matching PPE evidence was found, not that PPE is definitely absent.
- Image assessment: No Violation Detected, Potential Violation, or Insufficient Evidence. A clear result is not a compliance certificate.
- Backend: Flask; analysis history is stored in SQLite; the dashboard shows workers, PPE states, and review flags.
- Documents are optional and are used only when indexed and relevant."""

_IMAGE_MARKERS = (
    "this image",
    "the image",
    "current image",
    "this photo",
    "the photo",
    "current analysis",
    "this analysis",
    "the analysis",
    "this result",
    "current result",
    "worker assessment",
    "ppe assessment",
    "which worker",
    "how many workers",
    "how many people",
    "were detected",
    "was detected",
    "was marked",
    "marked unable",
    "explain the assessment",
    "before approving",
    "what should i verify",
    "unconfirmed ppe",
    "why was this",
)

# Conversation memory limits. Each turn stores one user message and one
# assistant message; the oldest sessions are evicted first.
MAX_SESSION_TURNS = 20
MAX_SESSIONS = 500


def is_image_question(question: str) -> bool:
    lowered = (question or "").lower()
    return any(marker in lowered for marker in _IMAGE_MARKERS)


def is_siteguardvision_question(question: str) -> bool:
    lowered = (question or "").lower()
    return any(marker in lowered for marker in _SITEGUARDVISION_MARKERS)


def _pct(value: Any) -> str:
    if value is None:
        return "no confidence recorded"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "no confidence recorded"


def _state(value: Any) -> str:
    return {
        "present": "Present",
        "missing": "Missing",
        "unknown": "Unconfirmed",
    }.get(str(value or ""), "Unconfirmed")


def _worker_status(value: Any) -> str:
    return {
        "potential_violation": "Potential violation",
        "clear": "Clear",
        "uncertain": "Unable to assess",
    }.get(str(value or ""), "Unable to assess")


def format_analysis_facts(analysis: dict[str, Any] | None, *, compact: bool = False) -> str:
    """Render structured CV output as plain facts for the LLM and for offline UI."""
    if not analysis:
        return ""
    if isinstance(analysis.get("result"), dict):
        analysis = analysis["result"]

    reasons = analysis.get("assessment_reasons") or []
    lines = [
        f"Assessment status: {analysis.get('assessment_status', 'insufficient_evidence')}",
        f"Assessment title: {analysis.get('assessment_title', 'Unable to assess fully')}",
        f"Assessment message: {analysis.get('assessment_message', '')}",
        (
            "Detection counts: "
            f"{int(analysis.get('person_count') or 0)} workers, "
            f"{int(analysis.get('hardhat_count') or 0)} hard hats, "
            f"{int(analysis.get('no_hardhat_count') or 0)} missing-hard-hat detections, "
            f"{int(analysis.get('vest_count') or 0)} safety vests, "
            f"{int(analysis.get('no_vest_count') or 0)} missing-safety-vest detections."
        ),
    ]
    if reasons:
        lines.append("Backend reasons: " + " ".join(str(reason) for reason in reasons))

    persons = analysis.get("persons") or []
    if persons:
        limit = 5 if compact else 12
        lines.append("Worker evidence:")
        for person in persons[:limit]:
            helmet = person.get("helmet") or {}
            vest = person.get("vest") or {}
            lines.append(
                f"- Worker P{person.get('id', '?')}: person confidence "
                f"{_pct(person.get('confidence'))}; hard hat {_state(helmet.get('state'))} "
                f"({_pct(helmet.get('confidence'))}); safety vest {_state(vest.get('state'))} "
                f"({_pct(vest.get('confidence'))}); assessment {_worker_status(person.get('status'))}."
            )
        if len(persons) > limit:
            lines.append(f"- {len(persons) - limit} further workers are not listed here.")

    flags = []
    for label, key in (("No hard hat", "potential_no_hardhat"), ("No safety vest", "potential_no_vest")):
        for item in analysis.get(key) or []:
            owner = item.get("assigned_person")
            owner_text = f"worker P{owner}" if owner else "not matched to a worker"
            flags.append(f"- {label}: {_pct(item.get('confidence'))} confidence, {owner_text}.")
    if flags:
        lines.append("Missing-PPE flags:")
        lines.extend(flags[:12])

    model_info = analysis.get("model_info") or {}
    if model_info:
        lines.append(
            "Models: PPE `{}`; person `{}`.".format(
                model_info.get("ppe_model") or "unknown",
                model_info.get("person_source") or "unknown",
            )
        )
    return "\n".join(lines)


def summarize_analysis(analysis: dict[str, Any] | None) -> str:
    """Give image-specific responses using only structured CV/backend facts."""
    if not analysis:
        return (
            "No current image-analysis result is available. Upload a photo or select a "
            "sample image first."
        )
    if isinstance(analysis.get("result"), dict):
        analysis = analysis["result"]

    status = analysis.get("assessment_title") or "Unable to assess fully"
    workers = analysis.get("persons") or []
    lines = [
        f"Backend assessment: {status}.",
        (
            f"The computer-vision pipeline detected {int(analysis.get('person_count') or 0)} "
            f"workers, {int(analysis.get('hardhat_count') or 0)} hard hats, "
            f"{int(analysis.get('vest_count') or 0)} safety vests, "
            f"{int(analysis.get('no_hardhat_count') or 0)} missing-hard-hat detections, and "
            f"{int(analysis.get('no_vest_count') or 0)} missing-safety-vest detections."
        ),
    ]
    uncertain = []
    for person in workers:
        helmet = (person.get("helmet") or {}).get("state")
        vest = (person.get("vest") or {}).get("state")
        if helmet == "unknown" or vest == "unknown" or person.get("status") == "uncertain":
            uncertain.append(f"P{person.get('id', '?')}")
    if uncertain:
        lines.append("Worker PPE with unconfirmed evidence: " + ", ".join(uncertain) + ".")
    flags = []
    for label, key in (("hard hat", "potential_no_hardhat"), ("safety vest", "potential_no_vest")):
        for item in analysis.get(key) or []:
            who = f"P{item['assigned_person']}" if item.get("assigned_person") else "unmatched"
            flags.append(f"{label} flag for {who}")
    if flags:
        lines.append("Backend flags: " + "; ".join(flags) + ".")
    reasons = analysis.get("assessment_reasons") or []
    if reasons:
        lines.append("Assessment notes: " + " ".join(str(reason) for reason in reasons))
    lines.append(
        "These are CV detections and backend assessment facts, not a compliance verdict. "
        "Manually verify the image."
    )
    return "\n".join(lines)


class SafetyAssistant:
    def __init__(
        self,
        *,
        llm: OllamaService | None = None,
        embeddings: OllamaEmbeddings | None = None,
        store: VectorStore | None = None,
        retriever: Retriever | None = None,
        top_k: int | None = None,
    ) -> None:
        self.llm = llm or OllamaService()
        self.embeddings = embeddings or OllamaEmbeddings()
        self.store = store or VectorStore()
        self.retriever = retriever or Retriever(
            self.embeddings, self.store, top_k=top_k
        )
        self._lock = threading.Lock()
        self._status_cache: dict[str, Any] | None = None
        self._status_at = 0.0
        self._cache_seconds = 10.0
        self._sessions: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
        self._sessions_lock = threading.Lock()

    # -- configuration ------------------------------------------------------

    @staticmethod
    def configured_chat_model() -> str:
        try:
            from storage import get_settings

            return str(get_settings().get("ollama_model") or config.OLLAMA_CHAT_MODEL)
        except Exception:  # noqa: BLE001 - storage must not block the assistant
            return config.OLLAMA_CHAT_MODEL

    # -- conversation memory ------------------------------------------------

    def history(self, session_id: str | None) -> list[dict[str, str]]:
        if not session_id:
            return []
        with self._sessions_lock:
            return [dict(item) for item in self._sessions.get(session_id, [])]

    def clear_session(self, session_id: str | None) -> None:
        if not session_id:
            return
        with self._sessions_lock:
            self._sessions.pop(session_id, None)

    def _remember(self, session_id: str | None, role: str, content: str) -> None:
        if not session_id:
            return
        with self._sessions_lock:
            turns = self._sessions.setdefault(session_id, [])
            turns.append({"role": role, "content": content})
            while len(turns) > MAX_SESSION_TURNS * 2:
                turns.pop(0)
            self._sessions.move_to_end(session_id)
            while len(self._sessions) > MAX_SESSIONS:
                self._sessions.popitem(last=False)

    @staticmethod
    def _sanitized_history(history: Any) -> list[dict[str, str]]:
        if not isinstance(history, list):
            return []
        clean: list[dict[str, str]] = []
        for item in history:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                continue
            text = content.strip()
            if not text:
                continue
            clean.append({"role": role, "content": text})
        return clean[-MAX_SESSION_TURNS * 2 :]

    # -- status -------------------------------------------------------------

    def status(self, force: bool = False) -> dict[str, Any]:
        now = time.monotonic()
        with self._lock:
            cached = self._status_cache
            if not force and cached is not None and (now - self._status_at) < self._cache_seconds:
                return dict(cached)

        chat_model = self.configured_chat_model()
        ollama = self.llm.status(force=force, chat_model=chat_model)

        rag: dict[str, Any] = {
            "documents": 0,
            "chunks": 0,
            "pending": 0,
            "embedding_model": self.embeddings.model,
            "index_embedding_model": None,
            "dependency_error": None,
            "index_error": None,
            "retrieval_available": False,
        }
        try:
            manifest = read_manifest()
            rag["documents"] = len(manifest.get("documents") or {})
            rag["index_embedding_model"] = manifest.get("embedding_model")
            rag["chunks"] = self.store.count()
            disk_model = self.store.collection_embedding_model()
            if disk_model:
                rag["index_embedding_model"] = disk_model
            rag["pending"] = len(pending_documents())
        except RagDependencyError as exc:
            rag["dependency_error"] = str(exc)
        except RagError as exc:
            rag["index_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 - status must never crash the app
            rag["index_error"] = f"{type(exc).__name__}: {exc}"

        reindex_required = bool(
            rag["index_embedding_model"]
            and rag["embedding_model"]
            and rag["index_embedding_model"] != rag["embedding_model"]
        )

        # Documents are optional. Only the chat model decides whether the
        # assistant can hold a normal conversation.
        if not ollama.get("reachable"):
            mode = "offline"
        elif not ollama.get("chat_model_available"):
            mode = "models_missing"
        elif rag["dependency_error"]:
            mode = "dependency_missing"
        elif rag["index_error"]:
            mode = "index_error"
        elif reindex_required:
            mode = "reindex_required"
        else:
            mode = "ready"

        rag["retrieval_available"] = bool(
            rag["chunks"] > 0
            and ollama.get("embed_model_available")
            and not rag["dependency_error"]
            and not rag["index_error"]
            and not reindex_required
        )

        chat_ready = mode not in {"offline", "models_missing"}
        status = {
            "mode": mode,
            "ready": chat_ready,
            "chat_ready": chat_ready,
            "documents_available": bool(rag["chunks"] > 0),
            "available": bool(ollama.get("reachable")),
            "ollama": ollama,
            "rag": rag,
            "chat_model": chat_model,
        }
        with self._lock:
            self._status_cache = dict(status)
            self._status_at = time.monotonic()
        return status

    # -- answering ----------------------------------------------------------

    def answer(
        self,
        question: str,
        analysis: dict[str, Any] | None = None,
        *,
        analysis_id: int | None = None,
        history: list[dict[str, str]] | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValueError("Please enter a question.")
        if len(cleaned) > config.MAX_QUESTION_CHARS:
            raise ValueError(
                f"Question is too long. Keep it under {config.MAX_QUESTION_CHARS} characters."
            )

        wants_image = is_image_question(cleaned)
        facts = format_analysis_facts(analysis) if analysis else ""

        if wants_image and not facts:
            return self._result(
                started,
                ok=True,
                mode="no_analysis",
                answer=NO_ANALYSIS_MESSAGE,
                sources=[],
                model=None,
                analysis_id=analysis_id,
                basis="none",
            )

        status = self.status(force=True)
        mode = status["mode"]
        chat_model = status["chat_model"]

        if mode == "offline":
            answer = OFFLINE_MESSAGE
            if wants_image and facts:
                answer += "\n\n" + summarize_analysis(analysis)
            return self._result(
                started, ok=False, mode="offline", answer=answer,
                sources=[], model=chat_model, status=status, analysis_id=analysis_id,
                basis="none",
            )

        if mode == "models_missing":
            ollama = status["ollama"]
            answer = (
                "Ollama is running, but the conversational model "
                f"`{ollama.get('chat_model')}` is not installed. Run "
                f"`ollama pull {ollama.get('chat_model')}`."
            )
            if wants_image and facts:
                answer += "\n\n" + summarize_analysis(analysis)
            return self._result(
                started, ok=False, mode="models_missing", answer=answer,
                sources=[], model=chat_model, status=status, analysis_id=analysis_id,
                basis="none",
            )

        # Optional document retrieval. Any failure here degrades to a normal
        # conversational answer instead of blocking the user.
        chunks: list[RetrievedChunk] = []
        retrieval_ms = 0.0
        if status["rag"].get("retrieval_available"):
            retrieval_started = time.perf_counter()
            try:
                chunks = self.retriever.retrieve(cleaned)
            except (OllamaError, RagError):
                chunks = []
            except Exception:  # noqa: BLE001 - documents must never block chat
                chunks = []
            retrieval_ms = round((time.perf_counter() - retrieval_started) * 1000.0, 1)

        if history:
            conversation = self._sanitized_history(history)
        else:
            conversation = self.history(session_id)

        product_facts = (
            _PRODUCT_FACTS if is_siteguardvision_question(cleaned) else ""
        )
        messages = build_messages(
            cleaned, chunks, facts, conversation, product_facts=product_facts
        )
        try:
            answer, llm_ms = self.llm.chat(messages, model=chat_model)
        except ChatError as exc:
            return self._result(
                started, ok=False, mode="error",
                answer=f"The local language model could not answer: {exc}",
                sources=_sources(chunks), model=chat_model,
                retrieval_ms=retrieval_ms, status=status, analysis_id=analysis_id,
                basis="none",
            )

        self._remember(session_id, "user", cleaned)
        self._remember(session_id, "assistant", answer)

        basis = "documents" if chunks else ("analysis" if facts else "general")
        return self._result(
            started,
            ok=True,
            mode="document" if chunks else "chat",
            answer=answer,
            sources=_sources(chunks),
            model=chat_model,
            retrieval_ms=retrieval_ms,
            llm_ms=llm_ms,
            status=status,
            analysis_id=analysis_id,
            basis=basis,
        )

    @staticmethod
    def _result(
        started: float,
        *,
        ok: bool,
        mode: str,
        answer: str,
        sources: list[dict[str, Any]],
        model: str | None,
        retrieval_ms: float = 0.0,
        llm_ms: float = 0.0,
        hint: str | None = None,
        status: dict[str, Any] | None = None,
        analysis_id: int | None = None,
        basis: str = "general",
    ) -> dict[str, Any]:
        return {
            "ok": ok,
            "mode": mode,
            "answer": answer,
            "sources": sources,
            "model": model,
            "analysis_id": analysis_id,
            "hint": hint,
            "basis": basis,
            "timings": {
                "retrieval_ms": round(float(retrieval_ms), 1),
                "llm_ms": round(float(llm_ms), 1),
                "total_ms": round((time.perf_counter() - started) * 1000.0, 1),
            },
            "status": (status or {}).get("mode") if status else None,
        }


def build_messages(
    question: str,
    chunks: list[RetrievedChunk],
    analysis_facts: str,
    history: list[dict[str, str]] | None = None,
    product_facts: str = "",
) -> list[dict[str, str]]:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        location = f"{chunk.filename}, page {chunk.page}" if chunk.page else chunk.filename
        context_blocks.append(f"[{index}] ({location})\n{chunk.text}")

    user_parts: list[str] = []
    if product_facts:
        user_parts.extend(
            [
                product_facts,
                "\nUse these product facts for SiteGuardVision questions. Do not add capabilities that are not listed.",
            ]
        )
    if context_blocks:
        user_parts.extend(
            [
                "\nCONTEXT (retrieved safety documents):",
                "\n\n".join(context_blocks),
            ]
        )
    if analysis_facts:
        user_parts.extend(
            [
                "\nANALYSIS FACTS (from the SiteGuardVision computer-vision backend):",
                analysis_facts,
            ]
        )
    user_parts.extend(["\nQUESTION:", question])

    messages: list[dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]
    if history:
        messages.extend(
            {"role": item["role"], "content": item["content"]} for item in history
        )
    messages.append({"role": "user", "content": "\n".join(user_parts)})
    return messages


def _sources(chunks: list[RetrievedChunk]) -> list[dict[str, Any]]:
    return [
        {
            "citation": index,
            "filename": chunk.filename,
            "page": chunk.page,
            "chunk_index": chunk.chunk_index,
            "similarity": chunk.similarity,
            "label": chunk.label,
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


_assistant_lock = threading.Lock()
_assistant: SafetyAssistant | None = None


def get_assistant() -> SafetyAssistant:
    global _assistant
    with _assistant_lock:
        if _assistant is None:
            _assistant = SafetyAssistant()
        return _assistant
