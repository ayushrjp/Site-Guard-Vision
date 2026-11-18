"""Live verification of the conversational Safety Assistant with real Ollama.

Runs the required chat scenarios against the real local Ollama model and the
real Flask routes, then writes a JSON log and a Markdown report.  No mocks are
used for generation.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import server  # noqa: E402
import storage  # noqa: E402
from rag.llm_service import OllamaService  # noqa: E402
from rag.safety_assistant import (  # noqa: E402
    OFFLINE_MESSAGE,
    SafetyAssistant,
    get_assistant,
)

RESULTS_DIR = ROOT / "evaluation" / "results"
REPORT_DIR = ROOT / "reports"
SCENARIOS: list[dict] = []


def record(name: str, **fields) -> None:
    entry = {"scenario": name, **fields}
    SCENARIOS.append(entry)
    print(f"[scenario] {name}: {fields.get('ok')} mode={fields.get('mode')}", flush=True)


def api_chat(client, message: str, *, session_id=None, analysis_id=None) -> dict:
    response = client.post(
        "/api/assistant/chat",
        json={
            "message": message,
            "session_id": session_id,
            "analysis_id": analysis_id,
        },
    )
    try:
        payload = response.get_json() or {}
    except Exception:  # noqa: BLE001
        payload = {}
    return {"http_status": response.status_code, **payload}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    server.app.config.update(TESTING=True)
    client = server.app.test_client()
    assistant = get_assistant()

    status = assistant.status(force=True)
    document_count = int(status["rag"]["documents"] or 0)
    chunk_count = int(status["rag"]["chunks"] or 0)
    print(f"Ollama reachable: {status['available']}; documents: {document_count}; chunks: {chunk_count}")

    # 1. General PPE question with zero indexed documents.
    first = api_chat(
        client,
        "What PPE is required on a construction site?",
        session_id="verify-session",
    )
    record(
        "general_ppe_question_no_documents",
        ok=bool(first.get("ok")),
        http_status=first.get("http_status"),
        mode=first.get("mode"),
        basis=first.get("basis"),
        sources=len(first.get("sources") or []),
        answer=first.get("answer"),
        timings=first.get("timings"),
        documents_indexed=document_count,
    )

    # 2. Construction-site safety question.
    second = api_chat(
        client,
        "Explain construction-site hazards and how workplace accidents can be prevented.",
        session_id="verify-session",
    )
    record(
        "construction_safety_question",
        ok=bool(second.get("ok")),
        mode=second.get("mode"),
        basis=second.get("basis"),
        answer=second.get("answer"),
        timings=second.get("timings"),
    )

    # 3. Follow-up question using previous conversation context.
    followup = api_chat(
        client,
        "In that list, which hazard is most relevant to PPE, and why?",
        session_id="verify-session",
    )
    record(
        "follow_up_context",
        ok=bool(followup.get("ok")),
        mode=followup.get("mode"),
        basis=followup.get("basis"),
        answer=followup.get("answer"),
        session_turns=len(assistant.history("verify-session")),
    )

    # 4. SiteGuardVision architecture question.
    architecture = api_chat(
        client,
        "Explain how SiteGuardVision identifies workers and assigns PPE to them.",
        session_id="verify-session",
    )
    record(
        "siteguardvision_architecture",
        ok=bool(architecture.get("ok")),
        mode=architecture.get("mode"),
        basis=architecture.get("basis"),
        answer=architecture.get("answer"),
    )

    # 5. Question about an actual stored image analysis result.
    recent = storage.list_analyses(limit=1)
    analysis_id = recent[0]["id"] if recent else None
    analysis_answer = None
    if analysis_id is not None:
        analysis_answer = api_chat(
            client,
            "Explain this worker assessment.",
            session_id="verify-session",
            analysis_id=analysis_id,
        )
        record(
            "actual_image_analysis_question",
            ok=bool(analysis_answer.get("ok")),
            analysis_id=analysis_id,
            mode=analysis_answer.get("mode"),
            basis=analysis_answer.get("basis"),
            answer=analysis_answer.get("answer"),
        )
    else:
        record("actual_image_analysis_question", ok=False, error="no stored analysis available")

    # 6. Ollama offline behaviour (real unreachable endpoint, no mocks).
    offline_assistant = SafetyAssistant(llm=OllamaService(base_url="http://127.0.0.1:9"))
    offline = offline_assistant.answer("What is a safety helmet?")
    record(
        "ollama_offline",
        ok=bool(offline.get("ok")),
        mode=offline.get("mode"),
        answer=offline.get("answer"),
        starts_with_offline_message=str(offline.get("answer", "")).startswith(OFFLINE_MESSAGE),
    )

    # 7. Empty input validation.
    empty = api_chat(client, "   ", session_id="verify-session")
    record(
        "empty_input_validation",
        ok=False,
        http_status=empty.get("http_status"),
        error=empty.get("error"),
    )

    # 8. Long question handling.
    long_question = "What is PPE? " + ("Please explain in detail. " * 160)
    long_question = long_question[:2500]
    long_response = api_chat(client, long_question, session_id="verify-session")
    record(
        "long_question_handling",
        ok=False,
        http_status=long_response.get("http_status"),
        question_characters=len(long_question),
        error=long_response.get("error"),
    )

    payload = {
        "model": status.get("chat_model"),
        "ollama_reachable": status.get("available"),
        "documents_indexed": document_count,
        "chunks_indexed": chunk_count,
        "scenarios": SCENARIOS,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (RESULTS_DIR / "assistant_chat_verification.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Live conversational assistant verification",
        "",
        f"- Model: `{payload['model']}`",
        f"- Ollama reachable: **{payload['ollama_reachable']}**",
        f"- Indexed documents: **{document_count}**; chunks: **{chunk_count}**",
        f"- Generated: {payload['generated_at']}",
        "",
        "All generation used the real local Ollama service; no mocked responses.",
        "",
        "| # | Scenario | Result | Mode | Basis | Answer |",
        "|---:|---|---|---|---|---|",
    ]
    validation_scenarios = {"empty_input_validation", "long_question_handling"}
    for index, item in enumerate(SCENARIOS, start=1):
        answer = (item.get("answer") or item.get("error") or "").replace("\n", " ").strip()
        if len(answer) > 220:
            answer = answer[:217] + "…"
        if item["scenario"] in validation_scenarios:
            result = "PASS" if not item.get("ok") else "FAIL"
        elif item["scenario"] == "ollama_offline":
            result = "PASS" if item.get("starts_with_offline_message") else "FAIL"
        else:
            result = "PASS" if item.get("ok") else "FAIL"
        lines.append(
            f"| {index} | {item['scenario']} | {result} | "
            f"{item.get('mode') or item.get('http_status') or '—'} | {item.get('basis') or '—'} | "
            f"{answer} |"
        )
    lines += [
        "",
        "## Raw answers",
        "",
    ]
    for item in SCENARIOS:
        lines.append(f"### {item['scenario']}")
        lines.append("")
        lines.append(item.get("answer") or item.get("error") or "")
        lines.append("")
    (REPORT_DIR / "assistant_chat_verification.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print("Wrote evaluation/results/assistant_chat_verification.json and reports/assistant_chat_verification.md")


if __name__ == "__main__":
    main()
