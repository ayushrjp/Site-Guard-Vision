"""Ollama embedding client used by the RAG index.

The client uses the modern ``/api/embed`` endpoint when available and falls
back to the older per-text ``/api/embeddings`` endpoint.  Network failures are
converted into :class:`OllamaError` messages the UI can display verbatim.
"""

from __future__ import annotations

from typing import Any, Iterable

import requests

from . import config


class OllamaError(RuntimeError):
    """Human-readable failure talking to Ollama."""


class OllamaEmbeddings:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or config.OLLAMA_EMBED_MODEL
        self.timeout = float(timeout or config.OLLAMA_TIMEOUT)
        self._session = session or requests.Session()

    def embed(self, texts: Iterable[str], timeout: float | None = None) -> list[list[float]]:
        """Embed a batch of texts, preserving input order."""
        batch = [str(text) for text in texts]
        if not batch:
            return []
        if not any(text.strip() for text in batch):
            raise OllamaError("Cannot embed empty text.")

        request_timeout = float(timeout or self.timeout)
        try:
            response = self._session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model, "input": batch},
                timeout=request_timeout,
            )
        except requests.Timeout as exc:
            raise OllamaError(
                f"Embedding timed out after {request_timeout:.0f}s. Ollama may still be loading the model."
            ) from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code == 404:
            # Older Ollama builds expose only /api/embeddings.
            return [self._embed_single(text, request_timeout) for text in batch]
        if response.status_code != 200:
            raise OllamaError(_response_error(response, "embedding"))

        try:
            payload: dict[str, Any] = response.json()
        except ValueError as exc:
            raise OllamaError("Ollama returned a non-JSON embedding response.") from exc

        embeddings = payload.get("embeddings")
        if not isinstance(embeddings, list) or len(embeddings) != len(batch):
            raise OllamaError("Ollama returned an unexpected embedding payload.")
        return [[float(value) for value in vector] for vector in embeddings]

    def _embed_single(self, text: str, timeout: float) -> list[float]:
        try:
            response = self._session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=timeout,
            )
        except requests.Timeout as exc:
            raise OllamaError(f"Embedding timed out after {timeout:.0f}s.") from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code != 200:
            raise OllamaError(_response_error(response, "embedding"))
        try:
            payload = response.json()
            vector = payload.get("embedding")
        except (ValueError, AttributeError) as exc:
            raise OllamaError("Ollama returned an unexpected embedding payload.") from exc
        if not isinstance(vector, list) or not vector:
            raise OllamaError("Ollama returned an empty embedding vector.")
        return [float(value) for value in vector]


def _response_error(response: requests.Response, action: str) -> str:
    detail = ""
    try:
        body = response.json()
        detail = str(body.get("error") or body.get("message") or "")
    except ValueError:
        detail = (response.text or "").strip()[:300]
    if response.status_code == 404:
        tail = f" ({detail})" if detail else ""
        return (
            f"Ollama could not run the {action} model. Install it with "
            f"`ollama pull {config.OLLAMA_EMBED_MODEL}`{tail}."
        )
    return f"Ollama {action} failed with HTTP {response.status_code}: {detail or 'no detail'}"
