"""Ollama health checks and chat generation.

The app must keep working when Ollama is not installed.  Therefore every call
is defensive: short status timeouts, explicit model-availability checks, and
errors returned as data rather than raised out of Flask.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

from . import config


class OllamaError(RuntimeError):
    pass


class OllamaService:
    def __init__(
        self,
        base_url: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
        timeout: float | None = None,
        status_timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = (base_url or config.OLLAMA_BASE_URL).rstrip("/")
        self.chat_model = chat_model or config.OLLAMA_CHAT_MODEL
        self.embed_model = embed_model or config.OLLAMA_EMBED_MODEL
        self.timeout = float(timeout or config.OLLAMA_TIMEOUT)
        self.status_timeout = float(status_timeout or config.OLLAMA_STATUS_TIMEOUT)
        self._session = session or requests.Session()
        self._lock = threading.Lock()
        self._chat_lock = threading.Lock()
        self._status_cache: dict[str, Any] | None = None
        self._status_at = 0.0
        self._cache_seconds = 8.0

    # -- health -------------------------------------------------------------

    def status(self, force: bool = False, chat_model: str | None = None) -> dict[str, Any]:
        model = chat_model or self.chat_model
        now = time.monotonic()
        with self._lock:
            cached = self._status_cache
            if (
                not force
                and cached is not None
                and cached.get("chat_model") == model
                and (now - self._status_at) < self._cache_seconds
            ):
                return dict(cached)

        result: dict[str, Any] = {
            "reachable": False,
            "base_url": self.base_url,
            "chat_model": model,
            "chat_model_available": False,
            "embed_model": self.embed_model,
            "embed_model_available": False,
            "installed_models": [],
            "error": None,
        }
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags", timeout=self.status_timeout
            )
            if response.status_code != 200:
                result["error"] = (
                    f"Ollama responded with HTTP {response.status_code} at {self.base_url}."
                )
            else:
                payload = response.json()
                models = payload.get("models") or []
                names = sorted(
                    {
                        str(entry.get("name") or entry.get("model") or "")
                        for entry in models
                        if isinstance(entry, dict)
                    }
                    - {""}
                )
                result.update(
                    {
                        "reachable": True,
                        "installed_models": names,
                        "chat_model_available": _model_present(names, model),
                        "embed_model_available": _model_present(names, self.embed_model),
                    }
                )
        except requests.Timeout:
            result["error"] = (
                f"Ollama did not answer within {self.status_timeout:.1f}s at {self.base_url}."
            )
        except requests.RequestException as exc:
            result["error"] = f"Ollama is not reachable at {self.base_url}. {exc}"
        except ValueError:
            result["error"] = "Ollama returned an unreadable model list."

        with self._lock:
            self._status_cache = dict(result)
            self._status_at = time.monotonic()
        return result

    def installed_models(self) -> list[str]:
        return list(self.status(force=True).get("installed_models") or [])

    # -- generation ---------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.2,
        num_predict: int = 600,
    ) -> tuple[str, float]:
        """Return (answer, elapsed_ms); raise OllamaError with a clear message."""
        selected = model or self.chat_model
        status = self.status(chat_model=selected)
        if not status.get("reachable"):
            raise OllamaError(status.get("error") or "Ollama is not reachable.")
        if not _model_present(status.get("installed_models") or [], selected):
            raise OllamaError(
                f"Ollama model `{selected}` is not installed. Run `ollama pull {selected}`."
            )

        started = time.perf_counter()
        with self._chat_lock:
            try:
                response = self._session.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": selected,
                        "messages": messages,
                        "stream": False,
                        "keep_alive": "10m",
                        "options": {
                            "temperature": temperature,
                            "num_predict": num_predict,
                            "num_ctx": 4096,
                        },
                    },
                    timeout=self.timeout,
                )
            except requests.Timeout as exc:
                raise OllamaError(
                    f"Ollama did not finish within {self.timeout:.0f}s. "
                    "The first response can be slow while the model loads."
                ) from exc
            except requests.RequestException as exc:
                raise OllamaError(f"Could not reach Ollama at {self.base_url}: {exc}") from exc

        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)

        if response.status_code == 404:
            raise OllamaError(
                f"Ollama model `{selected}` is unavailable. Run `ollama pull {selected}`."
            )
        if response.status_code != 200:
            detail = ""
            try:
                detail = str((response.json() or {}).get("error") or "")
            except ValueError:
                detail = (response.text or "").strip()[:300]
            raise OllamaError(
                f"Ollama chat failed with HTTP {response.status_code}: {detail or 'no detail'}"
            )

        try:
            payload = response.json()
            content = str(payload["message"]["content"]).strip()
        except (ValueError, KeyError, TypeError) as exc:
            raise OllamaError("Ollama returned an unexpected chat payload.") from exc
        if not content:
            raise OllamaError("Ollama returned an empty answer.")
        return content, elapsed_ms


def _model_present(installed: list[str], requested: str) -> bool:
    """Match exact names and tolerate the implicit ``:latest`` tag."""
    if not requested:
        return False
    requested = requested.strip()
    if requested in installed:
        return True
    # Ollama treats a missing tag as :latest, but explicit tags must match.
    return ":" not in requested and f"{requested}:latest" in installed
