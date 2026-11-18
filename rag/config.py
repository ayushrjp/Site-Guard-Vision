"""Configuration for the local Ollama + RAG Safety Assistant.

Everything is environment-configurable so the assistant can run against a
different Ollama host, models, or vector store without code changes.  Defaults
match the project brief: qwen2.5:3b for generation and nomic-embed-text for
embeddings.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5:3b")
OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "300"))
OLLAMA_STATUS_TIMEOUT = float(os.environ.get("OLLAMA_STATUS_TIMEOUT", "2.5"))

VECTOR_DB_DIR = Path(
    os.environ.get("SITEGUARD_VECTOR_DB", str(ROOT_DIR / "vector_db"))
)
KNOWLEDGE_BASE_DIR = Path(
    os.environ.get("SITEGUARD_KNOWLEDGE_BASE", str(ROOT_DIR / "knowledge_base" / "documents"))
)
MANIFEST_PATH = VECTOR_DB_DIR / "manifest.json"

COLLECTION_NAME = os.environ.get("SITEGUARD_COLLECTION", "safety_documents")
TOP_K = int(os.environ.get("SITEGUARD_TOP_K", "5"))
MIN_SIMILARITY = float(os.environ.get("SITEGUARD_MIN_SIMILARITY", "0.18"))
CHUNK_SIZE = int(os.environ.get("SITEGUARD_CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.environ.get("SITEGUARD_CHUNK_OVERLAP", "150"))
MAX_DOCUMENT_BYTES = int(os.environ.get("SITEGUARD_MAX_DOCUMENT_BYTES", str(15 * 1024 * 1024)))
MAX_QUESTION_CHARS = 2000

ALLOWED_DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
