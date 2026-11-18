"""Persistent ChromaDB storage plus a small ingestion manifest.

ChromaDB is imported lazily: importing it costs several seconds and the PPE
application must start even when the RAG dependencies are missing.  Every
public method raises :class:`RagError` with a user-facing explanation instead
of leaking Chroma internals.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config
from .document_loader import TextChunk

_lock = threading.RLock()
_manifest_lock = threading.Lock()


class RagError(RuntimeError):
    """Base class for assistant/RAG failures."""


class RagDependencyError(RagError):
    """ChromaDB (or another optional dependency) is not installed."""


class RagIndexError(RagError):
    """The vector index exists but cannot serve the current embedding model."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def read_manifest() -> dict[str, Any]:
    with _manifest_lock:
        path = config.MANIFEST_PATH
        if not path.is_file():
            return {"embedding_model": None, "documents": {}, "updated_at": None}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {"embedding_model": None, "documents": {}, "updated_at": None}
    if not isinstance(data, dict):
        return {"embedding_model": None, "documents": {}, "updated_at": None}
    data.setdefault("documents", {})
    data.setdefault("embedding_model", None)
    data.setdefault("updated_at", None)
    return data


def write_manifest(manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    with _manifest_lock:
        path = config.MANIFEST_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)


class VectorStore:
    def __init__(self, path: str | Path | None = None, collection_name: str | None = None) -> None:
        self.path = Path(path or config.VECTOR_DB_DIR)
        self.collection_name = collection_name or config.COLLECTION_NAME
        self._client = None
        self._collection = None
        self._dependency_error: str | None = None

    # -- lifecycle ----------------------------------------------------------

    def _load(self):
        with _lock:
            if self._collection is not None:
                return self._collection
            if self._dependency_error is not None:
                raise RagDependencyError(self._dependency_error)
            try:
                import chromadb
                from chromadb.config import Settings
            except ImportError as exc:
                self._dependency_error = (
                    "ChromaDB is not installed. Run `pip install -r requirements.txt` "
                    "to enable document retrieval."
                )
                raise RagDependencyError(self._dependency_error) from exc

            try:
                self.path.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self.path),
                    settings=Settings(anonymized_telemetry=False, allow_reset=False),
                )
                self._collection = self._client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"},
                    embedding_function=None,
                )
            except Exception as exc:  # noqa: BLE001 - surfaced to the UI
                raise RagIndexError(f"Could not open the vector database: {exc}") from exc
            return self._collection

    def count(self) -> int:
        collection = self._load()
        try:
            return int(collection.count())
        except Exception as exc:  # noqa: BLE001
            raise RagIndexError(f"Could not read the vector index: {exc}") from exc

    def collection_embedding_model(self) -> str | None:
        collection = self._load()
        try:
            metadata = collection.metadata or {}
            value = metadata.get("embedding_model")
            return str(value) if value else None
        except Exception:  # noqa: BLE001
            return None

    # -- writes -------------------------------------------------------------

    def replace_document(
        self,
        *,
        filename: str,
        doc_hash: str,
        embedding_model: str,
        chunks: list[TextChunk],
        embeddings: list[list[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise RagIndexError("Chunk and embedding counts do not match.")
        collection = self._load()
        try:
            collection.delete(where={"filename": filename})
        except Exception:
            # Deleting from an empty collection is harmless on most versions;
            # a genuine storage failure is caught by the add below.
            pass

        ids = [_chunk_id(filename, chunk.chunk_index) for chunk in chunks]
        metadatas = [
            {
                "filename": filename,
                "page": int(chunk.page or 0),
                "chunk_index": int(chunk.chunk_index),
                "doc_hash": doc_hash,
                "embedding_model": embedding_model,
            }
            for chunk in chunks
        ]
        try:
            collection.upsert(
                ids=ids,
                documents=[chunk.text for chunk in chunks],
                embeddings=embeddings,
                metadatas=metadatas,
            )
            try:
                collection.modify(metadata={"hnsw:space": "cosine", "embedding_model": embedding_model})
            except Exception:
                pass
        except Exception as exc:  # noqa: BLE001
            raise RagIndexError(f"Could not write to the vector index: {exc}") from exc
        return len(ids)

    def delete_document(self, filename: str) -> None:
        collection = self._load()
        try:
            collection.delete(where={"filename": filename})
        except Exception as exc:  # noqa: BLE001
            raise RagIndexError(f"Could not remove the document from the index: {exc}") from exc

    def reset(self, embedding_model: str) -> None:
        with _lock:
            client = self._client
            if client is None:
                try:
                    self._load()
                except RagDependencyError:
                    raise
                client = self._client
            try:
                client.delete_collection(self.collection_name)
            except Exception:
                pass
            self._collection = None
            self._client = client
            collection = self._load()
            try:
                collection.modify(metadata={"hnsw:space": "cosine", "embedding_model": embedding_model})
            except Exception:
                pass

    # -- reads --------------------------------------------------------------

    def query(self, embedding: list[float], top_k: int) -> list[dict[str, Any]]:
        collection = self._load()
        try:
            payload = collection.query(
                query_embeddings=[embedding],
                n_results=max(1, int(top_k)),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:  # noqa: BLE001 - includes dimension mismatch
            message = str(exc)
            if "dimension" in message.lower():
                raise RagIndexError(
                    "The vector index was built with a different embedding model. "
                    "Re-index the knowledge base from Settings."
                ) from exc
            raise RagIndexError(f"Vector search failed: {exc}") from exc

        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]
        results: list[dict[str, Any]] = []
        for text, metadata, distance in zip(documents, metadatas, distances):
            metadata = metadata or {}
            results.append(
                {
                    "text": str(text or ""),
                    "filename": str(metadata.get("filename") or "unknown"),
                    "page": int(metadata.get("page") or 0) or None,
                    "chunk_index": int(metadata.get("chunk_index") or 0),
                    "distance": float(distance) if distance is not None else 1.0,
                    "similarity": round(max(0.0, 1.0 - float(distance or 1.0)), 4),
                }
            )
        return results


def _chunk_id(filename: str, chunk_index: int) -> str:
    raw = f"{filename}::{chunk_index}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:32]


def manifest_document_state(manifest: dict[str, Any], filename: str, digest: str) -> bool:
    entry = (manifest.get("documents") or {}).get(filename)
    return bool(entry and entry.get("sha256") == digest)


def list_documents() -> list[dict[str, Any]]:
    """Return manifest records joined with files that are present on disk."""
    manifest = read_manifest()
    documents = manifest.get("documents") or {}
    known: list[dict[str, Any]] = []
    for filename, entry in sorted(documents.items()):
        stored = config.KNOWLEDGE_BASE_DIR / filename
        known.append(
            {
                "filename": filename,
                "chunks": int(entry.get("chunks") or 0),
                "pages": entry.get("pages"),
                "size": int(entry.get("size") or 0),
                "indexed_at": entry.get("indexed_at"),
                "present": stored.is_file(),
                "sha256": entry.get("sha256"),
            }
        )
    return known


def pending_documents() -> list[dict[str, Any]]:
    """Files in knowledge_base/documents that are not in the manifest."""
    manifest = read_manifest()
    documents = manifest.get("documents") or {}
    pending: list[dict[str, Any]] = []
    if not config.KNOWLEDGE_BASE_DIR.is_dir():
        return pending
    for path in sorted(config.KNOWLEDGE_BASE_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in config.ALLOWED_DOCUMENT_EXTENSIONS:
            continue
        if path.name not in documents:
            pending.append({"filename": path.name, "size": path.stat().st_size})
    return pending


def iter_document_files() -> Iterable[Path]:
    if not config.KNOWLEDGE_BASE_DIR.is_dir():
        return []
    return [
        path
        for path in sorted(config.KNOWLEDGE_BASE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in config.ALLOWED_DOCUMENT_EXTENSIONS
    ]
