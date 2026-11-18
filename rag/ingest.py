"""Document ingestion: load → chunk → embed → store, with idempotent indexing.

A manifest records SHA-256 hashes so unchanged documents are skipped instead of
re-embedded on every restart.  Re-indexing resets collections built with a
different embedding model, which prevents confusing dimension errors later.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from . import config
from .document_loader import DocumentError, load_document
from .embeddings import OllamaEmbeddings, OllamaError
from .vector_store import (
    RagError,
    RagIndexError,
    VectorStore,
    file_sha256,
    read_manifest,
    write_manifest,
)

EMBED_BATCH_SIZE = 16


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ingest_document(
    path: str | Path,
    *,
    embeddings: OllamaEmbeddings,
    store: VectorStore,
    force: bool = False,
) -> dict[str, Any]:
    """Index one document.  Returns a status dict; raises on hard failures."""
    path = Path(path)
    filename = path.name
    digest = file_sha256(path)
    manifest = read_manifest()

    current_model = manifest.get("embedding_model")
    index_model = store.collection_embedding_model()
    if index_model and index_model != embeddings.model:
        raise RagIndexError(
            "The index was built with embedding model "
            f"`{index_model}` but the configured model is `{embeddings.model}`. "
            "Run a full re-index to rebuild it."
        )

    entry = (manifest.get("documents") or {}).get(filename)
    if (
        not force
        and entry
        and entry.get("sha256") == digest
        and current_model == embeddings.model
    ):
        return {
            "filename": filename,
            "status": "skipped",
            "chunks": int(entry.get("chunks") or 0),
            "message": "Unchanged since the last index.",
        }

    document = load_document(path)
    texts = [chunk.text for chunk in document.chunks]
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(embeddings.embed(texts[start : start + EMBED_BATCH_SIZE]))

    written = store.replace_document(
        filename=filename,
        doc_hash=digest,
        embedding_model=embeddings.model,
        chunks=document.chunks,
        embeddings=vectors,
    )

    manifest = read_manifest()
    manifest.setdefault("documents", {})[filename] = {
        "sha256": digest,
        "chunks": written,
        "pages": document.page_count,
        "size": path.stat().st_size,
        "indexed_at": _now(),
    }
    manifest["embedding_model"] = embeddings.model
    write_manifest(manifest)

    return {
        "filename": filename,
        "status": "indexed",
        "chunks": written,
        "pages": document.page_count,
        "message": f"Indexed {written} chunks.",
    }


def ingest_files(
    paths: Iterable[str | Path],
    *,
    embeddings: OllamaEmbeddings,
    store: VectorStore,
    force: bool = False,
    reset: bool = False,
) -> dict[str, Any]:
    """Index several files, collecting per-file failures without aborting."""
    manifest = read_manifest()
    if reset or (
        manifest.get("embedding_model")
        and manifest.get("embedding_model") != embeddings.model
    ):
        store.reset(embeddings.model)
        manifest = {"embedding_model": embeddings.model, "documents": {}, "updated_at": _now()}
        write_manifest(manifest)
        force = True

    indexed: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for path in paths:
        path = Path(path)
        try:
            outcome = ingest_document(
                path, embeddings=embeddings, store=store, force=force
            )
        except (DocumentError, OllamaError, RagError) as exc:
            failed.append({"filename": path.name, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the batch
            failed.append({"filename": path.name, "error": f"{type(exc).__name__}: {exc}"})
        else:
            (skipped if outcome["status"] == "skipped" else indexed).append(outcome)

    return {
        "indexed": indexed,
        "skipped": skipped,
        "failed": failed,
        "documents": len(read_manifest().get("documents") or {}),
    }


def reindex_all(
    *,
    embeddings: OllamaEmbeddings,
    store: VectorStore,
) -> dict[str, Any]:
    if not config.KNOWLEDGE_BASE_DIR.is_dir():
        config.KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    paths = [
        path
        for path in sorted(config.KNOWLEDGE_BASE_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in config.ALLOWED_DOCUMENT_EXTENSIONS
    ]
    if not paths:
        # Clear stale vectors so deleted files cannot keep being cited.
        store.reset(embeddings.model)
        write_manifest({"embedding_model": embeddings.model, "documents": {}, "updated_at": _now()})
        return {"indexed": [], "skipped": [], "failed": [], "documents": 0}

    return ingest_files(paths, embeddings=embeddings, store=store, force=True, reset=True)


def remove_document(filename: str, *, store: VectorStore) -> None:
    """Delete a knowledge-base file, its vectors and its manifest entry."""
    safe_name = Path(filename).name
    if safe_name != filename:
        raise ValueError("Invalid document name.")
    path = config.KNOWLEDGE_BASE_DIR / safe_name
    try:
        store.delete_document(safe_name)
    except RagError:
        # The file must still be removable when the index is unavailable; a
        # later full re-index will purge orphaned vectors.
        pass
    path.unlink(missing_ok=True)
    manifest = read_manifest()
    (manifest.get("documents") or {}).pop(safe_name, None)
    write_manifest(manifest)
