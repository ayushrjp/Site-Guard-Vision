"""Similarity retrieval over the persisted safety-document index."""

from __future__ import annotations

from dataclasses import dataclass

from . import config
from .embeddings import OllamaEmbeddings
from .vector_store import RagError, VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    filename: str
    page: int | None
    chunk_index: int
    similarity: float

    @property
    def label(self) -> str:
        if self.page:
            return f"{self.filename} — page {self.page}"
        return self.filename


class Retriever:
    def __init__(
        self,
        embeddings: OllamaEmbeddings,
        store: VectorStore,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.store = store
        self.top_k = int(top_k or config.TOP_K)
        self.min_similarity = float(
            config.MIN_SIMILARITY if min_similarity is None else min_similarity
        )

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        if not question.strip():
            return []
        if self.store.count() == 0:
            return []
        vector = self.embeddings.embed([question])[0]
        matches = self.store.query(vector, self.top_k)
        chunks = [
            RetrievedChunk(
                text=match["text"],
                filename=match["filename"],
                page=match["page"],
                chunk_index=match["chunk_index"],
                similarity=float(match["similarity"]),
            )
            for match in matches
            if match["text"].strip() and float(match["similarity"]) >= self.min_similarity
        ]
        return chunks
