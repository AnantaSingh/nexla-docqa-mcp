"""Thin ChromaDB wrapper for persistent, metadata-filterable vector search.

We pass precomputed embeddings in and out (embedding_function=None) so the embedding
provider stays fully under our control in `embeddings.py`. Cosine space is used to match
normalized text embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import Settings

_COLLECTION = "docqa"


@dataclass
class VectorHit:
    id: str
    text: str
    metadata: dict
    score: float  # cosine similarity in [-1, 1]; higher is better


class VectorStore:
    def __init__(self, settings: Settings):
        settings.chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )

    def reset_collection(self):
        try:
            self._client.delete_collection(_COLLECTION)
        except Exception:
            pass
        return self._collection()

    def _collection(self):
        return self._client.get_or_create_collection(
            _COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
        batch_size: int = 512,
    ):
        col = self._collection()
        for i in range(0, len(ids), batch_size):
            sl = slice(i, i + batch_size)
            col.add(
                ids=ids[sl],
                embeddings=embeddings[sl],
                documents=documents[sl],
                metadatas=metadatas[sl],
            )

    def count(self) -> int:
        return self._collection().count()

    def query(
        self, embedding: list[float], n_results: int, where: dict | None = None
    ) -> list[VectorHit]:
        col = self._collection()
        res = col.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[VectorHit] = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append(VectorHit(id=cid, text=doc, metadata=meta, score=1.0 - dist))
        return hits
