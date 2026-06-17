"""Embedding providers behind a tiny interface so the backend is swappable.

Default: OpenAI `text-embedding-3-small` (high retrieval quality, tiny dependency).
Fallback: local `bge-small-en-v1.5` via fastembed (ONNX, no torch) for fully-offline use.

Symmetric vs asymmetric:
  - OpenAI embeddings are symmetric -> queries and documents are embedded the same way.
  - bge models are asymmetric -> queries get the recommended instruction prefix. That logic
    lives here so callers never have to think about it.
"""

from __future__ import annotations

from typing import Protocol

from .config import Settings

_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class OpenAIEmbedder:
    """OpenAI embeddings; batched to stay well under request limits."""

    def __init__(self, settings: Settings, batch_size: int = 128):
        from openai import OpenAI

        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to .env or switch EMBED_PROVIDER=fastembed."
            )
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embed_model
        self._batch = batch_size

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self._batch):
            batch = [t if t.strip() else " " for t in texts[i : i + self._batch]]
            resp = self._client.embeddings.create(model=self._model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]


class FastEmbedEmbedder:
    """Local, offline embeddings via fastembed (ONNX bge-small). No API key required."""

    def __init__(self, settings: Settings):
        from fastembed import TextEmbedding

        self._model = TextEmbedding(
            model_name=settings.local_embed_model,
            cache_dir=str(settings.fastembed_cache_path),
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [v.tolist() for v in self._model.embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        return next(iter(self._model.embed([_BGE_QUERY_PREFIX + text]))).tolist()


def build_embedder(settings: Settings) -> Embedder:
    provider = settings.embed_provider.lower()
    if provider == "openai":
        return OpenAIEmbedder(settings)
    if provider == "fastembed":
        return FastEmbedEmbedder(settings)
    raise ValueError(f"Unknown EMBED_PROVIDER={settings.embed_provider!r} (use openai|fastembed)")
