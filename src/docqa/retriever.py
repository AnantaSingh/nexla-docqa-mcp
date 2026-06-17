"""Hybrid retrieval: dense + lexical, fused, then cross-encoder reranked.

Pipeline (recall-first -> precision-second):

    query
      |-- vector search (OpenAI embeddings, cosine)   -> top VECTOR_TOP_K
      '-- BM25 lexical (exact figures / proper nouns)  -> top BM25_TOP_K
            '-- Reciprocal Rank Fusion (RRF_K)          -> fused candidate pool
                  '-- cross-encoder rerank (ms-marco)   -> top RERANK_TOP_N

Why hybrid: embeddings capture paraphrase/semantics; BM25 nails exact numbers and names that
embeddings blur — essential for financial Q&A. RRF combines the two ranked lists without having
to reconcile incompatible score scales. The cross-encoder then reorders for precision.

Metadata filtering is applied consistently to *both* arms so `document=` scoping is correct.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .config import Settings
from .documents import DOCUMENTS
from .embeddings import Embedder, build_embedder
from .vector_store import VectorStore

_POOL_SIZE = 40  # candidates passed into the reranker
_TOKEN_RE = re.compile(r"[a-z]+|\d[\d.,]*")


def _tokenize(text: str) -> list[str]:
    # keep numeric groups intact ("222,730") so exact figures are matchable
    return _TOKEN_RE.findall(text.lower())


def rrf_fuse(rankings: list[list[str]], k: int) -> dict[str, float]:
    """Reciprocal Rank Fusion: combine ranked id-lists into id -> fused score.

    score(id) = sum over lists of 1 / (k + rank), with rank starting at 1. Pure function
    (no model state) so the fusion behaviour is unit-testable in isolation.
    """
    scores: dict[str, float] = {}
    for ranked in rankings:
        for rank, cid in enumerate(ranked, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    vector_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float = 0.0
    rerank_score: float | None = None

    @property
    def score(self) -> float:
        """Final ordering score: rerank if available, else fusion score."""
        return self.rerank_score if self.rerank_score is not None else self.rrf_score


def resolve_document_filter(value: str | None) -> dict | None:
    """Map a user-supplied document hint (company / ticker / file) to a Chroma where clause."""
    if not value:
        return None
    v = value.strip().lower()
    for fname, meta in DOCUMENTS.items():
        if v in (fname.lower(), meta.ticker.lower(), meta.company.lower()) or v in meta.company.lower():
            return {"file_name": fname}
    # last resort: substring match on file name
    for fname in DOCUMENTS:
        if v in fname.lower():
            return {"file_name": fname}
    raise ValueError(
        f"No indexed document matches {value!r}. "
        f"Known: {', '.join(m.ticker for m in DOCUMENTS.values())}."
    )


class Retriever:
    def __init__(self, settings: Settings, embedder: Embedder | None = None):
        self.settings = settings
        if not settings.chunks_path.exists():
            raise RuntimeError(
                "Index not found. Run `python -m docqa.ingest` before querying."
            )
        records = json.loads(settings.chunks_path.read_text())
        self.ids = [r["id"] for r in records]
        self.texts = [r["text"] for r in records]
        self.metas = [{k: v for k, v in r.items() if k != "text"} for r in records]
        self._by_id = {r["id"]: i for i, r in enumerate(records)}
        self._bm25 = BM25Okapi([_tokenize(t) for t in self.texts])
        self._store = VectorStore(settings)
        self._embedder = embedder or build_embedder(settings)
        self._reranker = None  # lazy: only loaded if reranking is enabled

    # -- lexical arm -------------------------------------------------------
    def _bm25_hits(self, query: str, k: int, where: dict | None) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(_tokenize(query))
        idxs = range(len(scores))
        if where and "file_name" in where:
            target = where["file_name"]
            idxs = [i for i in idxs if self.metas[i]["file_name"] == target]
        ranked = sorted(idxs, key=lambda i: scores[i], reverse=True)
        return [(self.ids[i], float(scores[i])) for i in ranked[:k] if scores[i] > 0]

    # -- fusion ------------------------------------------------------------
    def _rrf(
        self, vlist: list[tuple[str, float]], blist: list[tuple[str, float]]
    ) -> dict[str, RetrievedChunk]:
        fused = rrf_fuse([[cid for cid, _ in vlist], [cid for cid, _ in blist]], self.settings.rrf_k)
        vscore = dict(vlist)
        bscore = dict(blist)
        pool: dict[str, RetrievedChunk] = {}
        for cid, score in fused.items():
            i = self._by_id[cid]
            pool[cid] = RetrievedChunk(
                id=cid,
                text=self.texts[i],
                metadata=self.metas[i],
                vector_score=vscore.get(cid),
                bm25_score=bscore.get(cid),
                rrf_score=score,
            )
        return pool

    # -- reranking ---------------------------------------------------------
    def _rerank(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if self._reranker is None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            self._reranker = TextCrossEncoder(
                model_name=self.settings.rerank_model,
                cache_dir=str(self.settings.fastembed_cache_path),
            )
        scores = list(self._reranker.rerank(query, [c.text for c in candidates]))
        for c, s in zip(candidates, scores):
            c.rerank_score = float(s)
        return sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

    # -- public API --------------------------------------------------------
    def retrieve(
        self, query: str, top_n: int | None = None, where: dict | None = None
    ) -> list[RetrievedChunk]:
        if not query or not query.strip():
            return []
        top_n = top_n or self.settings.rerank_top_n

        qemb = self._embedder.embed_query(query)
        vhits = self._store.query(qemb, self.settings.vector_top_k, where=where)
        vlist = [(h.id, h.score) for h in vhits if h.id in self._by_id]
        blist = self._bm25_hits(query, self.settings.bm25_top_k, where)

        pool = self._rrf(vlist, blist)
        candidates = sorted(pool.values(), key=lambda c: c.rrf_score, reverse=True)[:_POOL_SIZE]

        if self.settings.rerank_enabled and candidates:
            candidates = self._rerank(query, candidates)
        return candidates[:top_n]
