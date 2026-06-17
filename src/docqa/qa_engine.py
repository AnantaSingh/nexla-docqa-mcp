"""Orchestration: retrieve -> ground -> answer, with faithful, attributable citations.

This is the heart of the system and the place where "accuracy and robustness" is enforced:
  - A token budget bounds how much context the LLM sees (predictable cost, fits the window).
  - The LLM is handed labelled SOURCES and can only cite those labels; the engine maps the
    labels back to real chunk metadata, so every citation points at a real page in a real doc.
  - If retrieval is empty or the LLM abstains, we return a clear "not found" instead of guessing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import tiktoken

from .config import Settings, get_settings
from .llm import AnswerLLM, ClaudeAnswerLLM
from .retriever import RetrievedChunk, Retriever, resolve_document_filter

_ENC = tiktoken.get_encoding("cl100k_base")
_SNIPPET_CHARS = 240


@dataclass
class Citation:
    label: str  # the SOURCE label shown to the LLM, e.g. "S1"
    company: str
    ticker: str
    year: int
    file_name: str
    page: int
    section: str
    chunk_type: str
    snippet: str


@dataclass
class QAResult:
    question: str
    answer: str
    answer_found: bool
    citations: list[Citation] = field(default_factory=list)
    retrieved_count: int = 0
    document_filter: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["citations"] = [asdict(c) for c in self.citations]
        return d


def _snippet(text: str) -> str:
    s = " ".join(text.split())
    return s[:_SNIPPET_CHARS] + ("…" if len(s) > _SNIPPET_CHARS else "")


def _source_header(c: RetrievedChunk, label: str) -> str:
    m = c.metadata
    sec = f", {m['section']}" if m.get("section") else ""
    return f"[{label}] {m['company']} (FY{m['year']}) — p.{m['page_start']}{sec} [{m['chunk_type']}]"


class QAEngine:
    def __init__(
        self,
        settings: Settings | None = None,
        retriever: Retriever | None = None,
        llm: AnswerLLM | None = None,
    ):
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(self.settings)
        self.llm = llm or ClaudeAnswerLLM(self.settings)

    def _build_sources(self, chunks: list[RetrievedChunk]) -> tuple[str, dict[str, RetrievedChunk]]:
        """Label chunks S1..Sn within the token budget; return the block + label->chunk map."""
        budget = self.settings.max_context_tokens
        blocks: list[str] = []
        label_map: dict[str, RetrievedChunk] = {}
        used = 0
        for i, c in enumerate(chunks, start=1):
            label = f"S{i}"
            block = f"{_source_header(c, label)}\n{c.text}"
            cost = len(_ENC.encode(block))
            if blocks and used + cost > budget:
                break
            blocks.append(block)
            label_map[label] = c
            used += cost
        return "\n\n".join(blocks), label_map

    def _citations(
        self, used_labels: list[str], label_map: dict[str, RetrievedChunk]
    ) -> list[Citation]:
        out: list[Citation] = []
        seen: set[str] = set()
        for label in used_labels:
            c = label_map.get(label.strip())
            if c is None or c.id in seen:
                continue
            seen.add(c.id)
            m = c.metadata
            out.append(
                Citation(
                    label=label.strip(),
                    company=m["company"],
                    ticker=m["ticker"],
                    year=m["year"],
                    file_name=m["file_name"],
                    page=m["page_start"],
                    section=m.get("section", ""),
                    chunk_type=m["chunk_type"],
                    snippet=_snippet(c.text),
                )
            )
        return out

    def answer(
        self, question: str, top_k: int | None = None, document: str | None = None
    ) -> QAResult:
        if not question or not question.strip():
            return QAResult(question=question or "", answer="Please provide a question.",
                            answer_found=False)
        try:
            where = resolve_document_filter(document)
        except ValueError as e:
            return QAResult(question=question, answer=str(e), answer_found=False,
                            document_filter=document)

        chunks = self.retriever.retrieve(question, top_n=top_k, where=where)
        if not chunks:
            return QAResult(
                question=question,
                answer="No relevant content was found in the indexed documents.",
                answer_found=False,
                retrieved_count=0,
                document_filter=document,
            )

        sources_block, label_map = self._build_sources(chunks)
        result = self.llm.generate(question, sources_block)

        citations = (
            self._citations(result.used_sources, label_map) if result.answer_found else []
        )
        # If the model answered but cited nothing valid, attach the top source for provenance.
        if result.answer_found and not citations:
            top_label = next(iter(label_map))
            citations = self._citations([top_label], label_map)

        return QAResult(
            question=question,
            answer=result.answer,
            answer_found=result.answer_found,
            citations=citations,
            retrieved_count=len(label_map),
            document_filter=document,
        )
