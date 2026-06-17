"""Orchestration: retrieve -> ground -> answer, with faithful, attributable citations.

This is the heart of the system and the place where "accuracy and robustness" is enforced:
  - A token budget bounds how much context the LLM sees (predictable cost, fits the window).
  - The LLM is handed labelled SOURCES and can only cite those labels; the engine maps the
    labels back to real chunk metadata, so every citation points at a real page in a real doc.
  - If retrieval is empty or the LLM abstains, we return a clear "not found" instead of guessing.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

import tiktoken

from .config import Settings, get_settings
from .doc_store import DocStore
from .llm import AnswerLLM, ClaudeAnswerLLM
from .retriever import RetrievedChunk, Retriever, resolve_document_filter

_ENC = tiktoken.get_encoding("cl100k_base")
_SNIPPET_CHARS = 240

# Deterministic "stats" questions RAG can't do well: page counts and term frequencies.
_PAGE_COUNT_RE = re.compile(r"how many pages", re.IGNORECASE)
_TERM_COUNT_RE = re.compile(r"how many times", re.IGNORECASE)
_QUOTED_RE = re.compile(r"[\"'“”‘’](.+?)[\"'“”‘’]")


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
        self._doc_store: DocStore | None = None

    def _stats_store(self) -> DocStore | None:
        """Lazily load the document-stats store; returns None if it isn't available."""
        if self._doc_store is None:
            try:
                self._doc_store = DocStore(self.settings)
            except RuntimeError:
                return None
        return self._doc_store

    def _maybe_stats_answer(self, question: str, where: dict | None) -> QAResult | None:
        """Answer page-count / term-frequency questions deterministically (no LLM, no guessing)."""
        if not where or "file_name" not in where:
            return None  # need a single document in scope to compute stats
        store = self._stats_store()
        if store is None or not store.has(where["file_name"]):
            return None
        fname = where["file_name"]

        if _PAGE_COUNT_RE.search(question):
            s = store.stats(fname)
            cite = Citation("D1", s.company, s.ticker, s.year, fname, s.page_count,
                            "document length", "metadata", f"{s.company} — {s.page_count} pages.")
            return QAResult(question, f"The {s.company} (FY{s.year}) report has {s.page_count} pages.",
                            True, [cite], retrieved_count=1, document_filter=where["file_name"])

        if _TERM_COUNT_RE.search(question):
            m = _QUOTED_RE.search(question)
            if not m:
                return None  # can't reliably identify the term -> fall back to RAG
            term = m.group(1)
            s = store.stats(fname, term=term)
            pages = ", ".join(map(str, (s.term_pages or [])[:10])) or "no pages"
            cite = Citation("D1", s.company, s.ticker, s.year, fname,
                            (s.term_pages or [0])[0], "term frequency", "metadata",
                            f'"{term}" appears on page(s): {pages}.')
            return QAResult(
                question,
                f'The term "{term}" appears {s.term_count} time(s) in the {s.company} (FY{s.year}) report.',
                True, [cite], retrieved_count=1, document_filter=where["file_name"],
            )
        return None

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

    def _vision_fallback(
        self, question: str, chunks: list[RetrievedChunk], document: str | None
    ) -> QAResult | None:
        """Retry an abstained question by sending the top retrieved pages to Claude vision.

        Only runs when explicitly enabled. Keeps the strict abstain contract, so genuinely
        unanswerable questions still abstain — vision only helps when the answer lives in an
        image/chart the text layer missed.
        """
        import fitz

        # distinct (file, page) from the top retrieved chunks, in order
        seen: set[tuple[str, int]] = set()
        targets: list[tuple[str, int, dict]] = []
        for c in chunks:
            key = (c.metadata["file_name"], c.metadata["page_start"])
            if key not in seen:
                seen.add(key)
                targets.append((key[0], key[1], c.metadata))
            if len(targets) >= self.settings.vision_fallback_max_pages:
                break
        if not targets:
            return None

        images: list[tuple[str, bytes]] = []
        page_meta: dict[str, dict] = {}
        for i, (fname, page, meta) in enumerate(targets, start=1):
            label = f"P{i}"
            try:
                doc = fitz.open(self.settings.data_path / fname)
                png = doc[page - 1].get_pixmap(dpi=150).tobytes("png")
                doc.close()
            except Exception:
                continue
            images.append((f"{label} — {meta['company']} p.{page}", png))
            page_meta[label] = meta | {"page": page}

        if not images or not hasattr(self.llm, "generate_from_images"):
            return None
        result = self.llm.generate_from_images(question, images)
        if not result.answer_found:
            return None

        # cite the page(s) the vision model said it used; if it named none, cite the pages we
        # actually showed it (faithful — never hardcode P1).
        used = [lbl.strip() for lbl in result.used_sources if lbl.strip() in page_meta]
        cited_labels = used or list(page_meta.keys())
        citations = [
            Citation(lbl, m["company"], m["ticker"], m["year"], m["file_name"],
                     m["page"], m.get("section", ""), "figure(vision)", "(read from page image)")
            for lbl in cited_labels for m in [page_meta[lbl]]
        ]
        return QAResult(question, result.answer, True, citations,
                        retrieved_count=len(images), document_filter=document)

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

        stats_answer = self._maybe_stats_answer(question, where)
        if stats_answer is not None:
            return stats_answer

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

        if not result.answer_found and self.settings.vision_fallback_enabled:
            vision = self._vision_fallback(question, chunks, document)
            if vision is not None:
                return vision

        citations = (
            self._citations(result.used_sources, label_map) if result.answer_found else []
        )
        # If the model answered but cited nothing valid, only attach a fallback citation when the
        # retrieved context came from a SINGLE document — then the top source can't point at the
        # wrong doc. With multiple docs in scope we leave citations empty rather than risk a
        # wrong-doc attribution (keeps the "never fabricated" contract in the docstring honest).
        if result.answer_found and not citations:
            docs = {c.metadata["file_name"] for c in label_map.values()}
            if len(docs) == 1:
                citations = self._citations([next(iter(label_map))], label_map)

        return QAResult(
            question=question,
            answer=result.answer,
            answer_found=result.answer_found,
            citations=citations,
            retrieved_count=len(label_map),
            document_filter=document,
        )
