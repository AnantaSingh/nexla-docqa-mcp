"""Per-document text store for deterministic, non-RAG document statistics.

RAG is structurally bad at "how many pages does the document have?" or "how many times is
X mentioned?" — those are *computations over the whole document*, not retrieval. This store
keeps each document's full page text (built once at ingest) so those questions are answered
exactly, by counting, instead of being guessed by the LLM.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .config import Settings
from .documents import doc_meta_for
from .pdf_parser import ParsedDocument


@dataclass
class DocStats:
    file_name: str
    company: str
    ticker: str
    year: int
    page_count: int
    word_count: int
    term: str | None = None
    term_count: int | None = None
    term_pages: list[int] | None = None


def build_doc_records(parsed_docs: list[ParsedDocument]) -> dict:
    records = {}
    for pd in parsed_docs:
        m = pd.meta
        records[m.file_name] = {
            "company": m.company,
            "ticker": m.ticker,
            "year": m.year,
            "page_count": pd.page_count,
            "pages": [p.text for p in pd.pages],
        }
    return records


def save_doc_records(settings: Settings, records: dict) -> None:
    settings.documents_path.write_text(json.dumps(records))


class DocStore:
    def __init__(self, settings: Settings):
        if not settings.documents_path.exists():
            raise RuntimeError("Document store not found. Run `python -m docqa.ingest`.")
        self._docs = json.loads(settings.documents_path.read_text())

    def has(self, file_name: str) -> bool:
        return file_name in self._docs

    def stats(self, file_name: str, term: str | None = None) -> DocStats:
        d = self._docs[file_name]
        pages: list[str] = d["pages"]
        meta = doc_meta_for(file_name)
        out = DocStats(
            file_name=file_name,
            company=d["company"],
            ticker=d["ticker"],
            year=d["year"],
            page_count=d["page_count"],
            word_count=sum(len(p.split()) for p in pages),
        )
        if term:
            pat = re.compile(re.escape(term), re.IGNORECASE)
            count = 0
            hit_pages: list[int] = []
            for i, text in enumerate(pages, start=1):
                n = len(pat.findall(text))
                if n:
                    count += n
                    hit_pages.append(i)
            out.term = term
            out.term_count = count
            out.term_pages = hit_pages
        return out
