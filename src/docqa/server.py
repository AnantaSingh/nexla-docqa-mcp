"""MCP server exposing grounded Q&A over the indexed annual reports.

Transport: stdio (standard for local MCP clients like Claude Desktop / MCP Inspector).

Tools:
  - query_documents : natural-language question -> grounded answer + source citations
  - list_documents  : what's in the corpus (company, year, pages, chunk count)
  - search_chunks   : raw hybrid-retrieval hits (no LLM) for transparency / debugging

The heavy objects (index, embedder, reranker, LLM client) are built lazily on first use so
the process starts instantly and surfaces a clear, actionable error if the index is missing.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from mcp.server.fastmcp import FastMCP

from .config import get_settings
from .documents import DOCUMENTS
from .qa_engine import QAEngine
from .retriever import resolve_document_filter

mcp = FastMCP("nexla-docqa")

_engine: QAEngine | None = None


def get_engine() -> QAEngine:
    global _engine
    if _engine is None:
        _engine = QAEngine(get_settings())
    return _engine


@mcp.tool()
def query_documents(
    question: str, top_k: int = 8, document: str | None = None
) -> dict[str, Any]:
    """Answer a natural-language question grounded in the indexed annual reports.

    Args:
        question: The natural-language question to answer.
        top_k: How many retrieved passages to ground the answer on (default 8).
        document: Optional scope to ONE company's report. Accepts a ticker
            (e.g. "COST"), company name ("Costco"), or file name. Omit to search all.

    Returns:
        A dict with:
          - answer: the grounded answer text
          - answer_found: false when the documents don't support an answer (no guessing)
          - citations: list of {company, year, page, section, chunk_type, snippet, file_name}
          - retrieved_count: how many passages were considered
          - document_filter: the scope applied, if any

    Invalid inputs (unknown `document`, bad types) surface as MCP tool errors (isError) with a
    descriptive message — the single error contract shared by all tools here.
    """
    resolve_document_filter(document)  # raise ValueError -> isError on an unknown document
    result = get_engine().answer(question, top_k=top_k, document=document)
    return result.to_dict()


@mcp.tool()
def list_documents() -> list[dict[str, Any]]:
    """List the indexed documents with company, fiscal year, page count, and chunk count."""
    retriever = get_engine().retriever
    chunk_counts: dict[str, int] = defaultdict(int)
    max_page: dict[str, int] = defaultdict(int)
    for m in retriever.metas:
        chunk_counts[m["file_name"]] += 1
        max_page[m["file_name"]] = max(max_page[m["file_name"]], m["page_start"])
    return [
        {
            "company": meta.company,
            "ticker": meta.ticker,
            "year": meta.year,
            "file_name": fname,
            "pages": max_page.get(fname, 0),
            "num_chunks": chunk_counts.get(fname, 0),
        }
        for fname, meta in DOCUMENTS.items()
    ]


@mcp.tool()
def search_chunks(query: str, top_k: int = 10, document: str | None = None) -> list[dict[str, Any]]:
    """Return raw hybrid-retrieval hits (no LLM synthesis) for transparency/debugging.

    Args:
        query: Search query.
        top_k: Number of hits to return (default 10). Named to match query_documents.
        document: Optional ticker/company/file scope, same as query_documents.

    An unknown `document` surfaces as an MCP tool error (isError), like the other tools.
    """
    retriever = get_engine().retriever
    where = resolve_document_filter(document)  # raise ValueError -> isError on unknown document
    hits = retriever.retrieve(query, top_n=top_k, where=where)
    return [
        {
            "id": h.id,
            "company": h.metadata["company"],
            "year": h.metadata["year"],
            "page": h.metadata["page_start"],
            "section": h.metadata.get("section", ""),
            "chunk_type": h.metadata["chunk_type"],
            "rerank_score": h.rerank_score,
            "vector_score": h.vector_score,
            "bm25_score": h.bm25_score,
            "snippet": " ".join(h.text.split())[:240],
        }
        for h in hits
    ]


@mcp.tool()
def document_stats(document: str, term: str | None = None) -> dict[str, Any]:
    """Exact, computed statistics about one document (not retrieval — no guessing).

    Handles the "structural" questions RAG is bad at: page counts and term frequencies.

    Args:
        document: Which report (ticker like "MCD", company "McDonald's", or file name).
        term: Optional phrase to count; returns total occurrences and the pages it appears on.

    Returns:
        { company, ticker, year, file_name, page_count, word_count,
          term?, term_count?, term_pages? }

    An unknown/empty `document` surfaces as an MCP tool error (isError), like the other tools.
    """
    from .doc_store import DocStore

    where = resolve_document_filter(document)  # raise ValueError -> isError on unknown document
    if where is None:
        raise ValueError("Specify a document (ticker, company, or file name).")
    store = DocStore(get_settings())
    s = store.stats(where["file_name"], term=term)
    out = {
        "company": s.company,
        "ticker": s.ticker,
        "year": s.year,
        "file_name": s.file_name,
        "page_count": s.page_count,
        "word_count": s.word_count,
    }
    if term:
        out.update(term=s.term, term_count=s.term_count, term_pages=s.term_pages)
    return out


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
