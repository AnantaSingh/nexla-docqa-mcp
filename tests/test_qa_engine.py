"""QA-engine behaviour: grounding, abstention, and faithful citation mapping.

Uses fakes for the retriever and LLM so these run offline with no API keys.
"""

from docqa.config import get_settings
from docqa.llm import GroundedAnswer
from docqa.qa_engine import QAEngine
from docqa.retriever import RetrievedChunk


def _chunk(cid, company, ticker, year, page, text):
    return RetrievedChunk(
        id=cid,
        text=text,
        metadata={
            "company": company,
            "ticker": ticker,
            "year": year,
            "file_name": f"{ticker}.pdf",
            "page_start": page,
            "section": "",
            "chunk_type": "text",
        },
    )


class FakeRetriever:
    def __init__(self, chunks):
        self._chunks = chunks

    def retrieve(self, query, top_n=None, where=None):
        return self._chunks


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.seen_sources = None

    def generate(self, question, sources_block):
        self.seen_sources = sources_block
        return self.result


def _engine(chunks, result):
    return QAEngine(
        settings=get_settings(),
        retriever=FakeRetriever(chunks),
        llm=FakeLLM(result),
    )


def test_abstains_when_llm_reports_not_found():
    chunks = [_chunk("c1", "Costco", "COST", 2022, 40, "irrelevant text")]
    res = _engine(chunks, GroundedAnswer("Not in the documents.", False, [])).answer("q?")
    assert res.answer_found is False
    assert res.citations == []  # no citations when abstaining -> no fake provenance


def test_citation_maps_to_real_chunk_metadata():
    chunks = [
        _chunk("c1", "Costco", "COST", 2022, 40, "Total revenue 226,954"),
        _chunk("c2", "McDonald's", "MCD", 2020, 18, "Total revenues 19,208"),
    ]
    res = _engine(chunks, GroundedAnswer("Costco 226,954M.", True, ["S1"])).answer("revenue?")
    assert res.answer_found is True
    assert len(res.citations) == 1
    cit = res.citations[0]
    assert cit.ticker == "COST" and cit.page == 40 and cit.year == 2022


def test_invalid_cited_label_is_dropped_then_falls_back_to_top():
    chunks = [_chunk("c1", "Costco", "COST", 2022, 40, "Total revenue 226,954")]
    # LLM cites a label we never provided -> must not fabricate; falls back to top source
    res = _engine(chunks, GroundedAnswer("226,954M.", True, ["S99"])).answer("revenue?")
    assert res.answer_found is True
    assert len(res.citations) == 1 and res.citations[0].ticker == "COST"


def test_empty_question_is_rejected_without_calling_models():
    res = _engine([], GroundedAnswer("x", True, [])).answer("   ")
    assert res.answer_found is False
    assert "provide a question" in res.answer.lower()


def test_no_retrieval_results_returns_not_found():
    res = _engine([], GroundedAnswer("x", True, [])).answer("anything?")
    assert res.answer_found is False
    assert res.retrieved_count == 0
