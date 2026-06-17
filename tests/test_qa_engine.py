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


def test_invalid_cited_label_falls_back_to_top_only_for_single_document():
    chunks = [_chunk("c1", "Costco", "COST", 2022, 40, "Total revenue 226,954")]
    # LLM cites a label we never provided -> must not fabricate; single doc in scope, so the
    # top source is an unambiguous, faithful fallback.
    res = _engine(chunks, GroundedAnswer("226,954M.", True, ["S99"])).answer("revenue?")
    assert res.answer_found is True
    assert len(res.citations) == 1 and res.citations[0].ticker == "COST"


def test_no_fallback_citation_when_multiple_documents_retrieved():
    # answer_found=True but no valid used_sources, and TWO docs in scope -> citing the top source
    # could point at the wrong document, so we must return NO citation rather than fabricate one.
    chunks = [
        _chunk("c1", "Costco", "COST", 2022, 40, "Total revenue 226,954"),
        _chunk("c2", "McDonald's", "MCD", 2020, 18, "Total revenues 19,208"),
    ]
    res = _engine(chunks, GroundedAnswer("Some grounded answer.", True, [])).answer("revenue?")
    assert res.answer_found is True
    assert res.citations == []


def test_empty_question_is_rejected_without_calling_models():
    res = _engine([], GroundedAnswer("x", True, [])).answer("   ")
    assert res.answer_found is False
    assert "provide a question" in res.answer.lower()


def test_no_retrieval_results_returns_not_found():
    res = _engine([], GroundedAnswer("x", True, [])).answer("anything?")
    assert res.answer_found is False
    assert res.retrieved_count == 0


def _real_chunk(cid, page):
    """A chunk pointing at a real data PDF so the vision fallback can render the page."""
    return RetrievedChunk(
        id=cid, text="x",
        metadata={"company": "Costco Wholesale Corporation", "ticker": "COST", "year": 2022,
                  "file_name": "NASDAQ_COST_2022.pdf", "page_start": page,
                  "section": "", "chunk_type": "table"},
    )


class _FakeVisionLLM:
    """Text path abstains; vision path returns a found answer that used the SECOND page (P2)."""
    def __init__(self, vision_result):
        self.vision_result = vision_result

    def generate(self, question, sources_block):
        return GroundedAnswer("Not in the text.", False, [])

    def generate_from_images(self, question, images):
        return self.vision_result


def test_vision_fallback_cites_the_page_the_model_used_not_hardcoded_p1():
    import pytest
    s = get_settings()
    if not (s.data_path / "NASDAQ_COST_2022.pdf").exists() or not s.vision_fallback_enabled:
        pytest.skip("needs the data PDF + vision fallback enabled")
    chunks = [_real_chunk("c1", 40), _real_chunk("c2", 41)]  # P1->p40, P2->p41
    eng = QAEngine(settings=s, retriever=FakeRetriever(chunks),
                   llm=_FakeVisionLLM(GroundedAnswer("From the chart on the page.", True, ["P2"])))
    res = eng.answer("a figure question?", document="COST")
    assert res.answer_found is True
    assert res.citations, "vision answer must carry a citation"
    assert res.citations[0].page == 41, "must cite the page the model used (P2=41), not hardcoded P1=40"
    assert res.citations[0].chunk_type == "figure(vision)"
    assert res.document_filter == "COST"  # scope restored, not None
