"""Pure-function tests for fusion, tokenization, and document-filter resolution."""

import pytest

from docqa.retriever import _tokenize, resolve_document_filter, rrf_fuse


def test_tokenizer_keeps_numeric_groups_intact():
    toks = _tokenize("Total revenue 226,954 up 13.5%")
    assert "226,954" in toks  # exact figure preserved for lexical matching
    assert "13.5" in toks
    assert "total" in toks and "revenue" in toks


def test_rrf_rewards_agreement_across_lists():
    vec = ["a", "b", "c"]
    bm25 = ["b", "a", "d"]
    scores = rrf_fuse([vec, bm25], k=60)
    # "a" (ranks 1,2) and "b" (ranks 2,1) appear in both -> outrank single-list "c"/"d"
    assert scores["a"] > scores["c"]
    assert scores["b"] > scores["d"]
    assert set(scores) == {"a", "b", "c", "d"}


def test_rrf_higher_rank_scores_more():
    scores = rrf_fuse([["x", "y"]], k=60)
    assert scores["x"] > scores["y"]


def test_resolve_document_filter_by_ticker_company_and_file():
    assert resolve_document_filter("COST") == {"file_name": "NASDAQ_COST_2022.pdf"}
    assert resolve_document_filter("Toyota")["file_name"] == "NYSE_TM_2021.pdf"
    assert resolve_document_filter("NYSE_MCD_2020.pdf")["file_name"] == "NYSE_MCD_2020.pdf"
    assert resolve_document_filter(None) is None


def test_resolve_document_filter_unknown_raises():
    with pytest.raises(ValueError):
        resolve_document_filter("Tesla")
