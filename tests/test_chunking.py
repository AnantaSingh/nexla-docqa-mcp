"""Chunking invariants that protect citation accuracy."""

from docqa.chunking import _is_heading, _window_lines, chunk_document
from docqa.documents import DocMeta
from docqa.pdf_parser import Page, ParsedDocument

META = DocMeta("ACME_2021.pdf", "Acme Corp", "ACME", 2021)


def _doc(pages):
    return ParsedDocument(meta=META, pages=pages)


def test_chunks_never_cross_page_boundaries():
    pages = [
        Page(1, "alpha beta\n" * 200, numeric_line_fraction=0.0),
        Page(2, "gamma delta\n" * 200, numeric_line_fraction=0.0),
    ]
    chunks = chunk_document(_doc(pages), target_tokens=120, overlap_tokens=20)
    assert chunks, "expected chunks"
    for c in chunks:
        assert c.page_start == c.page_end  # one page per chunk -> exact attribution


def test_table_dense_page_is_single_atomic_chunk():
    table_text = "\n".join(f"Row {i} 1,234 5,678 9,012" for i in range(40))
    pages = [Page(5, table_text, numeric_line_fraction=0.9)]
    chunks = chunk_document(_doc(pages), target_tokens=50, overlap_tokens=10)
    assert len(chunks) == 1  # not split even though it exceeds the token target
    assert chunks[0].chunk_type == "table"
    assert chunks[0].page_start == 5


def test_metadata_is_populated():
    pages = [Page(3, "Some narrative text about the business.", numeric_line_fraction=0.0)]
    c = chunk_document(_doc(pages))[0]
    assert c.company == "Acme Corp" and c.ticker == "ACME" and c.year == 2021
    assert c.file_name == "ACME_2021.pdf" and c.page_start == 3
    assert c.n_tokens > 0


def test_window_overlap_carries_context():
    lines = [f"line number {i}" for i in range(60)]
    windows = _window_lines(lines, target=40, overlap=15)
    assert len(windows) >= 2
    # consecutive windows should share at least one line (overlap)
    assert set(windows[0]) & set(windows[1])


def test_heading_detection():
    assert _is_heading("Item 1A—Risk Factors")
    assert _is_heading("NOTES TO CONSOLIDATED FINANCIAL STATEMENTS")
    assert not _is_heading("This is an ordinary sentence with normal casing.")
