"""PDF -> clean, page-level text with layout-aware reconstruction.

Why not `page.find_tables()` / `get_text()` directly?
  - On these annual reports `find_tables()` both *misses* the real (whitespace-aligned)
    financial statements and *hallucinates* tables out of prose, so it is unreliable here.
  - Plain `get_text()` puts each numeric value of a financial row on its own line, destroying
    the row -> value association an LLM needs to answer "what was net sales in 2021?".

Approach (validated against Costco's income statement and Toyota's multi-column report):
  1. Read word boxes (`get_text("words")`), which carry x/y coordinates.
  2. Detect 1- vs 2-column layout via a low-traffic vertical gutter.
  3. Within each column, cluster words into visual lines by y-coordinate and order by x.
     This keeps "Net sales  $222,730  $192,052  $163,220" together on one line.
  4. Clean: drop dotted leaders, fix hyphenation, strip running headers/footers.

The result is plain text that preserves row structure, which both the embedder and the LLM
read well, without depending on a brittle table detector.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from .documents import DocMeta, doc_meta_for

# ---- tunables (layout heuristics; conservative, validated on the 5 reports) ----
_Y_BUCKET = 3.0  # points; words within this vertical band are one visual line
_GUTTER_BAND = (0.40, 0.60)  # search the central 20% of page width for a column gutter
_GUTTER_MAX_CROSS_FRAC = 0.02  # gutter must be crossed by <2% of words to count as 2-col
_MIN_WORDS_FOR_COLUMNS = 80  # sparse pages are never treated as multi-column
_TABLE_WORD_NUMERIC_RATIO = 0.04  # numeric-dense pages are tables -> reconstruct full-width
_HEADER_FOOTER_LINES = 2  # inspect top/bottom N lines of each page
_HEADER_FOOTER_DOC_FRAC = 0.5  # a line repeated on >=50% of pages is a running header/footer

_NUM_TOKEN = re.compile(r"[(\-]?\$?\d[\d,.]*\)?%?")
_NUM_WORD = re.compile(r"^[(\-]?\$?\d[\d,.]*\)?%?$")


@dataclass
class Page:
    page_number: int  # 1-based, matches the PDF viewer
    text: str
    numeric_line_fraction: float = 0.0  # share of lines that look tabular (>=2 numbers)


@dataclass
class ParsedDocument:
    meta: DocMeta
    pages: list[Page] = field(default_factory=list)

    @property
    def page_count(self) -> int:
        return len(self.pages)


def _clean_line(s: str) -> str:
    s = s.replace("­", "")  # soft hyphen
    # dotted leaders: consecutive dots ("....") or spaced dots (". . . .") -> single space.
    # The two-dot minimum protects decimals like "13.17" (a single dot) from being touched.
    s = re.sub(r"(?:\.\s*){2,}", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _reconstruct_lines(words: list[tuple]) -> list[str]:
    """Cluster words (x0,y0,x1,y1,text,...) into visual lines by y, ordered by x."""
    rows: dict[int, list[tuple[float, str]]] = defaultdict(list)
    for w in words:
        rows[round(w[1] / _Y_BUCKET)].append((w[0], w[4]))
    lines = []
    for key in sorted(rows):
        line = _clean_line(" ".join(word for _, word in sorted(rows[key])))
        if line:
            lines.append(line)
    return lines


def _detect_gutter_x(words: list[tuple], width: float) -> float | None:
    """Return the x of a clean two-column gutter, or None for single-column pages."""
    if len(words) < _MIN_WORDS_FOR_COLUMNS:
        return None
    spans = [(w[0], w[2]) for w in words]
    best_x, best_cross = None, len(words)
    for frac in [_GUTTER_BAND[0] + i * (_GUTTER_BAND[1] - _GUTTER_BAND[0]) / 20 for i in range(21)]:
        x = frac * width
        cross = sum(1 for x0, x1 in spans if x0 < x < x1)
        if cross < best_cross:
            best_x, best_cross = x, cross
    if best_cross < _GUTTER_MAX_CROSS_FRAC * len(words):
        return best_x
    return None


def _word_numeric_ratio(words: list[tuple]) -> float:
    if not words:
        return 0.0
    return sum(1 for w in words if _NUM_WORD.match(w[4])) / len(words)


def _page_lines(page: "fitz.Page") -> list[str]:
    words = page.get_text("words")
    if not words:
        return []
    width = page.rect.width
    # Numeric-dense pages are financial tables: a wide label->value gap looks like a column
    # gutter but isn't. Reconstruct them full-width so each row label keeps its values.
    if _word_numeric_ratio(words) >= _TABLE_WORD_NUMERIC_RATIO:
        return _reconstruct_lines(words)
    gutter = _detect_gutter_x(words, width)
    if gutter is None:
        return _reconstruct_lines(words)
    left = [w for w in words if (w[0] + w[2]) / 2 < gutter]
    right = [w for w in words if (w[0] + w[2]) / 2 >= gutter]
    return _reconstruct_lines(left) + _reconstruct_lines(right)


def _strip_running_headers_footers(pages_lines: list[list[str]]) -> list[list[str]]:
    """Remove lines that recur at the top/bottom of most pages (nav bars, page numbers)."""
    n = len(pages_lines)
    if n < 4:
        return pages_lines
    counts: Counter[str] = Counter()
    for lines in pages_lines:
        edge = lines[:_HEADER_FOOTER_LINES] + lines[-_HEADER_FOOTER_LINES:]
        for ln in set(edge):
            counts[_norm_for_repeat(ln)] += 1
    threshold = max(2, int(_HEADER_FOOTER_DOC_FRAC * n))
    repeated = {k for k, c in counts.items() if c >= threshold and not _looks_substantive(k)}
    if not repeated:
        return pages_lines
    cleaned = []
    for lines in pages_lines:
        keep = [
            ln
            for i, ln in enumerate(lines)
            if not (
                (i < _HEADER_FOOTER_LINES or i >= len(lines) - _HEADER_FOOTER_LINES)
                and _norm_for_repeat(ln) in repeated
            )
        ]
        cleaned.append(keep)
    return cleaned


def _norm_for_repeat(s: str) -> str:
    # page-number-insensitive key so "Page 12" / "Page 13" collapse together
    return re.sub(r"\d+", "#", s).lower().strip()


def _looks_substantive(norm_line: str) -> bool:
    # don't strip long lines even if they repeat; only short nav/footer chrome
    return len(norm_line) > 80


def _join_hyphenation(lines: list[str]) -> str:
    """Join words split across visual lines by an end-of-line hyphen."""
    out: list[str] = []
    for ln in lines:
        if out and re.search(r"[A-Za-z]-$", out[-1]) and re.match(r"[a-z]", ln):
            out[-1] = out[-1][:-1] + ln
        else:
            out.append(ln)
    return "\n".join(out)


def _numeric_line_fraction(lines: list[str]) -> float:
    if not lines:
        return 0.0
    tabular = sum(1 for ln in lines if len(_NUM_TOKEN.findall(ln)) >= 2)
    return tabular / len(lines)


def parse_pdf(pdf_path: str | Path) -> ParsedDocument:
    """Parse a single PDF into cleaned, page-level text with layout reconstruction."""
    pdf_path = Path(pdf_path)
    meta = doc_meta_for(pdf_path.name)
    doc = fitz.open(pdf_path)
    try:
        raw_pages = [_page_lines(doc[i]) for i in range(doc.page_count)]
    finally:
        doc.close()
    raw_pages = _strip_running_headers_footers(raw_pages)
    pages = [
        Page(
            page_number=i + 1,
            text=_join_hyphenation(lines),
            numeric_line_fraction=_numeric_line_fraction(lines),
        )
        for i, lines in enumerate(raw_pages)
    ]
    return ParsedDocument(meta=meta, pages=pages)
