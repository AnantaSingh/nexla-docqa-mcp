"""Turn parsed pages into retrieval chunks with rich, attribution-ready metadata.

Design choices that protect accuracy:
  - **Page boundaries are never crossed.** Each chunk belongs to exactly one page, so the
    page number in a citation is always exact.
  - **Table-dense pages stay atomic.** Because financial statements lose meaning when split
    mid-table, a page that reads as tabular becomes a single `table` chunk (the row<->value
    layout is preserved whole). Prose pages are windowed (~600 tokens, ~100 overlap).
  - **Best-effort section headings** (e.g. "Item 1A—Risk Factors", "Note 5—Leases") are
    tracked and attached so citations can name a section, not just a page.
  - Every chunk carries company/ticker/year so cross-document and year-specific questions
    can be filtered and attributed correctly.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import tiktoken

from .pdf_parser import ParsedDocument

_ENC = tiktoken.get_encoding("cl100k_base")

_TABLE_PAGE_NUMERIC_FRAC = 0.35  # page is "tabular" when >=35% of lines look numeric
_HEADING_RE = re.compile(
    r"^(item\s+\d+[a-z]?\b|note\s+\d+\b|part\s+[ivx]+\b)", re.IGNORECASE
)


def _ntokens(text: str) -> int:
    return len(_ENC.encode(text))


@dataclass
class Chunk:
    id: str
    file_name: str
    company: str
    ticker: str
    year: int
    page_start: int
    page_end: int
    section: str
    chunk_type: str  # "text" | "table"
    text: str
    n_tokens: int

    def metadata(self) -> dict:
        """Chroma-safe metadata (scalars only); `text` is stored separately as the document."""
        d = asdict(self)
        d.pop("text")
        return d


def _is_heading(line: str) -> bool:
    s = line.strip()
    if not s or len(s) > 80:
        return False
    if _HEADING_RE.match(s):
        return True
    letters = [c for c in s if c.isalpha()]
    # ALL-CAPS short lines act as section headers in these reports
    return len(letters) >= 3 and all(c.isupper() for c in letters)


def _window_lines(lines: list[str], target: int, overlap: int) -> list[list[str]]:
    """Greedy line packing into ~target-token windows with ~overlap-token carryover."""
    windows: list[list[str]] = []
    cur: list[str] = []
    cur_tok = 0
    for ln in lines:
        lt = _ntokens(ln)
        if cur and cur_tok + lt > target:
            windows.append(cur)
            # carry trailing lines for overlap
            carry, ctok = [], 0
            for prev in reversed(cur):
                pt = _ntokens(prev)
                if ctok + pt > overlap:
                    break
                carry.insert(0, prev)
                ctok += pt
            cur, cur_tok = carry[:], ctok
        cur.append(ln)
        cur_tok += lt
    if cur:
        windows.append(cur)
    return windows


def chunk_document(
    parsed: ParsedDocument, target_tokens: int = 600, overlap_tokens: int = 100
) -> list[Chunk]:
    meta = parsed.meta
    chunks: list[Chunk] = []
    section = ""
    for page in parsed.pages:
        if not page.text.strip():
            continue
        lines = page.text.splitlines()
        # update running section from any heading on this page (before chunking it)
        page_heading = next((ln for ln in lines if _is_heading(ln)), None)
        if page_heading:
            section = page_heading.strip()

        is_table = page.numeric_line_fraction >= _TABLE_PAGE_NUMERIC_FRAC
        if is_table:
            groups = [lines]  # atomic: keep the whole tabular page together
            ctype = "table"
        else:
            groups = _window_lines(lines, target_tokens, overlap_tokens)
            ctype = "text"

        for idx, group in enumerate(groups):
            text = "\n".join(group).strip()
            if not text:
                continue
            chunks.append(
                Chunk(
                    id=f"{meta.ticker or meta.file_name}-p{page.page_number}-{idx}",
                    file_name=meta.file_name,
                    company=meta.company,
                    ticker=meta.ticker,
                    year=meta.year,
                    page_start=page.page_number,
                    page_end=page.page_number,
                    section=section,
                    chunk_type=ctype,
                    text=text,
                    n_tokens=_ntokens(text),
                )
            )
    return chunks
