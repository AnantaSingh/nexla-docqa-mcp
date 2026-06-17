"""Registry of the indexed corpus: maps PDF filename -> human-friendly metadata.

Keeping this explicit (rather than parsing it out of filenames at runtime) means
every chunk and every citation carries a clean company name + fiscal year, which is
what users actually want to see in an attribution.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DocMeta:
    file_name: str
    company: str
    ticker: str
    year: int


DOCUMENTS: dict[str, DocMeta] = {
    d.file_name: d
    for d in [
        DocMeta("NYSE_TM_2021.pdf", "Toyota Motor Corporation", "TM", 2021),
        DocMeta("NASDAQ_COST_2022.pdf", "Costco Wholesale Corporation", "COST", 2022),
        DocMeta("NYSE_MCD_2020.pdf", "McDonald's Corporation", "MCD", 2020),
        DocMeta("NYSE_ACN_2020.pdf", "Accenture plc", "ACN", 2020),
        DocMeta("NYSE_PM_2020.pdf", "Philip Morris International Inc.", "PM", 2020),
    ]
}


def doc_meta_for(file_name: str) -> DocMeta:
    """Resolve metadata for a PDF; fall back to a sensible default for unknown files."""
    if file_name in DOCUMENTS:
        return DOCUMENTS[file_name]
    stem = file_name.rsplit(".", 1)[0]
    return DocMeta(file_name=file_name, company=stem, ticker="", year=0)
