# Document provenance

These five PDFs are **public company annual reports** (publicly filed financial documents),
drawn from a labeled document-QA benchmark in which each PDF ships with a `_qa.jsonl` file of
ground-truth question/answer pairs. The QA files are kept here and used by `eval/run_eval.py`.

| Company | File | Fiscal year | Pages |
|---|---|---|---|
| Toyota Motor Corp. | `NYSE_TM_2021.pdf` | 2021 | 54 |
| Costco Wholesale | `NASDAQ_COST_2022.pdf` | 2022 | 76 |
| McDonald's Corp. | `NYSE_MCD_2020.pdf` | 2020 | 98 |
| Accenture plc | `NYSE_ACN_2020.pdf` | 2020 | 106 |
| Philip Morris Intl. | `NYSE_PM_2020.pdf` | 2020 | 141 |

## Why this set

- **Realistic enterprise documents** — exactly the kind of unstructured report data Nexla customers query.
- **Table-heavy** — exercises the table-aware extraction + hybrid (BM25) retrieval path.
- **Span 2020–2022** — exercises fiscal-year disambiguation across documents.
- **Well-known companies** — makes natural **cross-document comparison** questions meaningful.
- **Full question-type coverage** in the gold sets: `text-only`, `multimodal-t` (tables),
  `multimodal-f` (figures), `meta-data`, and `unanswerable` (proves correct abstention).

The `*_qa.jsonl` files are ground truth for evaluation only; the MCP server never reads them.
