# Nexla DocQA — MCP Server for Grounded Q&A over Annual Reports

An [MCP](https://modelcontextprotocol.io) server that lets an AI agent ask natural-language
questions over a set of **financial annual reports** and get **accurate, source-attributed**
answers — or an honest "not found" when the documents don't support an answer.

Indexed corpus (5 public annual reports): **Toyota** (FY2021), **Costco** (FY2022),
**McDonald's** (FY2020), **Accenture** (FY2020), **Philip Morris** (FY2020).

> **Why these documents?** They are realistic enterprise documents (the kind Nexla customers
> actually query), heavy on tables/numbers, span multiple fiscal years (forcing year
> disambiguation), and come from a labeled QA benchmark — every PDF ships with a `_qa.jsonl`
> of ground-truth Q&A, which we use to **measure accuracy** (see [Evaluation](#evaluation)).

---

## Table of contents
- [Quick start](#quick-start)
- [Architecture](#architecture)
- [Retrieval pipeline](#retrieval-pipeline-how-accuracy-is-earned)
- [MCP tools](#mcp-tools)
- [Connecting an MCP client](#connecting-an-mcp-client)
- [Accuracy & robustness](#accuracy--robustness)
- [Evaluation](#evaluation)
- [Example interactions](#example-interactions)
- [Vibe-coding section](#vibe-coding-section)
- [Limitations & future work](#limitations--future-work)

---

## Quick start

Requirements: Python 3.11+, an **OpenAI API key** (embeddings) and an **Anthropic API key**
(answer synthesis).

```bash
# 1. Install (venv + pip)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"                              # convenient (latest compatible deps)
# …or for a byte-for-byte reproducible environment, use the committed lockfile:
# pip install -r requirements.txt && pip install -e . --no-deps

# 2. Configure secrets
cp .env.example .env             # then paste your two keys into .env

# 3. Build the index over the 5 PDFs (one-time; idempotent)
python -m docqa.ingest           # ~1 min: parse → chunk → embed → persist

# 4a. Run the MCP server (stdio)
python -m docqa.server

# 4b. …or explore it interactively with MCP Inspector
npx @modelcontextprotocol/inspector python -m docqa.server

# 5. (optional) Measure accuracy against the gold QA
python -m eval.run_eval          # writes eval/results.md
```

The index lives under `.chroma/` (gitignored). Re-running `ingest` is a no-op unless the PDFs
or chunking settings change; use `--force` to rebuild.

---

## Architecture

```
                         ┌──────────────────────────  INGESTION (one-time)  ──────────────────────────┐
  data/*.pdf  ──►  pdf_parser.py            ──►  chunking.py          ──►  embeddings.py  ──►  vector_store.py
                   layout-aware text             page/table-aware          OpenAI                ChromaDB (persist)
                   (column + row reconstruct)     chunks + metadata        text-embedding-3      + chunks.json (BM25)
                         └───────────────────────────────────────────────────────────────────────────┘

                         ┌──────────────────────────────  QUERY TIME  ───────────────────────────────┐
  MCP client  ──stdio──► server.py  ──►  qa_engine.py  ──►  retriever.py  ──►  llm.py (Claude)
   (Claude /              FastMCP          orchestrate        hybrid + rerank     grounded / abstain
    Inspector)            4 tools          + citations        (see below)         forced tool-use JSON
                         └───────────────────────────────────────────────────────────────────────────┘
```

**Separation of concerns** — each module does one thing and is independently testable:

| Module | Responsibility |
|---|---|
| `pdf_parser.py` | PDF → clean page text. Column detection, y-coordinate row reconstruction, header/footer + dotted-leader cleanup. |
| `chunking.py` | Page-bounded, table-aware chunks with company/year/page/section metadata. |
| `embeddings.py` | Embedding provider behind a `Protocol` (OpenAI default; local fastembed fallback). |
| `vector_store.py` | ChromaDB wrapper (cosine, metadata filtering, persistence). |
| `retriever.py` | Hybrid dense + BM25 retrieval, RRF fusion, cross-encoder rerank, doc filtering. |
| `llm.py` | Claude answer synthesis with a strict grounded/abstain contract (forced tool-use). |
| `qa_engine.py` | Orchestration: retrieve → ground → answer, building **faithful** citations. |
| `server.py` | FastMCP server exposing the four tools over stdio. |
| `ingest.py` | CLI: parse → chunk → embed → persist (idempotent via content hashes). |
| `eval/run_eval.py` | Accuracy harness against the gold `_qa.jsonl` (LLM-as-judge). |

---

## Retrieval pipeline (how accuracy is earned)

```
query
  ├─ dense vector search (OpenAI embeddings, cosine)   → top 20   ← semantics / paraphrase
  └─ BM25 lexical (rank_bm25, numeric-aware tokenizer) → top 20   ← exact figures / names
        └─ Reciprocal Rank Fusion (k=60)               → fused candidate pool
              └─ cross-encoder rerank (ms-marco-MiniLM, ONNX) → top 8   ← precision
                    └─ Claude grounded synthesis (temp 0, cite-or-abstain)
```

- **Hybrid, because financial Q&A needs both.** Embeddings find the right *concept*; BM25 finds
  the exact *number* ("226,954") or proper noun. The numeric-aware tokenizer keeps figures like
  `222,730` intact so they're matchable. Reciprocal Rank Fusion combines the two ranked lists
  without having to reconcile their incompatible score scales.
- **Recall first, then precision.** Each arm casts a focused net (top-20 — measured to beat 30/50 on recall@8); the cross-encoder
  then reranks the fused pool down to the 8 passages the LLM actually sees.
- **Reranking is on by default** (it's the single biggest precision lever); set
  `RERANK_ENABLED=false` to disable.
- **Never drop the lexical champion.** The cross-encoder occasionally buries a chunk that
  *literally contains* the queried entity/figure (e.g. a geographic-segment table holding
  "United States … Total revenue $165,294"). If the top BM25 hit falls out of the top-8, it's
  appended so the evidence still reaches the LLM.
- **Vision fallback (on by default).** If the text-grounded answer abstains, the server renders the
  top retrieved pages to images and retries with Claude vision — recovering answers that live inside
  charts/figures. The strict abstain contract is preserved, so unanswerable questions still abstain.

---

## MCP tools

### `query_documents(question, top_k=8, document=None)`
The core tool. Returns a grounded answer with source citations.

| Field | Type | Notes |
|---|---|---|
| `question` | string (required) | Natural-language question. |
| `top_k` | int (default 8) | Reranked passages to ground on (clamped to ≥1). This is a **floor**: if the top BM25 "lexical champion" was reranked out, it's appended, so up to `top_k + 1` passages may be used. |
| `document` | string (optional) | Scope to one report by ticker (`COST`), company (`Costco`), or file. |

**Returns:** `{ answer, answer_found, citations[], retrieved_count, document_filter }`, where each
citation is `{ label, company, ticker, year, file_name, page, section, chunk_type, snippet }`.
`answer_found=false` means the documents didn't support an answer (no guessing).

**Error contract (all tools):** invalid inputs — an unknown `document`, or a wrong argument type —
surface as standard **MCP tool errors** (`isError: true`) with a descriptive message
(e.g. `"No indexed document matches 'Tesla'. Known: TM, COST, MCD, ACN, PM."`). `answer_found=false`
is **not** an error — it's a valid "the documents don't support an answer" result.

**Example queries:**
- `query_documents(question="What was Costco's total revenue in fiscal 2022?")`
- `query_documents(question="Compare Costco's and McDonald's total revenue.")` *(multi-document)*
- `query_documents(question="What was net revenue in the European Union in 2020?", document="PM")` *(scoped)*
→ `{ "answer": "Costco's total revenue in fiscal 2022 was $226,954 million.", "answer_found": true, "citations": [{ "company": "Costco Wholesale Corporation", "year": 2022, "page": 40, "chunk_type": "table", ... }] }`

### `list_documents()`
Returns the indexed corpus: `[{ company, ticker, year, file_name, pages, num_chunks }]`.
Lets an agent discover what it can ask about. **No inputs.**

**Example query:** `list_documents()`
→ `[{ "company": "Costco Wholesale Corporation", "ticker": "COST", "year": 2022, "pages": 76, "num_chunks": 117 }, … ]`

### `search_chunks(query, top_k=10, document=None)`
Raw hybrid-retrieval hits (no LLM) with `rerank/vector/bm25` scores and snippets — for
transparency and debugging retrieval independently of synthesis. Inputs: `query` (required),
`top_k` (default 10), optional `document` scope.

**Example query:** `search_chunks(query="warehouses operated worldwide", document="COST", top_k=5)`
→ `[{ "id": "COST-p45-0", "page": 45, "chunk_type": "text", "rerank_score": 5.9, "snippet": "…" }, … ]`

### `document_stats(document, term=None)`
Exact, **computed** statistics about one report — `page_count`, `word_count`, and, if `term` is
given, its `term_count` and the `term_pages` it appears on. This is deliberately *not* RAG: it
answers the "how many pages / how many times is X mentioned" questions that retrieval handles
poorly, by counting over the document's full text. `query_documents` also routes those phrasings
here automatically. Inputs: `document` (required), optional `term`.

**Example query:** `document_stats(document="McDonald's", term="franchised margins")`
→ `{ "company": "McDonald's Corporation", "page_count": 98, "term_count": 5, "term_pages": [16, 20, 29] }`

---

## Connecting an MCP client

**MCP Inspector** (quickest):
```bash
npx @modelcontextprotocol/inspector python -m docqa.server
```
This prints a `localhost:6274?...` URL — open it and click **Connect**. Then:

1. Click the **Tools** tab at the top (not *Resources*/*Prompts* — this server exposes **tools only**, so those tabs are intentionally empty).
2. Click **List Tools** → you'll see all four: `query_documents`, `list_documents`, `search_chunks`, `document_stats`.
3. Click **`query_documents`**; a form appears with `question`, `top_k`, `document`.
4. In **`question`** type e.g. `What was Costco's total revenue in fiscal 2022?` (leave `top_k`/`document` blank) and click **Run Tool**.
5. The result panel shows the JSON: `answer`, `answer_found: true`, and `citations` (Costco, p.40, `table`). It takes a few seconds — it's calling OpenAI (embeddings) + Claude (synthesis).

Other quick checks in the same UI:
- `list_documents` → Run (no inputs) → all 5 reports.
- `query_documents` with `question` = "How many stores does the company open in Shanghai?", `document` = `MCD` → `answer_found: false` (abstains, no hallucination).
- `document_stats` with `document` = `MCD`, `term` = `franchised margins` → `page_count: 98`, `term_count: 5`.

**Claude Desktop** — add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "nexla-docqa": {
      "command": "/absolute/path/to/.venv/bin/python",
      "args": ["-m", "docqa.server"],
      "cwd": "/absolute/path/to/nexla-docqa-mcp"
    }
  }
}
```
Restart Claude Desktop; the four tools appear under the server. (Ingest once first.)

---

## Verify it works

Three levels, fastest first:

```bash
# 1. Offline unit tests (no API keys needed) — chunking, fusion, abstention, tool schemas
pytest -q

# 2. One-command end-to-end smoke test (needs index + keys): launches the server over stdio
#    and checks every tool, a grounded answer with citation, exact stats, and correct abstention
python scripts/smoke_test.py
#    expected last line: RESULT: ALL PASSED

# 3. Ask a question straight from the terminal (no MCP client needed)
python scripts/ask.py "What was Costco's total revenue in fiscal 2022?"
python scripts/ask.py "What was net revenue in the EU in 2020?" --document PM

# 4. Interactive — open the server in MCP Inspector and click through the tools
npx @modelcontextprotocol/inspector python -m docqa.server
```

## Accuracy & robustness

Concrete decisions that defend answer quality — most were driven by **empirically inspecting the
actual PDFs** (see the vibe-coding section for the bug this caught):

1. **Anti-hallucination via abstention.** The LLM is instructed to answer *only* from the
   provided sources and to set `answer_found=false` otherwise. Validated against the benchmark's
   `unanswerable` questions.
2. **Faithful citations.** Citations are built from the **metadata of retrieved chunks**, never
   from LLM free-text. The model can only cite labels it was given; an invalid label is dropped,
   so a page number can never be fabricated.
3. **Layout-aware table extraction.** Financial statements are whitespace-aligned, not ruled —
   `find_tables()` both misses them and hallucinates tables from prose. Instead we reconstruct
   visual rows by y-coordinate so `Net sales  $222,730  $192,052  $163,220` stays on one line,
   preserving the row→value association the LLM needs.
4. **Column-aware parsing.** Multi-column reports (e.g. Toyota) are split at the page gutter so
   text reads in the right order — but **numeric-dense (table) pages skip column-splitting**, which
   fixed a bug where a financial statement's label column was being torn from its value columns.
5. **Page-bounded chunks.** A chunk never crosses a page boundary, so the cited page is exact.
   Table-dense pages are kept **atomic** (never split mid-table).
6. **Year & company disambiguation.** Every chunk and citation carries company + fiscal year; the
   prompt is told to respect the year/company asked (reports span 2020–2022).
7. **Determinism (qualified).** Temperature 0, a fixed embedding model, and a persisted index make
   **retrieval deterministic**; answer *text* is near-deterministic but not guaranteed identical
   (LLMs retain minor nondeterminism even at temp 0 — the eval shows small run-to-run wobble, see
   [Evaluation](#evaluation)).
8. **Graceful failure.** Empty question, no hits, unknown `document` filter, or a missing index
   all return clear messages instead of crashing.

---

## Evaluation

We evaluate against the **55 ground-truth Q&A pairs** that ship with the 5 PDFs (`data/*_qa.jsonl`),
using a Claude **LLM-as-judge** (free-form answers can't be graded by exact match). Two views:

- **Scoped** (question routed to its source report): answer correctness for answerable questions;
  correct abstention for `unanswerable` ones.
- **Unscoped** (search all 5 reports): document-routing accuracy — did the correct report get cited?

Reproduce with `python -m eval.run_eval` → full table in [`eval/results.md`](eval/results.md).

**Results** (55 gold questions; judge = Claude, temperature 0):

| Question type | N | Accuracy (correct) | Incl. partial |
|---|---|---|---|
| text-only | 10 | 70% | 80% |
| multimodal-t (tables) | 21 | 76% | 81% |
| multimodal-f (figures) | 12 | 75% | 83% |
| meta-data | 7 | 57% | 86% |
| **answerable total** | **50** | **72%** | **82%** |

| Robustness metric | Result |
|---|---|
| **Correct abstention on `unanswerable`** | **5/5 (100%)** — no hallucinations (held even with the vision fallback on) |
| **Multi-document routing** (correct report cited among all 5) | **44/50 (88%)** |

**Reading the numbers honestly:**
- **Abstention is perfect (5/5)** — the property that matters most — and it stays perfect *with the
  vision fallback enabled*, i.e. the system reads page images when needed but still refuses to
  invent answers that aren't there.
- **Run-to-run variance is real.** The LLM judge (and vision fallback) aren't perfectly
  deterministic, so per-category numbers wobble by ~±1 question between runs (e.g. metadata has
  swung 57%↔71%); the **answerable strict total is stable at 72%**. Treat category splits as
  indicative, not exact.
- **Metadata went from 29%→~60-70%** after adding a deterministic `document_stats` path (page
  counts, term frequencies) — computations over the whole document, answered by counting rather
  than RAG guessing.
- **The numbers are a slight *under*-estimate** because the LLM judge is occasionally over-strict on
  units. Example: for Costco's U.S. revenue the system answered **"$165,294 million"** (correct — the
  statement is "in millions"), but the gold omits the unit ("$165,294") and the judge marked it
  INCORRECT over a non-existent 1000× discrepancy. A few "INCORRECT" verdicts are this judge artifact.

### Retrieval k is measured, not guessed

`eval/sweep_k.py` runs a **free, no-LLM recall@k sweep** (does the answer figure land in the top-k?)
to choose the retrieval sizes. Finding: a **tighter recall stage (vector/BM25 top-20) beats top-30/50**
— recall@8 of **92% vs 88%** — because a cleaner candidate pool lets the cross-encoder rank the answer
chunk higher, *and* it's cheaper. Recall plateaus at 96% by top-12; top-8 is the cost/recall knee.
End-to-end accuracy was flat across k=20 vs k=30 (the generator is robust to the small recall
difference), so k=20 was adopted as the equal-accuracy, lower-cost choice.

---

## Example interactions

Verbatim `query_documents` outputs (full log in [`docs/INTERACTION_LOG.md`](docs/INTERACTION_LOG.md)):

**1 · Factual table lookup**
> **Q:** "What was Costco's total revenue in fiscal 2022?"
> **A:** Costco's total revenue in fiscal 2022 was **$226,954 million**.
> **Source:** Costco (FY2022), p.40, `table`

**2 · Multi-document comparison** (context from two reports)
> **Q:** "Compare the total revenue of Costco and McDonald's in their latest reported fiscal years."
> **A:** Costco (FY2022) **$226,954M** vs McDonald's (FY2020) **$19,208M** — Costco higher by ~$207,746M.
> **Sources:** Costco (FY2022) p.40 · McDonald's (FY2020) p.18

**3 · Unanswerable → correct abstention**
> **Q:** "How many stores does the company open in Shanghai?" (scoped to McDonald's)
> **A:** *answer_found = false* — "The provided sources do not contain information about how many stores McDonald's opens in Shanghai." (no citations, no guess)

See [`docs/INTERACTION_LOG.md`](docs/INTERACTION_LOG.md) for the full log.

---

## Vibe-coding section

**Tool used:** [Claude Code](https://claude.com/claude-code) (Claude Opus), driven as an agentic
pair-programmer — it ran the shell, inspected the PDFs, wrote the modules, and executed the tests
and eval, while I set direction, challenged claims, and made the trade-off calls.

### How I directed it — and what worked

I ran the whole build **plan-first**. I asked it to enter a planning mode, *"deep dive into edge
cases,"* and then explicitly switch into a **reviewer mode to critique its own plan** before writing
any code. That second pass was where the plan got good: the AI's first draft parked the
cross-encoder reranker as "optional," and the self-review (plus my pushback) promoted it to a core
precision stage with concrete fan-in/fan-out numbers (30 + 30 → fuse → rerank → 8).

What worked best was forcing **evidence over assertion**:

- I acted as a **strict reviewer** on retrieval specifically — *"do tables lose row/column
  association? how many chunks for recall? what happens when the right passage isn't retrieved?"*
  Each question turned a hand-wave in the plan into a concrete mechanism (numeric-aware BM25
  tokenizer, RRF, atomic table chunks, abstention as the safety net).
- I challenged a recommendation directly: *"does moving embeddings backend help in any way?"* The
  AI correctly conceded that fastembed-vs-sentence-transformers is **zero** accuracy difference
  (pure install ergonomics) and that the real lever is the model tier — which led us to OpenAI
  `text-embedding-3-small`. Good example of the AI being right to **walk back** an oversold point.

### Where the AI genuinely shone

**Empirical, iterative debugging against the real data.** Before writing the parser, it probed the
actual PDFs and discovered that PyMuPDF's `find_tables()` both *misses* the whitespace-aligned
financial statements and *hallucinates* tables out of prose. It then found that plain text
extraction splits each number onto its own line, destroying the row→value link. It solved this with
y-coordinate row reconstruction — and then **caught its own bug**: a column-gutter heuristic was
tearing the label column of Costco's income statement away from its value columns, which is why
"total revenue" retrieval was failing. It traced the failing query, saw the income-statement chunk
ranked #11 with a broken `Total revenue ....` line, added a numeric-density gate, re-ingested, and
confirmed the chunk jumped to **rank #1**. That loop — hypothesize → run → inspect → fix → verify —
is the thing AI tooling accelerates most.

### Where I overrode or corrected it

- **Document selection.** The dataset had 229 PDFs; I steered toward the 5 financial reports
  because they make *cross-document comparison* questions meaningful, over the AI's equally-valid
  academic-papers option.
- **Honesty about guarantees.** I pushed it to stop implying retrieval "finds the best chunk."
  The honest framing — *no system guarantees that; maximize recall, rerank for precision, and
  **abstain** rather than hallucinate* — made it into both the prompt contract and the docs.
- **Scope discipline.** I kept vision-for-figures and query-expansion as documented future work
  rather than letting scope creep past the time box.

### Overall view on AI in a software workflow

AI tooling shifts the engineer's job from *typing code* to **specifying intent and verifying
behavior**. It's a force multiplier exactly where there's a tight feedback loop — real data to probe,
tests to run, an eval harness to score against — and it's most dangerous when allowed to *assert*
correctness without producing evidence. The highest-leverage things I did in this session weren't
writing code; they were (1) insisting on a plan-then-review structure, (2) acting as a strict
reviewer that demanded mechanisms and numbers, and (3) building a measurable eval so "accuracy"
was a number, not a vibe. Used that way, the AI did in a few hours what would otherwise take a day —
but the engineering judgment about *what* to build and *how to know it works* still had to come from
a human in the loop.

---

## Limitations & future work

- **Figure-only answers (`multimodal-f`).** Content inside an image isn't captured by text
  extraction. The **vision fallback** addresses many of these (render page → Claude vision on
  abstention), but it only triggers when the text path *abstains* — a figure question that retrieves
  plausible-but-wrong text won't trigger it. Always-on vision per query would help further at higher
  cost/latency.
- **LLM-judge unit strictness.** The eval's judge occasionally penalizes correct answers that add a
  unit the gold omitted (see the Costco U.S.-revenue example above), so reported accuracy is a
  slight under-estimate.
- **Scale.** ~800 chunks across 5 docs sit comfortably in memory and a local Chroma store. More
  documents would motivate batched ingestion and a server-backed vector DB — both are drop-in given
  the provider/store abstractions.
- **Single embedding provider online.** Ingestion calls OpenAI; for fully-offline use, set
  `EMBED_PROVIDER=fastembed` (local bge-small, no API key).

---

## Project layout

```
src/docqa/      parser, chunking, embeddings, vector store, retriever, llm, qa engine, server, ingest
eval/           run_eval.py + generated results.md/json
tests/          chunking, retriever fusion, qa-engine abstention, server schema (offline)
data/           the 5 PDFs + their gold _qa.jsonl + PROVENANCE.md
```

Run the tests with `pytest` (they're offline — no API keys needed).
