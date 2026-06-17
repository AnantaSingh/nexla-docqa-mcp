# Demo Transcript (no API keys required to read)

Recorded outputs from the MCP tools so a reviewer can see the system work **without running it**.
These are real responses captured from the running server; values are deterministic (temperature 0).

> To reproduce live: `npx @modelcontextprotocol/inspector python -m docqa.server`, then call the
> tools below. To verify the numbers automatically: `python -m eval.run_eval`.

---

### `list_documents()` → corpus the agent can query
```json
[
  {"company": "Toyota Motor Corporation",        "ticker": "TM",   "year": 2021, "pages": 54,  "num_chunks": 138},
  {"company": "Costco Wholesale Corporation",    "ticker": "COST", "year": 2022, "pages": 76,  "num_chunks": 117},
  {"company": "McDonald's Corporation",          "ticker": "MCD",  "year": 2020, "pages": 98,  "num_chunks": 151},
  {"company": "Accenture plc",                   "ticker": "ACN",  "year": 2020, "pages": 106, "num_chunks": 157},
  {"company": "Philip Morris International Inc.", "ticker": "PM",   "year": 2020, "pages": 141, "num_chunks": 213}
]
```

---

### 1 · Factual table lookup with source attribution
**Call:** `query_documents(question="What was Costco's total revenue in fiscal 2022?")`
```json
{
  "answer": "Costco's total revenue in fiscal 2022 was $226,954 million.",
  "answer_found": true,
  "citations": [
    {"company": "Costco Wholesale Corporation", "year": 2022, "page": 40,
     "chunk_type": "table", "snippet": "CONSOLIDATED STATEMENTS OF INCOME … 2022 2021 2020 … Total revenue 226,954 195,929 166,761"}
  ]
}
```

### 2 · Multi-document comparison (two reports in one answer)
**Call:** `query_documents(question="Compare the total revenue of Costco and McDonald's in their latest reported fiscal years.")`
```json
{
  "answer": "Costco (FY2022) reported total revenue of $226,954 million, while McDonald's (FY2020) reported total revenues of $19,208 million. Costco's was higher by approximately $207,746 million.",
  "answer_found": true,
  "citations": [
    {"company": "Costco Wholesale Corporation", "year": 2022, "page": 40, "chunk_type": "table"},
    {"company": "McDonald's Corporation",       "year": 2020, "page": 18, "chunk_type": "table"}
  ]
}
```

### 3 · Geographic segment lookup (hard table, recovered by lexical-champion inclusion)
**Call:** `query_documents(question="What was the total revenue for the Company in the United States in 2022?", document="Costco")`
```json
{
  "answer": "The total revenue for Costco in the United States in 2022 was $165,294 million.",
  "answer_found": true,
  "citations": [
    {"company": "Costco Wholesale Corporation", "year": 2022, "page": 66,
     "chunk_type": "table", "snippet": "United States Canada International Total … Total revenue $ 165,294 $ 31,675 $ 29,985 $ 226,954"}
  ]
}
```

### 4 · Exact document statistics (deterministic, not RAG)
**Call:** `document_stats(document="McDonald's", term="franchised margins")`
```json
{
  "company": "McDonald's Corporation", "ticker": "MCD", "year": 2020,
  "page_count": 98, "word_count": 54079,
  "term": "franchised margins", "term_count": 5, "term_pages": [16, 20, 29]
}
```
The same data powers `query_documents` for "how many pages…" / "how many times is X mentioned…",
which plain RAG answers poorly.

### 5 · Vision fallback recovers a figure answer
**Call:** `query_documents(question="What is the difference in the rate of the engine being off during driving between HEVs and PHEVs?", document="Toyota")`
```json
{
  "answer": "According to market data in Japan, the engine is off for ~50% of driving time in HEVs, versus a higher share for PHEVs (read from the page figure).",
  "answer_found": true,
  "citations": [
    {"company": "Toyota Motor Corporation", "year": 2021, "page": 27, "chunk_type": "figure(vision)"}
  ]
}
```
Text extraction missed this chart; the server rendered the page and read it with Claude vision.

### 6 · Unanswerable → correct abstention (no hallucination)
**Call:** `query_documents(question="How many stores does the company open in Shanghai?", document="McDonald's")`
```json
{
  "answer": "The provided sources do not contain information about how many stores McDonald's opens in Shanghai.",
  "answer_found": false,
  "citations": []
}
```
Verified across all 5 `unanswerable` gold questions — 100% correct abstention, even with the vision
fallback enabled.
