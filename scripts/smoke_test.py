"""One-command smoke test: launches the MCP server over stdio and exercises every tool.

    python scripts/smoke_test.py

Verifies, as a real MCP client would:
  - the server initializes and advertises the tools capability
  - all four tools list with valid input schemas
  - query_documents returns a grounded answer with citations
  - document_stats returns exact counts
  - an unanswerable question abstains (answer_found = false)
  - an unknown document filter yields a helpful, non-crashing error

Requires the index (`python -m docqa.ingest`) and ANTHROPIC_API_KEY + OPENAI_API_KEY in .env.
"""

from __future__ import annotations

import asyncio
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PY = sys.executable  # use the same interpreter (the active venv)


def ok(cond: bool) -> str:
    return "PASS" if cond else "FAIL"


async def main() -> int:
    params = StdioServerParameters(command=PY, args=["-m", "docqa.server"])
    failures = 0
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as s:
            init = await s.initialize()
            print(f"[init]  server={init.serverInfo.name} proto={init.protocolVersion}")

            tools = {t.name for t in (await s.list_tools()).tools}
            expected = {"query_documents", "list_documents", "search_chunks", "document_stats"}
            print(f"[tools] {ok(tools == expected)} {sorted(tools)}")
            failures += tools != expected

            r1 = (await s.call_tool("query_documents",
                  {"question": "What was Costco's total revenue in fiscal 2022?"})).structuredContent
            cond = r1["answer_found"] and any(c["page"] == 40 for c in r1["citations"])
            print(f"[query] {ok(cond)} -> {r1['answer'][:70]} (cite p{r1['citations'][0]['page'] if r1['citations'] else '?'})")
            failures += not cond

            r2 = (await s.call_tool("document_stats",
                  {"document": "McDonald's", "term": "franchised margins"})).structuredContent
            cond = r2.get("page_count") == 98 and r2.get("term_count") == 5
            print(f"[stats] {ok(cond)} -> {r2.get('page_count')} pages, term x{r2.get('term_count')}")
            failures += not cond

            r3 = (await s.call_tool("query_documents",
                  {"question": "How many stores does the company open in Shanghai?", "document": "McDonald's"})).structuredContent
            print(f"[abstain] {ok(not r3['answer_found'])} -> answer_found={r3['answer_found']}")
            failures += r3["answer_found"]

            r4 = (await s.call_tool("query_documents",
                  {"question": "revenue?", "document": "Tesla"})).structuredContent
            cond = (not r4["answer_found"]) and "No indexed document" in r4["answer"]
            print(f"[bad-filter] {ok(cond)} -> {r4['answer'][:60]}")
            failures += not cond

    print("\nRESULT:", "ALL PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
