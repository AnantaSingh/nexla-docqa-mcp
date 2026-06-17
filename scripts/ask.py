"""Ask the DocQA engine a question from the terminal (no MCP client needed).

    python scripts/ask.py "What was Costco's total revenue in fiscal 2022?"
    python scripts/ask.py "What was net revenue in the EU in 2020?" --document PM

A thin CLI over the same QAEngine the MCP server uses — handy for quick manual checks.
"""

from __future__ import annotations

import argparse

from docqa.qa_engine import QAEngine


def main() -> int:
    ap = argparse.ArgumentParser(description="Ask the document Q&A engine a question.")
    ap.add_argument("question", help="The natural-language question.")
    ap.add_argument("--document", "-d", default=None, help="Scope to one report (e.g. COST).")
    ap.add_argument("--top_k", type=int, default=8)
    args = ap.parse_args()

    res = QAEngine().answer(args.question, top_k=args.top_k, document=args.document)
    print(f"\nQ: {res.question}")
    print(f"answer_found: {res.answer_found}")
    print(f"\nA: {res.answer}\n")
    if res.citations:
        print("Citations:")
        for c in res.citations:
            print(f"  - [{c.label}] {c.company} (FY{c.year}) p.{c.page} [{c.chunk_type}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
