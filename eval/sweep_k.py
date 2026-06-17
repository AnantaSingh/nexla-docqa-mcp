"""Tune the retrieval k's against the gold data via recall@k — no LLM calls.

Why this design: the k values mainly affect *retrieval*, not generation. So instead of running
the full (slow, rate-limited, paid) LLM eval for every k, we measure a cheap proxy:

    recall@k = fraction of gold questions whose answer figure appears in the top-k retrieved chunks

The answer "needle" is the distinctive number in each gold answer (e.g. 226954, 165294). This
covers the 26 answerable questions that have such a figure. It's a proxy (a number can co-occur
without truly answering), but it's an honest, reproducible signal for choosing k — and it's free.

Efficiency: for each recall-stage setting we retrieve a long list ONCE per question, then slice it
to score every final-k. Run:  python -m eval.sweep_k
"""

from __future__ import annotations

import glob
import json
import re

from docqa.config import get_settings
from docqa.documents import doc_meta_for
from docqa.retriever import Retriever

ANSWERABLE = {"text-only", "multimodal-t", "multimodal-f"}
RECALL_STAGE_KS = [20, 30, 50]  # vector_top_k = bm25_top_k
FINAL_KS = [3, 5, 8, 12, 16, 20]


def needles(answer: str) -> set[str]:
    out: set[str] = set()
    for m in re.findall(r"\$?\d[\d,]*\.?\d*%?", answer):
        norm = m.replace("$", "").replace(",", "").rstrip("%").rstrip(".")
        digits = norm.replace(".", "")
        if len(digits) < 3:
            continue
        if len(digits) == 4 and digits.isdigit() and 2000 <= int(digits) <= 2025:
            continue  # drop bare years (appear on every page)
        out.add(norm)
    return out


def load_needle_questions():
    items = []
    for f in sorted(glob.glob("data/*_qa.jsonl")):
        src = f.split("/")[-1].replace("_qa.jsonl", ".pdf")
        company = doc_meta_for(src).company
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["type"] not in ANSWERABLE:
                continue
            ns = needles(d["answer"])
            if ns:
                items.append((d["question"], company, src, ns))
    return items


def chunk_has_needle(chunk_text: str, ns: set[str]) -> bool:
    norm = chunk_text.replace(",", "")
    for n in ns:
        # digit-boundary match so "160" doesn't match inside "2160"
        if re.search(r"(?<!\d)" + re.escape(n) + r"(?!\d)", norm):
            return True
    return False


def main() -> int:
    settings = get_settings()
    items = load_needle_questions()
    print(f"Recall@k sweep over {len(items)} answerable questions with a numeric needle.\n")
    maxk = max(FINAL_KS)

    print(f"{'recall_k':>9} | " + " ".join(f"@{k:<4}" for k in FINAL_KS))
    print("-" * (12 + 6 * len(FINAL_KS)))
    for rk in RECALL_STAGE_KS:
        settings.vector_top_k = rk
        settings.bm25_top_k = rk
        retriever = Retriever(settings)
        hits = {k: 0 for k in FINAL_KS}
        for question, _company, src, ns in items:
            ranked = retriever.retrieve(question, top_n=maxk, where={"file_name": src})
            texts = [c.text for c in ranked]
            for k in FINAL_KS:
                if any(chunk_has_needle(t, ns) for t in texts[:k]):
                    hits[k] += 1
        row = " ".join(f"{hits[k] / len(items):>4.0%}" for k in FINAL_KS)
        print(f"{rk:>9} | {row}")

    print("\nRead the knee: the smallest (recall_k, final_k) where recall stops climbing is the")
    print("efficient choice — bigger k past that just adds context/cost without recall.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
