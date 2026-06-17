"""Accuracy harness over the gold `_qa.jsonl` shipped with each PDF.

It runs every gold question through the real QA engine and scores it two ways:

  1. SCOPED (document = the question's source report):
       - answerable questions  -> Claude LLM-as-judge: CORRECT / PARTIAL / INCORRECT
       - unanswerable questions -> did the system correctly abstain (answer_found == False)?
  2. UNSCOPED (search all 5 reports):
       - document routing: for answerable questions, did a citation land on the right report?
       - abstention held under the harder all-corpus setting?

Run:  python -m eval.run_eval   (writes eval/results.md + eval/results.json)

The LLM-as-judge is used because free-form answers can't be graded by exact match; numbers,
units, and phrasing vary. Temperature 0 keeps grading stable.
"""

from __future__ import annotations

import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import anthropic

from docqa.config import get_settings
from docqa.documents import doc_meta_for
from docqa.qa_engine import QAEngine

ANSWERABLE = {"text-only", "multimodal-t", "multimodal-f", "meta-data"}
REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"

_JUDGE_TOOL = {
    "name": "grade",
    "description": "Grade the system answer against the reference answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": ["CORRECT", "PARTIAL", "INCORRECT"]},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
    },
}
_JUDGE_SYSTEM = (
    "You grade a document-QA system. Compare the SYSTEM answer to the REFERENCE answer for the "
    "QUESTION. Verdict CORRECT if it conveys the same key fact (numbers and units must match); "
    "PARTIAL if partly right or missing detail; INCORRECT if wrong or unsupported. Be strict on numbers."
)


@dataclass
class GoldItem:
    question: str
    answer: str
    type: str
    source_file: str
    company: str


def load_gold() -> list[GoldItem]:
    items: list[GoldItem] = []
    for qa in sorted(DATA.glob("*_qa.jsonl")):
        source_file = qa.name.replace("_qa.jsonl", ".pdf")
        meta = doc_meta_for(source_file)
        for line in qa.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(
                GoldItem(d["question"], d["answer"], d["type"], source_file, meta.company)
            )
    return items


def judge(client, model, q: str, gold: str, sys_ans: str) -> tuple[str, str]:
    msg = client.messages.create(
        model=model,
        max_tokens=300,
        temperature=0,
        system=_JUDGE_SYSTEM,
        tools=[_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "grade"},
        messages=[
            {
                "role": "user",
                "content": f"QUESTION: {q}\n\nREFERENCE: {gold}\n\nSYSTEM: {sys_ans}",
            }
        ],
    )
    for b in msg.content:
        if b.type == "tool_use":
            return b.input.get("verdict", "INCORRECT"), b.input.get("reason", "")
    return "INCORRECT", "no verdict"


def main() -> int:
    settings = get_settings()
    engine = QAEngine(settings)
    client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=8)
    gold = load_gold()
    print(f"Evaluating {len(gold)} gold questions (parallel)...\n")

    progress = {"done": 0}
    lock = Lock()

    def process(g: GoldItem) -> dict:
        row = {
            "type": g.type,
            "company": g.company,
            "question": g.question,
            "gold": g.answer,
        }
        try:
            scoped = engine.answer(g.question, document=g.company)
            unscoped = engine.answer(g.question)
            row.update(
                scoped_answer=scoped.answer,
                scoped_found=scoped.answer_found,
                unscoped_found=unscoped.answer_found,
            )
            if g.type in ANSWERABLE:
                verdict, reason = judge(client, settings.judge_model, g.question, g.answer, scoped.answer)
                row["verdict"] = verdict
                row["reason"] = reason
                row["routed"] = any(c.file_name == g.source_file for c in unscoped.citations)
            else:
                row["verdict"] = "ABSTAINED" if not scoped.answer_found else "HALLUCINATED"
                row["routed"] = None
        except Exception as e:  # never let one question crash the whole run
            row.setdefault("scoped_found", None)
            row.setdefault("unscoped_found", None)
            row["verdict"] = "ERROR"
            row["reason"] = f"{type(e).__name__}: {e}"
            row["routed"] = None
        with lock:
            progress["done"] += 1
            print(f"[{progress['done']:2}/{len(gold)}] {g.type:13} {row['verdict']:12} | {g.question[:55]}",
                  flush=True)
        return row

    # Modest concurrency: throughput is bounded by the Anthropic input-token/min rate limit,
    # not latency, so a few workers + SDK retry/backoff is the sweet spot (more just causes 429s).
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(process, gold))

    _write_report(rows)
    return 0


def _write_report(rows: list[dict]) -> None:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    lines = ["# Evaluation Results", ""]
    lines.append(f"Total gold questions: **{len(rows)}**. Judge: Claude (LLM-as-judge), temperature 0.")
    lines.append("")
    lines.append("## Accuracy by question type (scoped to source document)")
    lines.append("")
    lines.append("| Type | N | Correct | Partial | Incorrect | Accuracy (C) | C+P |")
    lines.append("|---|---|---|---|---|---|---|")
    tot_c = tot_p = tot_n = 0
    for t in ["text-only", "multimodal-t", "multimodal-f", "meta-data"]:
        rs = by_type.get(t, [])
        if not rs:
            continue
        c = sum(r["verdict"] == "CORRECT" for r in rs)
        p = sum(r["verdict"] == "PARTIAL" for r in rs)
        n = len(rs)
        tot_c += c
        tot_p += p
        tot_n += n
        lines.append(f"| {t} | {n} | {c} | {p} | {n-c-p} | {c/n:.0%} | {(c+p)/n:.0%} |")
    if tot_n:
        lines.append(
            f"| **answerable total** | {tot_n} | {tot_c} | {tot_p} | {tot_n-tot_c-tot_p} "
            f"| **{tot_c/tot_n:.0%}** | **{(tot_c+tot_p)/tot_n:.0%}** |"
        )
    lines.append("")

    # abstention
    un = by_type.get("unanswerable", [])
    if un:
        ok = sum(r["verdict"] == "ABSTAINED" for r in un)
        un_ok = sum(not r["unscoped_found"] for r in un)
        lines.append("## Abstention on unanswerable questions (anti-hallucination)")
        lines.append("")
        lines.append(f"- Correctly abstained (scoped): **{ok}/{len(un)}** ({ok/len(un):.0%})")
        lines.append(f"- Correctly abstained (unscoped, all docs): **{un_ok}/{len(un)}** ({un_ok/len(un):.0%})")
        lines.append("")

    # document routing
    answerable = [r for r in rows if r["type"] in ANSWERABLE]
    routed = sum(bool(r["routed"]) for r in answerable)
    if answerable:
        lines.append("## Multi-document routing (unscoped: correct report cited among all 5)")
        lines.append("")
        lines.append(f"- Correct source document cited: **{routed}/{len(answerable)}** ({routed/len(answerable):.0%})")
        lines.append("")

    # per-question detail
    lines.append("## Per-question detail")
    lines.append("")
    lines.append("| Type | Company | Verdict | Routed | Question |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        routed = "—" if r["routed"] is None else ("✓" if r["routed"] else "✗")
        q = r["question"].replace("|", "\\|")
        lines.append(f"| {r['type']} | {r['company'].split()[0]} | {r['verdict']} | {routed} | {q} |")
    lines.append("")

    (REPO / "eval" / "results.md").write_text("\n".join(lines))
    (REPO / "eval" / "results.json").write_text(json.dumps(rows, indent=2))
    print("\nWrote eval/results.md and eval/results.json")
    if tot_n:
        print(f"Answerable accuracy (strict): {tot_c}/{tot_n} = {tot_c/tot_n:.0%}; C+P = {(tot_c+tot_p)/tot_n:.0%}")
    if un:
        print(f"Abstention (scoped): {ok}/{len(un)}")


if __name__ == "__main__":
    raise SystemExit(main())
