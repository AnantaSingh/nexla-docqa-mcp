"""Answer synthesis via Anthropic Claude, with a strict grounded/abstain contract.

Two design choices that protect accuracy:
  - **Forced tool-use** for structured output. Instead of parsing free-text JSON (brittle),
    we make Claude call a `submit_answer` tool, so we always get a typed
    {answer, answer_found, used_sources} object.
  - **The model can only cite labels we gave it.** It returns the source labels it relied on;
    the engine maps those back to real chunk metadata, so a citation can never be fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import Settings

_SYSTEM = """You are a precise financial-document analyst. Answer the user's question using \
ONLY the numbered SOURCES provided below. Follow these rules strictly:

- Use only facts that appear in the SOURCES. Never use outside or prior knowledge.
- Each SOURCE is labelled and tagged with its company, fiscal year, and page. Respect the \
company and fiscal year the question asks about — do not mix figures across companies or years.
- Any base figure you cite must match a SOURCE exactly (including units like millions/billions).
- You MAY compute derived values (differences, sums, ratios, percentage changes) FROM figures that \
appear in the SOURCES — show the arithmetic briefly. Do not invent base figures to compute from.
- If the SOURCES do not clearly contain the answer (or the figures needed to compute it), you MUST \
set answer_found=false and say you could not find it in the provided documents. Do not guess or extrapolate.
- In used_sources, list the labels (e.g. "S1", "S3") of the SOURCES you actually relied on.
- Keep the answer concise and factual."""

_TOOL = {
    "name": "submit_answer",
    "description": "Return the grounded answer and the sources it relied on.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Concise, grounded answer."},
            "answer_found": {
                "type": "boolean",
                "description": "True only if the answer is supported by the SOURCES.",
            },
            "used_sources": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Labels of the SOURCES used, e.g. ['S1','S3'].",
            },
        },
        "required": ["answer", "answer_found", "used_sources"],
    },
}


@dataclass
class GroundedAnswer:
    answer: str
    answer_found: bool
    used_sources: list[str]


class AnswerLLM(Protocol):
    def generate(self, question: str, sources_block: str) -> GroundedAnswer: ...


class ClaudeAnswerLLM:
    def __init__(self, settings: Settings):
        import anthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set. Add it to .env.")
        # max_retries lets the SDK ride through 429 rate-limit responses (honoring Retry-After)
        # instead of crashing — important on lower API tiers and under concurrent load.
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key, max_retries=8)
        self._model = settings.answer_model

    def generate(self, question: str, sources_block: str) -> GroundedAnswer:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0,  # deterministic
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "submit_answer"},
            messages=[
                {
                    "role": "user",
                    "content": f"SOURCES:\n{sources_block}\n\nQUESTION: {question}",
                }
            ],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                data = block.input
                return GroundedAnswer(
                    answer=str(data.get("answer", "")).strip(),
                    answer_found=bool(data.get("answer_found", False)),
                    used_sources=[str(s) for s in data.get("used_sources", [])],
                )
        # Should not happen with forced tool_choice; abstain rather than invent.
        return GroundedAnswer(
            answer="I could not produce a grounded answer from the documents.",
            answer_found=False,
            used_sources=[],
        )

    def generate_from_images(
        self, question: str, images: list[tuple[str, bytes]]
    ) -> GroundedAnswer:
        """Grounded answer from rendered page images (vision fallback for figures/charts).

        Same strict abstain contract as `generate`: answer only from what is visibly in the
        pages, else answer_found=false. `images` is a list of (label, png_bytes).
        """
        import base64

        content: list[dict] = []
        for label, png in images:
            content.append({"type": "text", "text": f"SOURCE {label}:"})
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.standard_b64encode(png).decode(),
                    },
                }
            )
        content.append({"type": "text", "text": f"QUESTION: {question}"})

        msg = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            temperature=0,
            system=_SYSTEM,
            tools=[_TOOL],
            tool_choice={"type": "tool", "name": "submit_answer"},
            messages=[{"role": "user", "content": content}],
        )
        for block in msg.content:
            if block.type == "tool_use" and block.name == "submit_answer":
                d = block.input
                return GroundedAnswer(
                    answer=str(d.get("answer", "")).strip(),
                    answer_found=bool(d.get("answer_found", False)),
                    used_sources=[str(s) for s in d.get("used_sources", [])],
                )
        return GroundedAnswer("Could not read an answer from the page images.", False, [])
