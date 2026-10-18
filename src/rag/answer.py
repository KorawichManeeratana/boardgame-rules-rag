"""Assemble the retrieved chunks into a grounded, cited answer.

Three things keep this honest:
  1. "Use ONLY the context" -- the model may not fall back on what it happens
     to know about Wingspan from training.
  2. An explicit permission to say "I don't know" -- without it the model
     invents a plausible rule rather than admitting the gap.
  3. The citation list is built by OUR code from the hits we actually fed in,
     not copied from the model's prose, so it cannot cite a page it never saw.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import config
from .embeddings import client

NO_ANSWER = "I couldn't find that in the rulebooks."

SYSTEM = (
    "You answer board game rules questions for players mid-game.\n"
    "Answer using ONLY the numbered context below. Do not use any other "
    "knowledge about this game, even if you are confident.\n"
    f'If the context does not contain the answer, reply exactly: "{NO_ANSWER}"\n'
    "Be concise -- two or three sentences. Cite the context you used with "
    "its number in square brackets, like [1] or [2]."
)


@dataclass
class Answer:
    text: str
    hits: list[dict] = field(default_factory=list)
    grounded: bool = True

    def citations(self) -> list[str]:
        """Built from the hits WE retrieved -- the model cannot fake these."""
        return [
            f"[{i}] {h['source']} p.{h['page']}  (score {h['score']:.3f})"
            for i, h in enumerate(self.hits, start=1)
        ]

    def format(self) -> str:
        # No sources under a refusal. Retrieval can return something that
        # clears the floor and still answers nothing -- printing "I couldn't
        # find that" above a Sources list invites the reader to go check a
        # citation that was never used.
        if not self.hits or not self.grounded:
            return self.text
        return self.text + "\n\nSources:\n" + "\n".join(self.citations())


def assemble(question: str, hits: list[dict]) -> str:
    """Number the chunks so the model can cite them by index."""
    blocks = [
        f"[{i}] {h['text']}\n(source: {h['source']} p.{h['page']})"
        for i, h in enumerate(hits, start=1)
    ]
    return "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def generate(question: str, hits: list[dict]) -> Answer:
    """Short-circuit when retrieval found nothing -- don't call the model.

    Spending a call to have it answer from an empty context is how a confident
    fabrication gets in. No evidence, no answer.
    """
    if not hits:
        return Answer(NO_ANSWER, hits=[], grounded=False)

    resp = client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=config.TEMPERATURE,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": assemble(question, hits)},
        ],
    )
    text = (resp.choices[0].message.content or "").strip()
    return Answer(text, hits=hits, grounded=NO_ANSWER.lower() not in text.lower())