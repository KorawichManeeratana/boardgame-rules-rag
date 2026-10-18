"""Capture the demo as text, while the proxy is working.

    uv run --env-file .env python scripts/demo.py --out eval/TRANSCRIPT.md

Runs a fixed script of questions through both paths and writes what came
back. Three reasons this exists rather than a folder of screenshots:

  * the proxy is intermittent, and a transcript captured today still shows
    real behaviour on the day the report is written
  * terminal screenshots are unreadable in a printed PDF and you cannot copy
    a number out of one
  * the questions double as a running order for the video -- same questions,
    same sequence, no deciding what to type on camera

The output is a record of one run, not a claim about every run: the model is
at temperature 0, but retrieval scores depend on the index, so the file says
which index produced it.
"""

from __future__ import annotations

import argparse
import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import config  # noqa: E402
from rag.agent import Agent  # noqa: E402
from rag.cli import ask, ask_agent  # noqa: E402
from rag.retriever import Retriever  # noqa: E402
from rag.store import VectorStore  # noqa: E402

# (question, modes, what this is meant to show)
SCRIPT = [
    (
        "What is the egg limit on a bird?",
        ["agent"],
        "The ordinary case: search once, answer, cite the passages used.",
    ),
    (
        "thanks, that helped!",
        ["rag", "agent"],
        "Side by side. The RAG path retrieves for a social message because "
        "it always retrieves; the agent skips the tool entirely. This is "
        "the difference that justifies the agent loop.",
    ),
    (
        "What is the capital of France?",
        ["rag"],
        "Out of scope. Retrieval returns nothing above the floor, so no "
        "model call is made at all and the answer is the refusal line.",
    ),
    (
        "Can a bird power let me go over my egg limit?",
        ["agent"],
        "THE FAILURE CASE (FC-5). The answer is 'no power allows it', cited "
        "correctly -- and wrong, because the Asia expansion card holding the "
        "exception never reached the top-4. Nothing in the output is false "
        "and a player following it would still play the rule wrong.",
    ),
    (
        "go 2 over its egg limit once between turns",
        ["rag"],
        "The exception IS in the corpus. FC-5 is retrieval missing it, not "
        "the rulebook lacking it.",
    ),
]


def capture(fn, *args) -> str:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn(*args)
    return buffer.getvalue().strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a demo transcript.")
    parser.add_argument("--out", type=Path, default=Path("eval/TRANSCRIPT.md"))
    args = parser.parse_args()

    store = VectorStore.load(config.INDEX_PATH)
    retriever = Retriever(store)
    meta = store.meta

    lines = [
        "# Demo transcript",
        "",
        f"One run against the index of {len(store)} chunks "
        f"({meta.get('strategy')}, size={meta.get('size')}, "
        f"overlap={meta.get('overlap')}), "
        f"top_k={config.TOP_K}, min_score={config.MIN_SCORE}, "
        f"temperature={config.TEMPERATURE}.",
        "",
        "Reproduce any line below with the command shown above it.",
        "",
    ]

    for question, modes, note in SCRIPT:
        lines += [f"## {question}", "", note, ""]
        for mode in modes:
            flag = " --agent" if mode == "agent" else ""
            label = "agent" if mode == "agent" else "retrieve-then-read"
            print(f"  [{label}] {question}", flush=True)
            if mode == "agent":
                output = capture(ask_agent, Agent(retriever), question, True)
            else:
                output = capture(ask, retriever, question, None, True)
            lines += [
                f"**{label}**",
                "",
                "```",
                f"$ python -m rag.cli{flag} --show-chunks \"{question}\"",
                output,
                "```",
                "",
            ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()