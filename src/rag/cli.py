"""Ask the rulebooks a question from the terminal.

    uv run --env-file .env python -m rag.cli "How many eggs can a nest hold?"
    uv run --env-file .env python -m rag.cli --agent "thanks, that helped!"
    uv run --env-file .env python -m rag.cli --game wingspan --show-chunks "..."
    uv run --env-file .env python -m rag.cli          # interactive

Two modes, on purpose, because the difference is the point:

  default   retrieve once, then answer. Simple and predictable; it also
            retrieves for "thanks", and cannot recover from a bad first
            search.
  --agent   retrieval is a tool the model chooses to call, skip, or call
            again with better words. See rag/agent.py.

--show-chunks prints what retrieval actually returned. When an answer is
wrong, that is the first thing to look at -- most bad answers are bad
retrieval, not a bad model. With --agent it also prints the tool trace.
"""

from __future__ import annotations

import argparse

from . import config
from .agent import Agent
from .answer import generate
from .retriever import Retriever
from .store import VectorStore


def show_chunks(hits: list[dict]) -> None:
    if not hits:
        print("\n  (retrieval returned nothing above the min_score floor)")
        return
    print()
    for i, h in enumerate(hits, start=1):
        preview = " ".join(h["text"].split())[:160]
        print(f"  [{i}] {h['score']:.3f}  {h['source']} p.{h['page']}")
        print(f"      {preview}...")


def show_trace(trace: list[dict], stopped_because: str) -> None:
    print()
    if not trace:
        print("  (the agent answered without searching)")
    for entry in trace:
        print(f"  step {entry['step']}: {entry['tool']}({entry['arguments']})")
        print(f"          -> {', '.join(entry['hits']) or 'nothing'}")
    print(f"  stopped: {stopped_because}")


def ask(retriever: Retriever, question: str, game: str | None, debug: bool) -> None:
    hits = retriever.retrieve(question, game=game)
    if debug:
        show_chunks(hits)
    print()
    print(generate(question, hits).format())


def ask_agent(agent: Agent, question: str, debug: bool) -> None:
    result = agent.run(question)
    if debug:
        show_trace(result.trace, result.stopped_because)
    print()
    print(result.format())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the rulebooks a question.")
    parser.add_argument("question", nargs="*", help="leave empty for interactive mode")
    parser.add_argument("--game", help="restrict to one game (pre-filter, then rank)")
    parser.add_argument("--top-k", type=int, default=config.TOP_K)
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    parser.add_argument("--show-chunks", action="store_true",
                        help="print the retrieved chunks (and the tool trace, with --agent)")
    parser.add_argument("--agent", action="store_true",
                        help="let the model decide whether to search")
    parser.add_argument("--max-steps", type=int, default=4,
                        help="agent only: cap on turns before an answer is forced")
    args = parser.parse_args()

    store = VectorStore.load(config.INDEX_PATH)
    retriever = Retriever(store, top_k=args.top_k, min_score=args.min_score)
    agent = Agent(retriever, max_steps=args.max_steps) if args.agent else None

    def handle(question: str) -> None:
        if agent:
            ask_agent(agent, question, args.show_chunks)
        else:
            ask(retriever, question, args.game, args.show_chunks)

    if args.question:
        handle(" ".join(args.question))
        return

    mode = "agent" if args.agent else "retrieve-then-read"
    print(f"{len(store)} chunks indexed, {mode} mode. "
          "Ask a rules question, or Ctrl-C to quit.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question:
            handle(question)


if __name__ == "__main__":
    main()