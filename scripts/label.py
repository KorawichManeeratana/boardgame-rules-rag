"""Build the labelled eval set, one question at a time.

    uv run --env-file .env python scripts/label.py
    uv run --env-file .env python scripts/label.py --from eval/questions.txt

For each question, this shows candidate chunks and you record the ids that
GENUINELY answer it. Progress saves after every question, so you can stop and
resume. --from walks a plain-text file of questions, one per line, instead of
making you type each one; blank lines and lines starting with # are skipped,
and questions already in the eval set are passed over.

A caution this tool cannot fix for you: ranking the candidates with the same
retriever you are evaluating biases the labels toward what it already finds.
You cannot label a chunk it never shows you, and the measured recall comes
out flattering. Two defences, both built in:

  /find <words>   look up candidates lexically (BM25) instead of by vector
  typing any id   you are never restricted to the list on screen

Use /find on at least a few questions, and write some questions from reading
the rulebook rather than from what you know the system handles.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.baselines import BM25  # noqa: E402
from rag import config  # noqa: E402
from rag.embeddings import cosine, embed_one  # noqa: E402
from rag.store import VectorStore  # noqa: E402

CANDIDATES = 10
PREVIEW = 150
SKIPPED_PATH = Path(__file__).resolve().parents[1] / "eval" / "skipped.txt"

HELP = """
commands
  <ids>          record these chunk ids as the answer   e.g.  c0007 c0012
  none           this question has NO answer in the corpus (tests "I don't know")
  /show <ids>    print those chunks IN FULL -- the previews are truncated,
                 so check before deciding a chunk does not answer the question
  /find <words>  show lexical (BM25) candidates instead of vector ones
  /more          show the next 10 vector candidates
  /skip          throw this question away
  /quit          save and exit
"""


def preview(record: dict, score: float | None = None) -> str:
    text = " ".join(record["text"].split())[:PREVIEW]
    head = f"  {record['id']}"
    if score is not None:
        head += f"  {score:.3f}"
    return f"{head}  {record['source']} p.{record['page']}\n      {text}..."


def load_set(path: Path) -> list[dict]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return []


def save_set(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def load_skipped() -> set[str]:
    """Questions discarded in an earlier run, so they are not asked again."""
    if not SKIPPED_PATH.exists():
        return set()
    return {
        ln.strip() for ln in SKIPPED_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.startswith("#")
    }


def record_skip(question: str) -> None:
    SKIPPED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SKIPPED_PATH.open("a", encoding="utf-8") as fh:
        fh.write(question + "\n")


def read_questions(path: Path, already: set[str]) -> list[str]:
    """One question per line; skip blanks, comments, and ones already done."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]
    return [
        ln for ln in lines
        if ln and not ln.startswith("#") and ln not in already
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Label the retrieval eval set.")
    parser.add_argument("--from", dest="source", type=Path,
                        help="a text file of questions, one per line")
    args = parser.parse_args()

    store = VectorStore.load(config.INDEX_PATH)
    by_id = {r["id"]: r for r in store.records}
    bm25 = BM25(store.records)

    rows = load_set(config.EVAL_PATH)
    print(f"{len(store)} chunks indexed. {len(rows)} question(s) already labelled.")

    queue: list[str] = []
    if args.source:
        done = {r["q"] for r in rows} | load_skipped()
        queue = read_questions(args.source, done)
        print(f"{len(queue)} question(s) left in {args.source}.")
    print(HELP)

    while True:
        if queue:
            question = queue.pop(0)
            print(f"\nquestion > {question}   ({len(queue)} left)")
        else:
            try:
                question = input("\nquestion > ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not question:
            continue
        if question == "/quit":
            break

        query_vec = embed_one(question)
        ranked = sorted(
            ((cosine(query_vec, r["embedding"]), r) for r in store.records),
            key=lambda p: -p[0],
        )
        shown = 0

        def show(start: int, count: int = CANDIDATES) -> int:
            for score, record in ranked[start : start + count]:
                print(preview(record, score))
            return start + count

        shown = show(0)

        while True:
            answer = input("  ids > ").strip()

            if answer == "/skip":
                record_skip(question)
                print(f"  skipped (recorded in {SKIPPED_PATH.name}, won't be asked again)")
                question = None
                break
            if answer == "/quit":
                save_set(config.EVAL_PATH, rows)
                print(f"saved {len(rows)} question(s) to {config.EVAL_PATH}")
                return
            if answer == "/more":
                shown = show(shown)
                continue
            if answer.startswith("/show"):
                for cid in answer[5:].split():
                    record = by_id.get(cid)
                    if record is None:
                        print(f"  unknown id: {cid}")
                        continue
                    print(f"\n  --- {cid}  {record['source']} p.{record['page']} ---")
                    print("  " + " ".join(record["text"].split()))
                print()
                continue
            if answer.startswith("/find"):
                for cid in bm25.search(answer[5:].strip() or question, CANDIDATES):
                    print(preview(by_id[cid]))
                continue
            if answer == "none":
                rows.append({"q": question, "relevant": [], "answerable": False})
                break

            ids = answer.split()
            unknown = [i for i in ids if i not in by_id]
            if unknown or not ids:
                print(f"  unknown id(s): {' '.join(unknown) or '(none given)'}")
                continue
            rows.append({"q": question, "relevant": ids, "answerable": True})
            break

        if question is not None:
            save_set(config.EVAL_PATH, rows)
            answerable = sum(1 for r in rows if r["answerable"])
            print(f"  saved. {len(rows)} total ({answerable} answerable)")

    save_set(config.EVAL_PATH, rows)
    print(f"\nsaved {len(rows)} question(s) to {config.EVAL_PATH}")


if __name__ == "__main__":
    main()