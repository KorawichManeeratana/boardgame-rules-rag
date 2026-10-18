"""Make the eval set survive a re-chunk.

The labelled answers are chunk ids, and an id only means something inside the
chunking that produced it. Re-ingest with a different size and every gold id
silently points at a different passage -- the eval still runs, it just scores
against the wrong thing, which is worse than failing.

Two steps, run in this order around any change to chunking:

  BEFORE re-ingest:  pin the gold ids to the TEXT they currently refer to
      uv run python scripts/pin_eval_text.py

  AFTER re-ingest:   resolve that text back to ids in the new index
      uv run python scripts/pin_eval_text.py --reindex

After the first pin, `relevant_text` is the real answer key and `relevant` is
a convenience for the labelling tool. Both are committed, so the numbers stay
reproducible from the repo.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.align import containment, same_passage, shingles  # noqa: E402
from rag import config  # noqa: E402
from rag.store import VectorStore  # noqa: E402


def pin(rows: list[dict], store: VectorStore) -> tuple[int, list[str]]:
    """Copy each gold id's text into the row. Idempotent."""
    by_id = {r["id"]: r["text"] for r in store.records}
    pinned = 0
    problems: list[str] = []

    for row in rows:
        if not row.get("answerable", True):
            row.setdefault("relevant_text", [])
            continue
        if row.get("relevant_text"):
            continue  # already pinned; leave it alone
        texts = []
        for cid in row["relevant"]:
            if cid in by_id:
                texts.append(by_id[cid])
            else:
                problems.append(f"{cid} (question: {row['q'][:50]}...)")
        if texts:
            row["relevant_text"] = texts
            pinned += 1
    return pinned, problems


def reindex(rows: list[dict], store: VectorStore) -> tuple[int, list[str]]:
    """Point `relevant` at whichever chunks of the CURRENT index hold the gold text.

    A gold passage with no match is worth knowing about: the new chunking
    broke it up badly enough that no single chunk still carries it.
    """
    updated = 0
    lost: list[str] = []

    for row in rows:
        if not row.get("answerable", True):
            continue
        gold = row.get("relevant_text") or []
        if not gold:
            lost.append(f"(not pinned) {row['q'][:60]}")
            continue

        ids: list[str] = []
        for text in gold:
            want = len(shingles(text))
            # Several chunks can contain the gold equally well once the
            # chunks get bigger. Break that tie on tightness of fit, so the
            # id points at the passage rather than at the padding around it.
            matches = [
                (-containment(r["text"], text),
                 abs(len(shingles(r["text"])) - want),
                 r["id"])
                for r in store.records
                if same_passage(r["text"], text)
            ]
            if matches:
                matches.sort()
                ids.append(matches[0][2])
        if ids:
            row["relevant"] = sorted(set(ids))
            updated += 1
        else:
            lost.append(row["q"][:60])
    return updated, lost


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin the eval set's gold answers to text.")
    parser.add_argument("--reindex", action="store_true",
                        help="after a re-ingest: refresh `relevant` ids from the pinned text")
    parser.add_argument("--path", type=Path, default=config.EVAL_PATH)
    args = parser.parse_args()

    store = VectorStore.load(config.INDEX_PATH)
    rows = json.loads(args.path.read_text(encoding="utf-8"))
    meta = store.meta
    print(f"index: {len(store)} chunks "
          f"({meta.get('strategy')}, size={meta.get('size')}, overlap={meta.get('overlap')})")
    print(f"eval:  {len(rows)} question(s)\n")

    if args.reindex:
        updated, lost = reindex(rows, store)
        print(f"re-pointed {updated} question(s) at the current index.")
        if lost:
            print(f"\n{len(lost)} gold passage(s) no longer survive in any single chunk:")
            for item in lost:
                print(f"  - {item}")
            print("\nThat is a finding, not a bug: this chunking splits those "
                  "passages. They still score correctly (matching is by text), "
                  "but the ids are left as they were.")
    else:
        pinned, problems = pin(rows, store)
        already = sum(1 for r in rows if r.get("relevant_text")) - pinned
        print(f"pinned {pinned} question(s); {already} already had text.")
        if problems:
            print(f"\n{len(problems)} gold id(s) are not in this index:")
            for item in problems:
                print(f"  - {item}")
            print("\nThe index has been rebuilt since labelling. Restore the "
                  "chunking those labels were made against, pin, then re-ingest.")

    args.path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {args.path}")


if __name__ == "__main__":
    main()