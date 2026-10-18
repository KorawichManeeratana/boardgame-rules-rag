"""Score the retriever against the labelled set and the baselines.

    uv run --env-file .env python -m eval.run_eval
    uv run --env-file .env python -m eval.run_eval --k 4 --out eval/RESULTS.md

Reports two things, because they fail differently:

  ranking quality   recall@k / precision@k / MRR / nDCG over the ANSWERABLE
                    questions -- did we fetch the right chunk, and how high?
  abstention        on the UNANSWERABLE questions, did the min_score floor
                    correctly return nothing? A system that answers those is
                    hallucinating, however good its recall looks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.align import evaluate as evaluate_by_text  # noqa: E402
from eval.baselines import BM25, RandomBaseline  # noqa: E402
from eval.metrics import evaluate as evaluate_by_id, format_table  # noqa: E402
from rag import config  # noqa: E402
from rag.embeddings import cosine, embed  # noqa: E402
from rag.store import VectorStore  # noqa: E402


def load_eval_set(path: Path) -> list[dict]:
    if not path.exists():
        raise SystemExit(
            f"No eval set at {path}. Build one:  python scripts/label.py"
        )
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not rows:
        raise SystemExit("The eval set is empty.")
    return rows


def vector_rankings(store: VectorStore, questions: list[str], k: int) -> list[list[str]]:
    """One batched embedding call for every question, then rank in Python."""
    query_vecs = embed(questions)
    rankings = []
    for qv in query_vecs:
        scored = [(cosine(qv, r["embedding"]), r["id"]) for r in store.records]
        scored.sort(key=lambda p: (-p[0], p[1]))
        rankings.append([cid for _, cid in scored[:k]])
    return rankings


def vector_scores(store: VectorStore, questions: list[str]) -> list[float]:
    """Top-1 cosine per question -- what the min_score floor is compared to."""
    return [
        max(cosine(qv, r["embedding"]) for r in store.records)
        for qv in embed(questions)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval against baselines.")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--min-score", type=float, default=config.MIN_SCORE)
    parser.add_argument("--out", type=Path, help="also write a markdown report here")
    args = parser.parse_args()

    store = VectorStore.load(config.INDEX_PATH)
    rows = load_eval_set(config.EVAL_PATH)

    answerable = [r for r in rows if r.get("answerable", True)]
    unanswerable = [r for r in rows if not r.get("answerable", True)]
    print(f"index: {len(store)} chunks  ({store.meta.get('strategy')}, "
          f"size={store.meta.get('size')}, overlap={store.meta.get('overlap')})")
    print(f"eval:  {len(answerable)} answerable + {len(unanswerable)} unanswerable\n")

    # --- ranking quality ----------------------------------------------------
    questions = [r["q"] for r in answerable]
    by_id = {r["id"]: r["text"] for r in store.records}

    # Prefer the pinned gold TEXT when the eval set carries it: ids stop
    # meaning anything once the chunking changes, and scoring against stale
    # ids fails silently rather than loudly. See scripts/pin_eval_text.py.
    by_text = all(row.get("relevant_text") for row in answerable)
    if by_text:
        gold = [row["relevant_text"] for row in answerable]
        print("matching gold answers by text (chunking-independent)\n")

        def score(ranked_ids: list[list[str]]) -> dict:
            ranked_texts = [[by_id[i] for i in ids] for ids in ranked_ids]
            return evaluate_by_text(list(zip(ranked_texts, gold)), args.k)
    else:
        gold = [set(row["relevant"]) for row in answerable]
        print("matching gold answers by chunk id -- run scripts/pin_eval_text.py\n"
              "to pin them to text, or these numbers break on the next re-ingest\n")

        def score(ranked_ids: list[list[str]]) -> dict:
            return evaluate_by_id(list(zip(ranked_ids, gold)), args.k)

    bm25 = BM25(store.records)
    rnd = RandomBaseline(store.records)

    systems = {
        "random (floor)": [rnd.search(q, args.k) for q in questions],
        "BM25 (keyword)": [bm25.search(q, args.k) for q in questions],
        "bge-m3 (vector)": vector_rankings(store, questions, args.k),
    }
    results = {name: score(ranked) for name, ranked in systems.items()}
    table = format_table(results)
    print(table)

    # --- abstention ---------------------------------------------------------
    abstention = ""
    if unanswerable:
        tops = vector_scores(store, [r["q"] for r in unanswerable])
        correct = sum(1 for s in tops if s < args.min_score)
        worst = max(tops)
        abstention = (
            f"\nAbstention at min_score={args.min_score}: "
            f"{correct}/{len(unanswerable)} unanswerable questions correctly "
            f"returned nothing.\nHighest top-1 score among them: {worst:.3f} "
            f"-- a floor below this would start hallucinating."
        )
        print(abstention)

        answerable_tops = vector_scores(store, questions)
        lowest = min(answerable_tops)
        print(f"Lowest top-1 score among answerable questions: {lowest:.3f}")
        print(f"Safe floor window: {worst:.3f} .. {lowest:.3f}"
              if worst < lowest else
              f"WARNING: the two ranges overlap ({worst:.3f} >= {lowest:.3f}); "
              "no single floor separates them cleanly.")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            f"# Retrieval results (k={args.k})\n\n"
            f"Index: {len(store)} chunks, strategy={store.meta.get('strategy')}, "
            f"size={store.meta.get('size')}, overlap={store.meta.get('overlap')}\n\n"
            f"{table}\n{abstention}\n",
            encoding="utf-8",
        )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()