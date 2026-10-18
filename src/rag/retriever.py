"""Retrieval: question -> a list of chunks, possibly EMPTY.

The empty list is the whole point. It is the single signal the answer layer
uses to say "I couldn't find that in the rulebooks" instead of inventing one.
Everything above this module trusts only that contract, so Day 4 can swap the
insides (chunking, scoring, a reranker) without touching the answer code.
"""

from __future__ import annotations

from . import config
from .embeddings import embed_one
from .store import VectorStore


def dedupe(hits: list[dict], threshold: float = 0.8) -> list[dict]:
    """Drop a hit that mostly repeats one already kept, best-first.

    Chunk overlap and the appendix restating the rulebook both produce
    near-duplicates, and each one costs a top-k slot a different fact could
    have used.
    """
    kept: list[dict] = []
    for hit in hits:
        words = set(hit["text"].lower().split())
        if not words:
            continue
        if all(
            len(words & set(k["text"].lower().split())) / len(words) < threshold
            for k in kept
        ):
            kept.append(hit)
    return kept


class Retriever:
    def __init__(
        self,
        store: VectorStore,
        top_k: int = config.TOP_K,
        min_score: float = config.MIN_SCORE,
    ):
        self.store = store
        self.top_k = top_k
        self.min_score = min_score

    def retrieve(self, question: str, game: str | None = None) -> list[dict]:
        """Embed, rank, drop weak matches, dedupe, return at most top_k hits."""
        query_vec = embed_one(question)

        # Over-fetch, because the floor and the dedupe both remove hits and we
        # still want top_k good ones to survive.
        candidates = self.store.search(query_vec, k=self.top_k * 3, game=game)

        above_floor = [h for h in candidates if h["score"] >= self.min_score]
        return dedupe(above_floor)[: self.top_k]
