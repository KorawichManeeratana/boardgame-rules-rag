"""Baselines the vector retriever has to beat.

A metric without a baseline means nothing: recall@4 = 0.80 is impressive or
embarrassing depending on what a dumb method scores on the same questions.

  random    sanity floor -- if the real system can't beat shuffling, stop.
  bm25      the honest competitor: lexical scoring with IDF, TF saturation
            and a length penalty. Unbeatable on exact rare tokens, blind to
            synonyms -- which is exactly where embeddings should win.
"""

from __future__ import annotations

import math
import random
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """BM25 in its three knobs: IDF rarity, TF saturation, length penalty."""

    def __init__(self, records: list[dict], k1: float = 1.5, b: float = 0.75):
        self.ids = [r["id"] for r in records]
        self.docs = [tokenize(r["text"]) for r in records]
        self.tf = [Counter(d) for d in self.docs]
        self.lengths = [len(d) for d in self.docs]
        self.avgdl = sum(self.lengths) / len(self.docs) if self.docs else 0.0
        self.k1 = k1
        self.b = b

        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(set(doc))
        self.df = df
        self.n = len(self.docs)

    def idf(self, term: str) -> float:
        n = self.df.get(term, 0)
        return math.log((self.n - n + 0.5) / (n + 0.5) + 1)

    def score(self, query: str, doc_index: int) -> float:
        tf = self.tf[doc_index]
        length = self.lengths[doc_index]
        total = 0.0
        for term in tokenize(query):
            freq = tf.get(term, 0)
            if not freq:
                continue
            # More hits help, with diminishing returns; a short focused
            # passage is worth more than the same hits buried in a long one.
            denom = freq + self.k1 * (1 - self.b + self.b * length / self.avgdl)
            total += self.idf(term) * freq * (self.k1 + 1) / denom
        return total

    def search(self, query: str, k: int) -> list[str]:
        scored = [(self.score(query, i), i) for i in range(self.n)]
        scored.sort(key=lambda p: (-p[0], p[1]))
        return [self.ids[i] for s, i in scored[:k] if s > 0]


class RandomBaseline:
    """The sanity floor. Seeded, so the reported number is reproducible."""

    def __init__(self, records: list[dict], seed: int = 0):
        self.ids = [r["id"] for r in records]
        self.rng = random.Random(seed)

    def search(self, query: str, k: int) -> list[str]:
        return self.rng.sample(self.ids, min(k, len(self.ids)))
