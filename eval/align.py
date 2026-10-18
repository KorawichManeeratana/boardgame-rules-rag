"""Match gold chunks across different chunkings, by text instead of by id.

The eval set names chunk ids, but an id only means something inside the
chunking that produced it: re-run ingest with a different size and c0002 is a
different passage. Comparing two chunkings therefore needs the gold answer
pinned to the TEXT it referred to, and a rule for when a chunk from another
chunking counts as the same answer.

The rule here is shingle containment. Two passages match when one largely
contains the other, which is exactly what happens when a chunking boundary
moves: a bigger chunk swallows the gold passage, a smaller one is a slice of
it. Dividing by the SMALLER shingle set makes the test symmetric, so it works
whichever direction the size moved.

Known bias, and it favours small chunks: a gold passage cut into three slices
is "found" when ANY slice is retrieved, even the slice that happens not to
contain the sentence a reader needs. So a sweep row with much smaller chunks
can post a higher recall than it deserves. Read such a row together with its
chunk count, and sanity-check the winner by hand before believing it.
"""

from __future__ import annotations

import re

SHINGLE = 5          # words per shingle; long enough to be distinctive
THRESHOLD = 0.5      # half the smaller passage must be shared
MIN_SHINGLES = 8     # a passage shorter than ~12 words cannot carry a match


def normalize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def shingles(text: str, n: int = SHINGLE) -> set[str]:
    words = normalize(text)
    if not words:
        return set()
    if len(words) < n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def containment(a: str, b: str) -> float:
    """Shared shingles over the smaller of the two sets (0.0 .. 1.0).

    MIN_SHINGLES floors the denominator. Without it a six-word fragment has
    two shingles, both of them inside the gold passage, and scores a perfect
    1.0 -- every stray snippet would count as the answer.
    """
    sa, sb = shingles(a), shingles(b)
    if not sa or not sb:
        return 0.0
    denominator = max(min(len(sa), len(sb)), MIN_SHINGLES)
    return len(sa & sb) / denominator


def same_passage(candidate: str, gold: str, threshold: float = THRESHOLD) -> bool:
    return containment(candidate, gold) >= threshold


# --- metrics over texts rather than ids -------------------------------------
#
# A gold passage counts as found when ANY retrieved chunk matches it. That
# generalises recall correctly: splitting one gold chunk into two should not
# count as two hits, and merging two golds into one chunk should still satisfy
# both.

def recall_at_k(ranked: list[str], gold: list[str], k: int,
                threshold: float = THRESHOLD) -> float:
    if not gold:
        return 0.0
    top = ranked[:k]
    found = sum(1 for g in gold if any(same_passage(c, g, threshold) for c in top))
    return found / len(gold)


def precision_at_k(ranked: list[str], gold: list[str], k: int,
                   threshold: float = THRESHOLD) -> float:
    if k <= 0 or not gold:
        return 0.0
    top = ranked[:k]
    useful = sum(1 for c in top if any(same_passage(c, g, threshold) for g in gold))
    return useful / k


def reciprocal_rank(ranked: list[str], gold: list[str],
                    threshold: float = THRESHOLD) -> float:
    for rank, candidate in enumerate(ranked, start=1):
        if any(same_passage(candidate, g, threshold) for g in gold):
            return 1 / rank
    return 0.0


def evaluate(runs: list[tuple[list[str], list[str]]], k: int,
             threshold: float = THRESHOLD) -> dict[str, float]:
    if not runs:
        return {"n": 0}
    n = len(runs)
    return {
        "n": n,
        f"recall@{k}": sum(recall_at_k(r, g, k, threshold) for r, g in runs) / n,
        f"precision@{k}": sum(precision_at_k(r, g, k, threshold) for r, g in runs) / n,
        "MRR": sum(reciprocal_rank(r, g, threshold) for r, g in runs) / n,
    }