"""Retrieval metrics. Pure functions, no model, no network -- so they are
cheap to run on every change and easy to unit-test.

Each answers a different question, which is why we report more than one:

  recall@k     did the right chunk land inside the cutoff we actually use?
  precision@k  how much junk did we make the model read?
  MRR          how high did the first right chunk land, anywhere in the list?
  nDCG         credit for every relevant hit, discounted by how deep it sits.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], relevant: set[str] | list[str], k: int) -> float:
    relevant = set(relevant)
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved[:k])) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str] | list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant = set(relevant)
    return sum(1 for doc in retrieved[:k] if doc in relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str] | list[str]) -> float:
    """1/rank of the FIRST relevant hit, scanning the whole list -- no cutoff.

    This is why RR and recall@k can disagree on the same query, on purpose:
    recall asks "inside my cutoff?", RR asks "how high overall?".
    """
    relevant = set(relevant)
    for rank, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1 / rank
    return 0.0


def _dcg(gains: list[float]) -> float:
    return sum(g / math.log2(i + 1) for i, g in enumerate(gains, start=1))


def ndcg_at_k(retrieved: list[str], relevant: set[str] | list[str], k: int) -> float:
    """Binary-relevance nDCG: position-aware, unlike recall and precision."""
    relevant = set(relevant)
    if not relevant:
        return 0.0
    gains = [1.0 if doc in relevant else 0.0 for doc in retrieved[:k]]
    ideal = [1.0] * min(len(relevant), k)
    denom = _dcg(ideal)
    return _dcg(gains) / denom if denom else 0.0


def evaluate(runs: list[tuple[list[str], set[str]]], k: int) -> dict[str, float]:
    """Average each metric over every query in the eval set.

    runs: [(ranked doc ids, the ids that SHOULD have been retrieved), ...]
    """
    if not runs:
        return {"n": 0}
    n = len(runs)
    return {
        "n": n,
        f"recall@{k}": sum(recall_at_k(r, g, k) for r, g in runs) / n,
        f"precision@{k}": sum(precision_at_k(r, g, k) for r, g in runs) / n,
        "MRR": sum(reciprocal_rank(r, g) for r, g in runs) / n,
        f"nDCG@{k}": sum(ndcg_at_k(r, g, k) for r, g in runs) / n,
    }


def format_table(rows: dict[str, dict[str, float]]) -> str:
    """rows: {system name: metrics dict} -> a markdown table for the report."""
    names = list(rows)
    if not names:
        return "(no results)"
    keys = [k for k in rows[names[0]] if k != "n"]

    header = "| system | " + " | ".join(keys) + " |"
    divider = "|---" * (len(keys) + 1) + "|"
    lines = [header, divider]
    for name, metrics in rows.items():
        cells = " | ".join(f"{metrics[k]:.3f}" for k in keys)
        lines.append(f"| {name} | {cells} |")
    return "\n".join(lines)
