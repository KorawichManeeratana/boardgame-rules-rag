"""Worked examples from the lecture, pinned as tests.

If a metric ever disagrees with these, the number in the report is wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.baselines import BM25, tokenize  # noqa: E402
from eval.metrics import (  # noqa: E402
    evaluate,
    format_table,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


# --- recall / precision -----------------------------------------------------

def test_recall_and_precision_share_the_intersection_but_not_the_denominator():
    retrieved, relevant = ["d2", "d7", "d5"], {"d7", "d9"}
    assert recall_at_k(retrieved, relevant, 2) == 0.5      # 1 of 2 relevant
    assert precision_at_k(retrieved, relevant, 2) == 0.5   # 1 of 2 returned


def test_growing_k_raises_recall_and_lowers_precision():
    retrieved = ["d1", "d2", "x", "y", "d3"]
    relevant = {"d1", "d2", "d3"}
    assert recall_at_k(retrieved, relevant, 2) < recall_at_k(retrieved, relevant, 5)
    assert precision_at_k(retrieved, relevant, 2) > precision_at_k(retrieved, relevant, 5)


def test_empty_relevant_set_is_zero_not_a_crash():
    assert recall_at_k(["d1"], set(), 3) == 0.0
    assert ndcg_at_k(["d1"], set(), 3) == 0.0


# --- MRR --------------------------------------------------------------------

def test_reciprocal_rank_scans_past_the_cutoff_recall_stops_at():
    retrieved, relevant = ["d8", "d4", "d1"], {"d1"}
    assert recall_at_k(retrieved, relevant, 2) == 0.0        # outside top-2
    assert round(reciprocal_rank(retrieved, relevant), 2) == 0.33  # found at rank 3


def test_reciprocal_rank_is_zero_when_nothing_relevant_appears():
    assert reciprocal_rank(["a", "b"], {"z"}) == 0.0


def test_a_hit_at_rank_one_scores_one():
    assert reciprocal_rank(["d1", "d2"], {"d1"}) == 1.0


# --- nDCG -------------------------------------------------------------------

def test_ndcg_is_one_when_the_relevant_docs_are_on_top():
    assert ndcg_at_k(["d1", "d2", "x"], {"d1", "d2"}, 3) == 1.0


def test_ndcg_penalises_a_buried_hit():
    top = ndcg_at_k(["d1", "x", "y"], {"d1"}, 3)
    deep = ndcg_at_k(["x", "y", "d1"], {"d1"}, 3)
    assert top == 1.0
    assert deep < top


# --- aggregation ------------------------------------------------------------

def test_evaluate_averages_over_queries():
    runs = [(["d1"], {"d1"}), (["x"], {"d2"})]
    out = evaluate(runs, k=1)
    assert out["n"] == 2
    assert out["recall@1"] == 0.5
    assert out["MRR"] == 0.5


def test_format_table_has_a_row_per_system():
    table = format_table({"a": evaluate([(["d1"], {"d1"})], 1)})
    assert "| a |" in table
    assert "recall@1" in table


# --- BM25 -------------------------------------------------------------------

def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("Lay 2 Eggs!") == ["lay", "2", "eggs"]


def test_bm25_prefers_the_doc_containing_the_rare_term():
    records = [
        {"id": "c0", "text": "the the the the the bird"},
        {"id": "c1", "text": "the duet map scores contiguous tokens"},
        {"id": "c2", "text": "the the the the the the"},
    ]
    assert BM25(records).search("duet map", k=1) == ["c1"]


def test_bm25_returns_nothing_when_no_term_matches():
    records = [{"id": "c0", "text": "lay eggs on a bird"}]
    assert BM25(records).search("capital of france", k=3) == []
