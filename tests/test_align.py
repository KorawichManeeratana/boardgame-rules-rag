"""The alignment rule decides every number in the chunking sweep, so it is
worth pinning down exactly when two passages count as the same answer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.align import (  # noqa: E402
    containment,
    evaluate,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    same_passage,
    shingles,
)

GOLD = ("Powers on bird cards fall into 3 categories WHEN ACTIVATED brown "
        "these powers may be activated from right to left whenever you use "
        "the corresponding habitat")


def test_shingles_are_word_ngrams():
    assert shingles("a b c d e f", n=5) == {"a b c d e", "b c d e f"}


def test_short_text_still_yields_one_shingle():
    assert shingles("two words", n=5) == {"two words"}


def test_identical_text_is_full_containment():
    assert containment(GOLD, GOLD) == 1.0


def test_a_bigger_chunk_that_swallows_the_gold_matches():
    bigger = "Some earlier sentence. " + GOLD + " And some following text here."
    assert same_passage(bigger, GOLD)


def test_a_smaller_slice_of_the_gold_matches():
    smaller = " ".join(GOLD.split()[:18])
    assert same_passage(smaller, GOLD)


def test_a_tiny_fragment_does_not_match():
    assert not same_passage(" ".join(GOLD.split()[:6]), GOLD)


def test_unrelated_text_does_not_match():
    assert not same_passage("the automa never draws bonus cards", GOLD)


def test_empty_text_is_not_a_match():
    assert containment("", GOLD) == 0.0
    assert not same_passage("", GOLD)


# --- metrics ---------------------------------------------------------------

def test_recall_counts_each_gold_once_even_if_split_across_chunks():
    half_a = " ".join(GOLD.split()[:18])
    half_b = " ".join(GOLD.split()[5:])
    # Both retrieved chunks match the same gold passage -> still 1 of 1.
    assert recall_at_k([half_a, half_b], [GOLD], k=2) == 1.0


def test_recall_with_two_golds_one_found():
    other = "the automa never draws bonus cards during its turn at all"
    assert recall_at_k([GOLD], [GOLD, other], k=1) == 0.5


def test_precision_counts_useful_slots():
    noise = "completely unrelated passage about scoring duet tokens on a map"
    assert precision_at_k([GOLD, noise], [GOLD], k=2) == 0.5


def test_reciprocal_rank_finds_the_first_match_anywhere():
    noise = "completely unrelated passage about scoring duet tokens on a map"
    assert reciprocal_rank([noise, noise, GOLD], [GOLD]) == 1 / 3


def test_reciprocal_rank_is_zero_with_no_match():
    assert reciprocal_rank(["nothing relevant here at all friend"], [GOLD]) == 0.0


def test_evaluate_averages_over_queries():
    out = evaluate([([GOLD], [GOLD]), (["unrelated text entirely"], [GOLD])], k=1)
    assert out["n"] == 2
    assert out["recall@1"] == 0.5
    assert out["MRR"] == 0.5