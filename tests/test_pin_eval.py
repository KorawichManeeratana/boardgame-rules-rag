"""Pinning protects hours of labelling from a one-line config change, so the
glue around it is worth testing even though it is thin.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pin_eval_text import pin, reindex  # noqa: E402
from rag.store import VectorStore  # noqa: E402

LONG_A = ("Round Structure when all players have placed all of their action cubes "
          "the round is over follow these steps in order remove all action cubes")
LONG_B = ("Powers on bird cards fall into three categories when activated brown "
          "these powers may be triggered whenever you use the matching habitat")


def store_with(*pairs):
    return VectorStore([
        {"id": cid, "text": text, "source": "r.pdf", "page": 1,
         "game": "w", "embedding": [0.0]}
        for cid, text in pairs
    ])


def test_pin_copies_the_text_behind_each_gold_id():
    rows = [{"q": "q1", "relevant": ["c0000"], "answerable": True}]
    pinned, problems = pin(rows, store_with(("c0000", LONG_A)))
    assert pinned == 1
    assert problems == []
    assert rows[0]["relevant_text"] == [LONG_A]


def test_pin_is_idempotent():
    rows = [{"q": "q1", "relevant": ["c0000"], "answerable": True,
             "relevant_text": ["already here"]}]
    pinned, _ = pin(rows, store_with(("c0000", LONG_A)))
    assert pinned == 0
    assert rows[0]["relevant_text"] == ["already here"]


def test_pin_reports_an_id_the_index_no_longer_has():
    rows = [{"q": "q1", "relevant": ["c9999"], "answerable": True}]
    pinned, problems = pin(rows, store_with(("c0000", LONG_A)))
    assert pinned == 0
    assert len(problems) == 1


def test_pin_gives_unanswerable_rows_an_empty_list():
    rows = [{"q": "q1", "relevant": [], "answerable": False}]
    pin(rows, store_with(("c0000", LONG_A)))
    assert rows[0]["relevant_text"] == []


def test_reindex_follows_the_text_to_its_new_id():
    rows = [{"q": "q1", "relevant": ["c0000"], "answerable": True,
             "relevant_text": [LONG_A]}]
    # Same passage, different chunking: it is now c0007, and bigger.
    after = store_with(("c0003", LONG_B), ("c0007", "preamble. " + LONG_A + " tail."))
    updated, lost = reindex(rows, after)
    assert updated == 1
    assert lost == []
    assert rows[0]["relevant"] == ["c0007"]


def test_reindex_reports_a_passage_no_chunk_still_carries():
    rows = [{"q": "q1", "relevant": ["c0000"], "answerable": True,
             "relevant_text": [LONG_A]}]
    updated, lost = reindex(rows, store_with(("c0000", LONG_B)))
    assert updated == 0
    assert len(lost) == 1


def test_reindex_flags_a_row_that_was_never_pinned():
    rows = [{"q": "q1", "relevant": ["c0000"], "answerable": True}]
    updated, lost = reindex(rows, store_with(("c0000", LONG_A)))
    assert updated == 0
    assert "not pinned" in lost[0]


def test_reindex_picks_the_best_containing_chunk():
    rows = [{"q": "q1", "relevant": [], "answerable": True, "relevant_text": [LONG_A]}]
    # Both contain it; the tighter one should win.
    after = store_with(
        ("c0001", LONG_A + " " + LONG_B + " plus a great deal of unrelated padding text"),
        ("c0002", LONG_A),
    )
    reindex(rows, after)
    assert rows[0]["relevant"] == ["c0002"]