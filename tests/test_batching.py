"""Request batching, which is what broke the 600-word chunking sweep.

Batching by item count made the payload scale with chunk length, so the one
variable the sweep was changing was also changing how much each request
asked the proxy to swallow.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.embeddings import batches  # noqa: E402


def words(n: int) -> str:
    return " ".join(["w"] * n)


def test_everything_is_emitted_exactly_once_and_in_order():
    texts = [f"text{i} " + words(100) for i in range(30)]
    flat = [t for b in batches(texts) for t in b]
    assert flat == texts


def test_a_batch_stays_under_the_word_budget():
    texts = [words(600) for _ in range(20)]
    for batch in batches(texts, max_words=6000):
        assert sum(len(t.split()) for t in batch) <= 6000


def test_big_chunks_make_smaller_batches_not_bigger_payloads():
    small = batches([words(200) for _ in range(64)], max_words=6000)
    large = batches([words(600) for _ in range(64)], max_words=6000)
    # More requests for the larger chunks -- that is the point.
    assert len(large) > len(small)
    worst_small = max(sum(len(t.split()) for t in b) for b in small)
    worst_large = max(sum(len(t.split()) for t in b) for b in large)
    assert worst_large <= 6000 and worst_small <= 6000


def test_item_count_is_also_capped():
    texts = [words(1) for _ in range(200)]
    assert all(len(b) <= 64 for b in batches(texts, max_items=64))


def test_an_oversized_single_text_gets_its_own_request():
    texts = [words(10), words(9000), words(10)]
    out = batches(texts, max_words=6000)
    assert [words(9000)] in out


def test_empty_input_makes_no_requests():
    assert batches([]) == []