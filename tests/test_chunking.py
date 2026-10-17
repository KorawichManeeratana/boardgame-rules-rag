import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.chunking import chunk_fixed, chunk_pages, chunk_paragraph  # noqa: E402


def test_fixed_no_overlap_partitions_the_text():
    assert chunk_fixed("a b c d e", 2, 0) == ["a b", "c d", "e"]


def test_fixed_overlap_shares_edge_words():
    assert chunk_fixed("a b c d e f", 3, 1) == ["a b c", "c d e", "e f"]


def test_empty_text_gives_no_chunks():
    assert chunk_fixed("", 10, 0) == []
    assert chunk_paragraph("   ", 10, 0) == []


def test_paragraph_keeps_a_short_rule_intact():
    text = "Rule one is short.\n\nRule two is also short."
    assert chunk_paragraph(text, size=100, overlap=10) == [
        "Rule one is short. Rule two is also short."
    ]


def test_paragraph_windows_an_oversized_paragraph():
    text = " ".join(str(n) for n in range(10))
    assert len(chunk_paragraph(text, size=4, overlap=1)) > 1


def test_chunk_pages_carries_citation_metadata():
    pages = [{"text": "setup rules here", "source": "catan.pdf", "page": 7, "game": "catan"}]
    records = chunk_pages(pages, size=50, overlap=5)
    assert records[0]["source"] == "catan.pdf"
    assert records[0]["page"] == 7
    assert records[0]["id"] == "c0000"
