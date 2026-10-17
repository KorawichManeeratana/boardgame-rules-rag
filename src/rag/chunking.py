"""Split extracted page text into chunks that carry their own citation.

Two strategies, so Day 4 can compare them against the eval set instead of
guessing which is better:

  fixed      - a sliding window over words (the Lecture 5 toy, with overlap)
  paragraph  - respect blank-line boundaries, then re-pack up to a word budget

Every chunk keeps `source` and `page` so an answer can cite "catan.pdf p.7".
"""

from __future__ import annotations

import re


def _window(words: list[str], size: int, overlap: int) -> list[list[str]]:
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and < size")

    step = size - overlap
    out: list[list[str]] = []
    for start in range(0, len(words), step):
        piece = words[start : start + size]
        if not piece:
            break
        out.append(piece)
        if start + size >= len(words):
            break
    return out


def chunk_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Sliding word window. Consecutive chunks share `overlap` words."""
    return [" ".join(w) for w in _window(text.split(), size, overlap)]


def chunk_paragraph(text: str, size: int, overlap: int) -> list[str]:
    """Pack whole paragraphs up to `size` words; window any oversized one.

    Rulebooks are short numbered rules, so a natural boundary usually keeps
    one rule intact -- which is exactly what we want a single hit to return.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    out: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            out.append(" ".join(buffer))
            buffer.clear()

    for para in paragraphs:
        words = para.split()
        if len(words) > size:
            flush()
            out.extend(" ".join(w) for w in _window(words, size, overlap))
            continue
        if len(buffer) + len(words) > size:
            flush()
        buffer.extend(words)
    flush()
    return out


STRATEGIES = {"fixed": chunk_fixed, "paragraph": chunk_paragraph}


def chunk_pages(
    pages: list[dict],
    size: int,
    overlap: int,
    strategy: str = "paragraph",
) -> list[dict]:
    """pages: [{"text":..., "source":..., "page":..., "game":...}] -> chunk records.

    Returns records shaped {id, text, source, page, game} -- the metadata is
    what makes citations and per-game filtering possible later.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; pick one of {sorted(STRATEGIES)}")
    split = STRATEGIES[strategy]

    records: list[dict] = []
    for page in pages:
        for piece in split(page["text"], size, overlap):
            if not piece.strip():
                continue
            records.append(
                {
                    "id": f"c{len(records):04d}",
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                    "game": page.get("game", "unknown"),
                }
            )
    return records
