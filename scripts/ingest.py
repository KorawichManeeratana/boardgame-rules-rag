"""Offline indexing: PDF -> pages -> chunks -> vectors -> data/index/chunks.json

Run once, and again whenever the rulebooks or the chunking settings change
(a stale index is a real RAG failure mode).

    uv run --env-file .env python scripts/ingest.py
    uv run --env-file .env python scripts/ingest.py --strategy fixed --size 300 --overlap 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pypdf import PdfReader  # noqa: E402

from rag import config  # noqa: E402
from rag.chunking import chunk_pages  # noqa: E402
from rag.embeddings import embed  # noqa: E402
from rag.store import VectorStore  # noqa: E402


def read_pdf(path: Path) -> list[dict]:
    """One record per page, carrying the citation metadata."""
    reader = PdfReader(str(path))
    game = path.stem.replace("-", " ").replace("_", " ")
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # a cover or an all-art page has nothing to retrieve
        pages.append({"text": text, "source": path.name, "page": number, "game": game})
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the vector index from rulebook PDFs.")
    parser.add_argument("--strategy", default="paragraph", choices=["paragraph", "fixed"])
    parser.add_argument("--size", type=int, default=config.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=config.CHUNK_OVERLAP)
    parser.add_argument("--out", type=Path, default=config.INDEX_PATH)
    args = parser.parse_args()

    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {config.RAW_DIR}. Put your rulebooks there first.")

    pages: list[dict] = []
    for pdf in pdfs:
        got = read_pdf(pdf)
        print(f"  {pdf.name}: {len(got)} pages with text")
        pages.extend(got)

    records = chunk_pages(pages, size=args.size, overlap=args.overlap, strategy=args.strategy)
    print(f"-> {len(records)} chunks; embedding with {config.EMBED_MODEL} ...")

    vectors = embed([r["text"] for r in records])
    for record, vector in zip(records, vectors):
        record["embedding"] = vector

    store = VectorStore(
        records,
        meta={
            "embed_model": config.EMBED_MODEL,
            "strategy": args.strategy,
            "size": args.size,
            "overlap": args.overlap,
            "sources": [p.name for p in pdfs],
        },
    )
    store.save(args.out)
    print(f"-> wrote {len(store)} chunks to {args.out}")


if __name__ == "__main__":
    main()
