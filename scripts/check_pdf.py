"""Is this rulebook usable as a corpus? Check BEFORE committing to a game.

pypdf extracts text, not pictures. A rulebook that teaches with icons and
arrows looks fine to a human and comes out nearly empty here -- and you only
find out when retrieval returns nothing. Run this first.

    uv run python scripts/check_pdf.py                    # checks data/raw/
    uv run python scripts/check_pdf.py data/raw/catan.pdf
    uv run python scripts/check_pdf.py data/raw/*.pdf

Takes files, directories, or glob patterns. Patterns are expanded here rather
than by the shell, because Windows cmd does not expand them for us.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

from pypdf import PdfReader

# A page with fewer characters than this is mostly art, a cover, or a
# component list -- not something a rules question can be answered from.
MIN_CHARS_PER_PAGE = 200


def report(path: Path) -> bool:
    reader = PdfReader(str(path))
    lengths = [len((page.extract_text() or "").strip()) for page in reader.pages]
    usable = [n for n in lengths if n >= MIN_CHARS_PER_PAGE]

    total_pages = len(lengths)
    total_chars = sum(lengths)
    pct_usable = 100 * len(usable) / total_pages if total_pages else 0.0
    words = total_chars / 5  # rough: ~5 chars per word

    print(f"\n{path.name}")
    print(f"  pages              {total_pages}")
    print(f"  pages with text    {len(usable)}  ({pct_usable:.0f}%)")
    print(f"  extracted chars    {total_chars:,}  (~{words:,.0f} words)")
    print(f"  est. 300-w chunks  {words / 300:.0f}")

    if pct_usable < 60:
        print("  VERDICT: reject -- too much of this book is pictures, not prose.")
        return False
    if words < 1500:
        print("  VERDICT: thin -- fine as a second or third game, not on its own.")
        return True
    print("  VERDICT: good corpus material.")
    return True


def collect(args: list[str]) -> list[Path]:
    """Turn arguments into a sorted list of PDF paths.

    Handles a bare filename, a directory, and a glob pattern the shell may or
    may not have expanded for us.
    """
    if not args:
        args = ["data/raw"]

    found: list[Path] = []
    for arg in args:
        path = Path(arg)
        if path.is_dir():
            found.extend(path.glob("*.pdf"))
        elif path.is_file():
            found.append(path)
        else:
            found.extend(Path(m) for m in glob.glob(arg))

    pdfs = sorted({p for p in found if p.suffix.lower() == ".pdf"})
    return pdfs


def main() -> None:
    pdfs = collect(sys.argv[1:])
    if not pdfs:
        raise SystemExit("No PDFs found. Put the rulebooks in data/raw/ first.")

    results = [report(p) for p in pdfs]
    good = sum(results)
    print(f"\n{good}/{len(results)} rulebook(s) usable.")
    if good < len(results):
        print("Replace the rejected ones before you build the index.")


if __name__ == "__main__":
    main()
