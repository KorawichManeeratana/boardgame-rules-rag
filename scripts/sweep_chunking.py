"""Measure what chunking actually costs you. One variable at a time.

    uv run --env-file .env python scripts/sweep_chunking.py
    uv run --env-file .env python scripts/sweep_chunking.py --k 4 --out eval/CHUNKING.md

For each (strategy, size, overlap) this re-chunks the corpus, re-embeds it,
and scores retrieval against the SAME labelled questions -- so the only thing
that changes between rows is the chunking.

Two things keep the comparison honest:

  * Gold answers are matched by text, not by chunk id. Ids are meaningless
    across chunkings (see eval/align.py), and comparing them would silently
    score every row against the wrong passages.
  * Query vectors are embedded once and reused for every row, so a row cannot
    win or lose on embedding jitter.

Two things keep it finishable. Embeddings are cached on disk, so a run that
dies halfway resumes instead of starting over, and a config that fails
outright is reported and skipped rather than taking the whole sweep with it.
The table is rewritten after every row, so partial results survive a crash.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.align import evaluate  # noqa: E402
from rag import config  # noqa: E402
from rag.chunking import chunk_pages  # noqa: E402
from rag.embed_cache import embed_cached, load as load_cache  # noqa: E402
from rag.embeddings import cosine  # noqa: E402
from rag.store import VectorStore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ingest import read_pdf  # noqa: E402

# The first row is the naive default we started from; everything below is the
# hypothesis being tested. Keep the baseline first so the table reads as
# before -> after.
CONFIGS = [
    ("paragraph", 300, 45),   # current default
    ("fixed", 300, 0),        # the naive split: no overlap, no boundaries
    ("paragraph", 300, 0),    # does overlap alone matter?
    ("paragraph", 450, 90),   # bigger chunks: fewer lists cut in half
    ("paragraph", 600, 120),  # bigger still -- expect precision to fall
    ("paragraph", 200, 40),   # smaller: sharper vectors, more fragments
]


def load_gold(index_path: Path, eval_path: Path) -> tuple[list[str], list[list[str]], list[str]]:
    """Resolve the eval set's chunk ids into the TEXT they pointed at.

    The ids are read from whichever index produced them, which is why the
    current index must still be the one the labelling ran against.
    """
    store = VectorStore.load(index_path)
    by_id = {r["id"]: r["text"] for r in store.records}
    rows = json.loads(eval_path.read_text(encoding="utf-8"))

    questions: list[str] = []
    gold_texts: list[list[str]] = []
    unanswerable: list[str] = []
    missing: set[str] = set()

    for row in rows:
        if not row.get("answerable", True):
            unanswerable.append(row["q"])
            continue
        texts = []
        for cid in row["relevant"]:
            if cid in by_id:
                texts.append(by_id[cid])
            else:
                missing.add(cid)
        if texts:
            questions.append(row["q"])
            gold_texts.append(texts)

    if missing:
        print(f"WARNING: {len(missing)} gold id(s) are not in {index_path.name}: "
              f"{', '.join(sorted(missing))}")
        print("         The index was rebuilt since labelling; those questions "
              "are excluded.\n")

    return questions, gold_texts, unanswerable


def rank_texts(records: list[dict], query_vecs: list[list[float]], k: int) -> list[list[str]]:
    """Top-k chunk TEXTS per query, best first."""
    out = []
    for qv in query_vecs:
        scored = [(cosine(qv, r["embedding"]), r["text"]) for r in records]
        scored.sort(key=lambda p: -p[0])
        out.append([text for _, text in scored[:k]])
    return out


def top1_scores(records: list[dict], query_vecs: list[list[float]]) -> list[float]:
    return [max(cosine(qv, r["embedding"]) for r in records) for qv in query_vecs]


def build(pages: list[dict], strategy: str, size: int, overlap: int,
          cache_path: Path, cache: dict) -> list[dict]:
    records = chunk_pages(pages, size=size, overlap=overlap, strategy=strategy)
    vectors = embed_cached([r["text"] for r in records], cache_path, cache)
    for record, vector in zip(records, vectors):
        record["embedding"] = vector
    return records


def render(rows: list[dict]) -> str:
    keys = [k for k in rows[0] if k != "config"]
    header = "| config | " + " | ".join(keys) + " |"
    divider = "|---" * (len(keys) + 1) + "|"
    lines = [header, divider]
    for row in rows:
        cells = [
            f"{row[key]}" if isinstance(row[key], int) else f"{row[key]:.3f}"
            for key in keys
        ]
        lines.append(f"| {row['config']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_report(path: Path, table: str, verdict: str, n_answerable: int,
                 n_unanswerable: int, k: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"# Chunking sweep (k={k})\n\n"
        f"{n_answerable} answerable questions, {n_unanswerable} unanswerable. "
        f"Gold answers matched by text containment, not chunk id "
        f"(see `eval/align.py`), because ids are not comparable across "
        f"chunkings.\n\n{table}\n{verdict}\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep chunking settings against the eval set.")
    parser.add_argument("--k", type=int, default=config.TOP_K)
    parser.add_argument("--out", type=Path, help="write a markdown report here")
    args = parser.parse_args()

    questions, gold_texts, unanswerable = load_gold(config.INDEX_PATH, config.EVAL_PATH)
    if not questions:
        raise SystemExit("No usable answerable questions. Re-run scripts/label.py.")

    pdfs = sorted(config.RAW_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {config.RAW_DIR}.")
    pages = [page for pdf in pdfs for page in read_pdf(pdf)]

    print(f"{len(pages)} pages, {len(questions)} answerable questions, "
          f"{len(unanswerable)} unanswerable\n")

    cache_path = config.ROOT / "data" / "index" / "embed_cache.jsonl"
    cache = load_cache(cache_path)
    if cache:
        print(f"embedding cache: {len(cache)} vector(s) already on disk\n")

    # Embedded once, reused by every config -- the queries are not the variable.
    query_vecs = embed_cached(questions, cache_path, cache)
    unanswerable_vecs = embed_cached(unanswerable, cache_path, cache) if unanswerable else []

    rows: list[dict] = []
    failed: list[str] = []
    recall_key = f"recall@{args.k}"

    for strategy, size, overlap in CONFIGS:
        label = f"{strategy} size={size} overlap={overlap}"
        print(f"  building {label} ...", flush=True)
        try:
            records = build(pages, strategy, size, overlap, cache_path, cache)
        except Exception as exc:
            # One unreachable config should not cost the rows already measured.
            print(f"    FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            failed.append(label)
            continue

        ranked = rank_texts(records, query_vecs, args.k)
        metrics = evaluate(list(zip(ranked, gold_texts)), args.k)

        # Where could the floor sit for this chunking?
        row = {
            "config": label,
            "chunks": len(records),
            **{key: value for key, value in metrics.items() if key != "n"},
            "min answerable top-1": min(top1_scores(records, query_vecs)),
        }
        if unanswerable_vecs:
            row["max unanswerable top-1"] = max(top1_scores(records, unanswerable_vecs))
        rows.append(row)
        print(f"    {recall_key}={row[recall_key]:.3f}  MRR={row['MRR']:.3f}  "
              f"chunks={row['chunks']}")

        # Rewritten every row, so a later crash keeps what is already measured.
        if args.out:
            write_report(args.out, render(rows), "", len(questions),
                         len(unanswerable), args.k)

    if not rows:
        raise SystemExit("Every config failed. Check the proxy and try again.")

    table = render(rows)
    print("\n" + table)

    # The baseline is the first entry in CONFIGS, not merely the first row
    # that survived -- otherwise a failed baseline silently promotes the next
    # config and every delta below is measured against the wrong thing.
    strategy, size, overlap = CONFIGS[0]
    baseline_label = f"{strategy} size={size} overlap={overlap}"
    baseline = next((r for r in rows if r["config"] == baseline_label), None)
    best = max(rows, key=lambda r: r[recall_key])

    if baseline is None:
        verdict = (
            f"\nBaseline ({baseline_label}) was not measured, so there is no "
            f"before/after to report. Re-run to fill it in.\n"
            f"Best so far ({best['config']}): {recall_key} = {best[recall_key]:.3f}"
        )
    else:
        delta = best[recall_key] - baseline[recall_key]
        verdict = (
            f"\nBaseline ({baseline['config']}): {recall_key} = {baseline[recall_key]:.3f}\n"
            f"Best     ({best['config']}): {recall_key} = {best[recall_key]:.3f}  "
            f"({delta:+.3f})"
        )
        if abs(delta) < 0.02:
            verdict += ("\nThat gap is within noise on a set this size -- report it as "
                        "'no measurable difference', not as an improvement.")
    if failed:
        verdict += f"\n\nNot measured (the proxy failed): {', '.join(failed)}"
    print(verdict)

    if args.out:
        write_report(args.out, table, verdict, len(questions), len(unanswerable), args.k)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()