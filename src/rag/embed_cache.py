"""A disk cache for embeddings, so a crashed sweep resumes instead of restarting.

Re-chunking the corpus six ways means embedding it six times. The proxy is
not always reliable for that long, and without a cache every failure throws
away the rows that already succeeded.

Keyed on (model, text), so it is safe across runs and across configs: two
chunkings that happen to produce an identical passage pay for it once.

Append-only JSONL. Appending costs nothing per batch, a truncated last line
from an interrupted write is skipped on load, and the file is derived data --
delete it any time, it rebuilds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import config
from .embeddings import embed


def key(text: str, model: str) -> str:
    return hashlib.sha256(f"{model}\x00{text}".encode()).hexdigest()


def load(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}
    cache: dict[str, list[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue  # a half-written final line from an interrupted run
        cache[row["k"]] = row["v"]
    return cache


def append(path: Path, entries: dict[str, list[float]]) -> None:
    if not entries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for k, v in entries.items():
            fh.write(json.dumps({"k": k, "v": v}) + "\n")


def embed_cached(
    texts: list[str],
    path: Path | None = None,
    cache: dict[str, list[float]] | None = None,
) -> list[list[float]]:
    """embed(), but only pay for texts this cache has never seen.

    Pass `cache` to reuse one dict across many calls; otherwise it is read
    from disk each time.
    """
    path = path or config.ROOT / "data" / "index" / "embed_cache.jsonl"
    owned = cache is None
    if owned:
        cache = load(path)

    model = config.EMBED_MODEL
    keys = [key(t, model) for t in texts]

    # Deduplicate within this call too -- repeated text costs one call.
    missing: list[str] = []
    seen: set[str] = set()
    for text, k in zip(texts, keys):
        if k not in cache and k not in seen:
            missing.append(text)
            seen.add(k)

    if missing:
        fresh = dict(zip([key(t, model) for t in missing], embed(missing)))
        cache.update(fresh)
        append(path, fresh)

    return [cache[k] for k in keys]