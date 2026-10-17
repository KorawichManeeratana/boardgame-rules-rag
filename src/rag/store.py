"""The repository layer: the only code that touches the stored index.

A JSON file + a brute-force cosine scan. That is the right call at this size:
a few thousand chunks scan in milliseconds, and an ANN index is complexity you
only buy once a linear scan actually hurts (past ~10k-100k vectors).
"""

from __future__ import annotations

import json
from pathlib import Path

from .embeddings import cosine


class VectorStore:
    def __init__(self, records: list[dict] | None = None, meta: dict | None = None):
        self.records = records or []
        self.meta = meta or {}

    # --- persistence --------------------------------------------------------
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"meta": self.meta, "records": self.records}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "VectorStore":
        if not path.exists():
            raise FileNotFoundError(
                f"No index at {path}. Build one first:  uv run --env-file .env python scripts/ingest.py"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload["records"], payload.get("meta", {}))

    # --- search -------------------------------------------------------------
    def search(self, query_vec: list[float], k: int, game: str | None = None) -> list[dict]:
        """Top-k by cosine, best first. `game` pre-filters before ranking."""
        pool = self.records
        if game:
            pool = [r for r in pool if r["game"].lower() == game.lower()]

        scored = [
            {**{key: r[key] for key in ("id", "text", "source", "page", "game")},
             "score": cosine(query_vec, r["embedding"])}
            for r in pool
        ]
        scored.sort(key=lambda r: (-r["score"], r["id"]))
        return scored[:k]

    def __len__(self) -> int:
        return len(self.records)
