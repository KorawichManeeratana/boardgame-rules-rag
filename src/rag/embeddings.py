"""bge-m3 embeddings through the class proxy.

Two rules this module exists to enforce:
  1. ONE model for documents and queries -- mixing models puts the vectors in
     different spaces and retrieval becomes noise.
  2. Batch the calls. resp.data[i] lines up with input[i]; never sort or
     filter the response, or the vectors stop matching their chunks.
"""

from __future__ import annotations

import math

from openai import OpenAI

from . import config

_client: OpenAI | None = None


def client() -> OpenAI:
    global _client
    if _client is None:
        config.require_env()
        _client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL
    return _client


def embed(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Embed a list of texts, preserving order."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        resp = client().embeddings.create(model=config.EMBED_MODEL, input=batch)
        for i, _ in enumerate(batch):
            vec = resp.data[i].embedding  # data[i] <-> batch[i]
            if len(vec) != config.EMBED_DIM:
                raise ValueError(f"expected {config.EMBED_DIM}-dim vector, got {len(vec)}")
            vectors.append(vec)
    return vectors


def embed_one(text: str) -> list[float]:
    return embed([text])[0]


def cosine(a: list[float], b: list[float]) -> float:
    """Angle, not distance. +1 identical direction, 0 unrelated, -1 opposite."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
