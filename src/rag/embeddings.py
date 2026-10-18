"""bge-m3 embeddings through the class proxy.

Two rules this module exists to enforce:
  1. ONE model for documents and queries -- mixing models puts the vectors in
     different spaces and retrieval becomes noise.
  2. Batch the calls. resp.data[i] lines up with input[i]; never sort or
     filter the response, or the vectors stop matching their chunks.
"""

from __future__ import annotations

import math
import random
import sys
import time

from openai import OpenAI

from . import config

_client: OpenAI | None = None

MAX_ATTEMPTS = 5
BASE_DELAY = 2.0   # seconds; doubles each attempt

# Batch by total SIZE, not by item count. A fixed count makes the payload
# scale with chunk length, so the chunking sweep was quietly sending bigger
# and bigger requests until the proxy refused: 600-word chunks x 64 is about
# 44k tokens in one call, and that is the only config that failed. A word
# budget keeps every request the same size whatever the chunking.
MAX_WORDS_PER_REQUEST = 6000
MAX_ITEMS_PER_REQUEST = 64


def client() -> OpenAI:
    global _client
    if _client is None:
        config.require_env()
        _client = OpenAI()  # reads OPENAI_API_KEY / OPENAI_BASE_URL
    return _client


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """One API call, retried on transient failures.

    The class proxy intermittently returns 500 / connection errors under
    load. A long ingest or an evaluation sweep makes hundreds of calls, so
    without a retry a single blip throws away all the work before it.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            resp = client().embeddings.create(model=config.EMBED_MODEL, input=batch)
            break
        except Exception as exc:
            if attempt == MAX_ATTEMPTS:
                raise
            # Jitter so parallel runs do not retry in lockstep.
            delay = BASE_DELAY * (2 ** (attempt - 1)) * (1 + random.random() * 0.1)
            print(f"    embedding call failed ({type(exc).__name__}); "
                  f"retry {attempt}/{MAX_ATTEMPTS - 1} in {delay:.0f}s",
                  file=sys.stderr, flush=True)
            time.sleep(delay)

    vectors = []
    for i, _ in enumerate(batch):
        vec = resp.data[i].embedding  # data[i] <-> batch[i]
        if len(vec) != config.EMBED_DIM:
            raise ValueError(f"expected {config.EMBED_DIM}-dim vector, got {len(vec)}")
        vectors.append(vec)
    return vectors


def batches(texts: list[str],
            max_words: int = MAX_WORDS_PER_REQUEST,
            max_items: int = MAX_ITEMS_PER_REQUEST) -> list[list[str]]:
    """Split into requests small enough for the proxy, by words and by count.

    A single text over the budget still gets its own request -- better to let
    the server reject that one item than to silently drop it here.
    """
    out: list[list[str]] = []
    current: list[str] = []
    words = 0
    for text in texts:
        n = len(text.split())
        if current and (words + n > max_words or len(current) >= max_items):
            out.append(current)
            current, words = [], 0
        current.append(text)
        words += n
    if current:
        out.append(current)
    return out


def _embed_resilient(batch: list[str]) -> list[list[float]]:
    """Embed one batch; on persistent failure, halve it and try again.

    Retrying the same payload cannot help when the payload itself is what
    the server rejects, so the last resort is a smaller one.
    """
    try:
        return _embed_batch(batch)
    except Exception:
        if len(batch) == 1:
            raise
        mid = len(batch) // 2
        print(f"    splitting a {len(batch)}-item batch and retrying",
              file=sys.stderr, flush=True)
        return _embed_resilient(batch[:mid]) + _embed_resilient(batch[mid:])


def embed(texts: list[str], batch_size: int | None = None) -> list[list[float]]:
    """Embed a list of texts, preserving order.

    batch_size is accepted for compatibility but ignored: batching is by
    payload size now, which is what the proxy actually cares about.
    """
    vectors: list[list[float]] = []
    for batch in batches(texts):
        vectors.extend(_embed_resilient(batch))
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