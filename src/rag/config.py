"""Central config. Every tunable the evaluation varies lives here, not inline."""

import os
from pathlib import Path

# --- models (class proxy) ---------------------------------------------------
CHAT_MODEL = "gemma-4-E4B-it"
EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024

# Extraction / retrieval / answering must be repeatable -> temperature 0.
TEMPERATURE = 0.0

# --- retrieval defaults -----------------------------------------------------
# These are the NAIVE defaults we start from. Day 4 tunes them against the
# eval set and records a before/after number -- never change them by vibes.
CHUNK_SIZE = 300          # words per chunk
CHUNK_OVERLAP = 45        # ~15% overlap, so a rule isn't cut at a boundary
TOP_K = 4                 # 3-5 good chunks beat 20 noisy ones
MIN_SCORE = 0.45          # below this we answer "I couldn't find that"

# --- paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
INDEX_PATH = ROOT / "data" / "index" / "chunks.json"
EVAL_PATH = ROOT / "eval" / "eval_set.json"


def require_env() -> None:
    """Fail fast and loudly if the proxy isn't configured."""
    missing = [k for k in ("OPENAI_API_KEY", "OPENAI_BASE_URL") if not os.environ.get(k)]
    if missing:
        raise RuntimeError(
            f"Missing environment variable(s): {', '.join(missing)}. "
            "Copy .env.example to .env and run with `uv run --env-file .env ...`"
        )
