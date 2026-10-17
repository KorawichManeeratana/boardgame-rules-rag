# Board Game Rules Assistant

Grounded Q&A over board game rulebooks. Ask a rules question in plain language;
the app retrieves the relevant passages from the rulebook PDFs, answers **only**
from them, and cites the file and page. When the rulebooks don't contain the
answer, it says so instead of inventing a rule.

Built for the KMITL *Building LLM-Powered Applications* capstone. It uses the
class proxy (`gemma-4-E4B-it` for chat, `bge-m3` for embeddings).

## Why this corpus

Rules disputes have a right answer printed on a specific page, which makes the
output checkable: a wrong citation is visibly wrong. That is also what makes a
labelled evaluation set cheap to build and honest to report.

## Capability families covered

| Family | Where |
|---|---|
| Retrieval / RAG (embed → index → retrieve → cite) | `src/rag/`, `scripts/ingest.py` |
| Retrieval evaluation (recall@k, MRR vs. a baseline) | `eval/` |
| Agent loop with tools (retrieve-or-answer + stop condition) | `src/rag/agent.py` |

## Setup

```bash
git clone <this repo> && cd boardgame-rag
cp .env.example .env          # paste your class API key
uv sync
```

Put the rulebook PDFs in `data/raw/`. See `data/raw/SOURCES.md` for which
rulebooks this project uses and where to download each one.

## Run

```bash
# 1. build the index (once, and after any chunking change)
uv run --env-file .env python scripts/ingest.py

# 2. ask a question
uv run --env-file .env python -m rag.cli "How many cards do I draw on my turn?"

# 3. reproduce the evaluation numbers in the report
uv run --env-file .env python -m eval.run_eval
```

## Tests

```bash
uv run pytest
```

## Layout

```
src/rag/          chunking, embeddings, store (repository), retriever, answer, agent
scripts/ingest.py offline indexing: PDF → pages → chunks → vectors
eval/             labelled set, metrics, the reproducible eval run
data/raw/         the rulebook PDFs (the corpus)
```

## Results

Filled in on Day 4 — see `eval/RESULTS.md`.
