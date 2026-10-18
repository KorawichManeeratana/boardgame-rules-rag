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

# 2. ask a question -- retrieve once, then answer
uv run --env-file .env python -m rag.cli "How many cards do I draw on my turn?"

#    or let the model decide whether to search at all
uv run --env-file .env python -m rag.cli --agent --show-chunks "thanks, that helped!"

# 3. build the labelled eval set (resumable; saves after every question)
uv run --env-file .env python scripts/label.py

# 4. reproduce the evaluation numbers in the report
uv run --env-file .env python -m eval.run_eval --out eval/RESULTS.md
```

`run_eval` scores the vector retriever against a random floor and a BM25
baseline on the answerable questions, then checks the abstention path: on the
questions with no answer in the corpus, did the `min_score` floor correctly
return nothing?

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

Full numbers in `eval/RESULTS.md`; the chunking sweep behind them is in
`eval/CHUNKING.md`. On 20 labelled questions over 74 chunks, k=4:

| system | recall@4 | precision@4 | MRR |
|---|---|---|---|
| random (floor) | 0.050 | 0.025 | 0.067 |
| BM25 (keyword) | 0.400 | 0.212 | 0.254 |
| bge-m3 (vector) | **0.683** | **0.350** | **0.642** |

**The measured improvement.** The naive default — 300-word paragraph chunks
with 15% overlap — scored recall@4 0.533. Sweeping six settings against the
same questions put the peak at 450 words with 90 overlap: **0.533 → 0.683
(+0.150)**, with MRR up 0.512 → 0.642. 600-word chunks matched that recall
but lost precision (0.350 → 0.312), so 450 is a peak and not just the
largest value tried.

The gold answers are matched by text, not chunk id, so the comparison is not
quietly scoring different chunkings against the wrong passages — see
`eval/align.py` and `scripts/pin_eval_text.py`.

**Known gap.** Abstention is unsolved: at `min_score=0.45` only 1 of 4
unanswerable questions correctly returns nothing, and the score ranges for
answerable and unanswerable questions overlap, so no single floor separates
them. Reported in full rather than tuned away.