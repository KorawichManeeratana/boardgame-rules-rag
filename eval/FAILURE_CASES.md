# Failure cases

Observed while building and labelling, each one reproducible from this repo.
They are the evidence behind the limitations and future-work sections of the
report, and FC-5 is the failure case shown in the demo.

Chunk ids below are from whichever index was live at the time. The index was
re-chunked from 111 chunks (300/45) to 74 (450/90) on day 5, so ids before
and after that point are not the same passages — the text is what identifies
them. This is exactly why the eval set pins gold answers to text rather than
ids (`scripts/pin_eval_text.py`).

---

## FC-1 — vector retrieval misses what BM25 finds

Three separate questions, all phrased the way a player would speak rather
than the way the rulebook writes:

| question | result |
|---|---|
| "Can I play a bird in any habitat I want?" | gold not in vector top-10; BM25 found it at rank 3 |
| "Where am I allowed to put a new bird?" | same gold passage, missed again |
| "What do I do when my turn comes around?" | top-10 contained no rulebook chunk at all; best score 0.558 |

The third is the clearest: "turn" matched "Automa's Turn", "Once Between
Turns" and "turn order dial" — all the wrong sense of the word.

The gold passage for the first two begins mid-sentence ("them as any 1 food
token. D. Place the bird card...") because it was cut out of a lettered
list, so its embedding has no topic to anchor to. That links FC-1 to FC-4.

**Reading.** Dense retrieval degrades when the question and the passage
share no vocabulary. This is the measured case for hybrid search (BM25 + RRF),
not a guess — and BM25 already scores 0.400 recall@4 on its own.

---

## FC-2 — the expansion crowds out the base rulebook

"How does the first player token move between rounds?" returned 8 of 10
chunks from `wingspan-asia.pdf`, and the answer (rulebook p.5) was not among
them.

Asia contributed 38 of 111 chunks (34%) and restates much of the base game
in near-identical wording, so several similar chunks compete for the same
top-k slots and push the original out.

**Reading.** A metadata pre-filter by source — the `--game` flag generalised
to filter by rulebook — would fix this for questions that are clearly about
base rules. Not implemented.

---

## FC-3 — a fact that is in no chunk at all

Asked "What is the Duet map?", the system answered:

> "The Duet map is used in Duet Mode for exactly **2 players** [3]."

No chunk in the corpus states the player count for Duet mode. This was
confirmed by labelling the question "How many players does Duet mode
support?" and finding no gold passage for it.

The `[3]` marker also pointed at a chunk about bonus cards, not player
counts.

**Reading.** "Use ONLY the context" steers the model; it does not bind it.
The fact came from the model's weights. Note that the grounding prompt,
the min_score floor and the citation list all held in every other respect —
this one claim still got through.

---

## FC-4 — a list split across a chunk boundary

`c0105` (rulebook p.10, old index) opens "Powers on bird cards fall into 3
categories", then defines brown and pink. The third, white, is in the next
chunk, separated by unrelated bird-card text that the two-column PDF layout
interleaves into the prose.

Asked "What does a white power on a bird card mean?", retrieval ranked
`c0105` first (0.671) — the chunk that announces the three categories but
does not define the one being asked about.

**Reading.** This was the hypothesis behind the chunking sweep, and the
sweep confirmed it: 450-word chunks with 90 overlap raised recall@4 from
0.533 to 0.683. The deeper cause — `pypdf` reading a two-column layout in
layout order rather than reading order — is unfixed and would need a
column-aware extractor.

---

## FC-5 — confidently incomplete, from sources that contradict each other

**This is the demo failure case.**

Question: *"Can a bird power let me go over my egg limit?"*

Answer: *"No, no power allows you to lay more eggs on a bird than its limit
allows [c0009]."*

The corpus contains both:

- the general rule (appendix p.8): no power exceeds the egg limit
- an Asia expansion card (asia p.12): "You may go 2 over its egg limit"

The agent retrieved four chunks, all restating the general rule from p.8 of
four different files, and none of them the exception. The citation it gave
is correct for the passage it read.

**Reading.** This is worse than a hallucination and harder to catch. Nothing
in the output is false, the citation checks out, and a player following the
answer would still play the rule wrong. Retrieval that returns *some* of the
relevant passages produces answers that are wrong by omission, and citations
do not reveal it. Near-duplicate chunks make it more likely: four restatements
of one rule filled every slot the exception could have used.

### Why the exception is unfindable, and what it cost us

The exception is in the index — `c0040`, `wingspan-asia.pdf` p.12 — confirmed
by searching the chunk text directly rather than by retrieving it:

```bash
uv run python -c "import json;d=json.load(open('data/index/chunks.json'));[print(r['id'],r['source'],'p.'+str(r['page'])) for r in d['records'] if '2 over' in r['text']]"
```

Two different queries fail to surface it, including one built from the
passage's own words:

| query | top-4 |
|---|---|
| "Can a bird power let me go over my egg limit?" | c0009, c0068, c0067, c0032 |
| "go 2 over its egg limit once between turns" | rulebook p.8, asia p.8, asia p.1, appendix p.8 |

A query containing the literal phrase "2 over its egg limit" does not
retrieve the chunk containing "2 over its egg limit". `c0040` opens with
tucking and flock scoring and only reaches `ONCE BETWEEN TURNS (pink)`
part-way in, so its single embedding is dominated by the content it leads
with. One vector per 450 words averages over everything in them, and a
specific fact inside a chunk mostly about something else stops being
reachable.

**This is the cost of the chunking change.** Under the old 300/45 index the
same rule sat at the *head* of its chunk ("ONCE BETWEEN TURNS: ...You may go
2 over its egg limit"), where it was not diluted. Moving to 450/90 raised
recall@4 from 0.533 to 0.683 across the eval set and made this passage
harder to find. The sweep measured the average; it could not see this.

Honest framing for the report: the chunking improvement is real and
measured, and it is not free. A reranker over a wider shortlist, or hybrid
retrieval, would address this particular loss better than tuning chunk size
further in either direction.

---

## FC-6 — extraction noise clears the min_score floor

Asked "thanks, that helped!", the non-agent path retrieved a chunk scoring
**0.458**, above the 0.45 floor, whose text is InDesign export metadata:

```
from the supply on this bird. 85cm WS_EE_BirdSwiftStart_r4.indd 1
WS_EE_BirdSwiftStart_r4.indd 1 6/9/21 4:19 PM ...
```

It then correctly said it could not find an answer — and printed that noise
chunk as a Source underneath, which has since been fixed (a refusal now
carries no sources).

**Reading.** Two findings. The floor is too low to exclude junk, which the
abstention numbers already show from another angle. And `.indd` filenames,
timestamps and bird-card fragments are indexed as if they were rules; a
cleanup pass at ingest would remove a measurable amount of the corpus.

The agent path never hit this, because it did not search at all.

---

## Unsolved: abstention

At `min_score=0.45`, only **1 of 4** unanswerable questions correctly
returns nothing.

```
highest top-1 among unanswerable questions: 0.578
lowest  top-1 among answerable   questions: 0.540
```

The ranges **overlap**, so no single threshold separates them. Raising the
floor to 0.578 would start refusing real questions; lowering it answers
junk.

The chunking sweep found one setting where the ranges do separate
(200/40: 0.568 vs 0.558) but by 0.010 on four questions, which is too
narrow to trust and came with worse recall.

**Reading.** A score threshold is the wrong instrument for this. Reranking
the shortlist with a cross-encoder, or asking the model to judge relevance
before answering, would decide on content rather than on a cosine value.

---

## Unsolved: information that is only in pictures

Several questions were dropped during labelling because the answer exists in
the rulebook but not in any text `pypdf` can extract — it is in a diagram or
a player-mat illustration:

- "How many action cubes does each player start with?" (the per-round counts
  are printed on the mat)
- "What happens if I have no space left on my player mat?" (the five slots
  per row are shown, not stated)
- "How do I know who goes first?" (first-player setup is in the setup
  diagram)

Both vector and BM25 search failed on each, which is what distinguishes
"not in the corpus" from "retrieval missed it".

Recorded in `eval/skipped.txt`.

**Reading.** The corpus is the extractable text, not the rulebook. Any
question whose answer is drawn rather than written is out of scope, and no
retrieval improvement will reach it.
