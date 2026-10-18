"""Retrieval behaviour that must hold without calling the model or the proxy."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag import retriever as retriever_mod  # noqa: E402
from rag.answer import NO_ANSWER, Answer, assemble, generate  # noqa: E402
from rag.retriever import Retriever, dedupe  # noqa: E402
from rag.store import VectorStore  # noqa: E402


def chunk(cid, text, vec, source="wingspan-rulebook.pdf", page=1, game="wingspan"):
    return {"id": cid, "text": text, "source": source, "page": page,
            "game": game, "embedding": vec}


FAKE = VectorStore([
    chunk("c0", "you may discard an egg to draw a bird card", [1.0, 0.0, 0.0]),
    chunk("c1", "each nest can hold a limited number of eggs", [0.0, 1.0, 0.0]),
    chunk("c2", "an egg may be discarded to draw a bird card", [0.9, 0.1, 0.0],
          source="wingspan-appendix.pdf", page=4),
    chunk("c3", "the automa never draws bonus cards", [0.0, 0.0, 1.0],
          source="wingspan-automa.pdf", page=2, game="wingspan"),
])


# --- store ------------------------------------------------------------------

def test_search_ranks_by_cosine_best_first():
    hits = FAKE.search([1.0, 0.0, 0.0], k=3)
    assert [h["id"] for h in hits] == ["c0", "c2", "c1"]
    assert hits[0]["score"] > hits[1]["score"]


def test_search_pre_filters_before_ranking():
    hits = FAKE.search([1.0, 0.0, 0.0], k=4, game="catan")
    assert hits == []


def test_search_does_not_leak_the_embedding_into_results():
    assert "embedding" not in FAKE.search([1.0, 0.0, 0.0], k=1)[0]


# --- dedupe -----------------------------------------------------------------

def test_dedupe_drops_a_near_duplicate_and_keeps_the_best_ranked():
    hits = [
        {"text": "you may discard an egg to draw a bird card", "id": "c0"},
        {"text": "you may discard an egg to draw a bird card", "id": "c2"},
        {"text": "the automa never draws bonus cards", "id": "c3"},
    ]
    assert [h["id"] for h in dedupe(hits)] == ["c0", "c3"]


# --- the min_score floor ----------------------------------------------------

def test_floor_turns_weak_matches_into_an_empty_list(monkeypatch):
    monkeypatch.setattr(retriever_mod, "embed_one", lambda q: [0.0, 0.0, 1.0])
    # c3 scores 1.0; nothing else is close. A floor above that clears the list.
    assert Retriever(FAKE, min_score=0.99).retrieve("anything") != []
    assert Retriever(FAKE, min_score=1.01).retrieve("anything") == []


def test_top_k_caps_the_number_of_hits(monkeypatch):
    monkeypatch.setattr(retriever_mod, "embed_one", lambda q: [1.0, 1.0, 1.0])
    assert len(Retriever(FAKE, top_k=2, min_score=0.0).retrieve("anything")) == 2


# --- answering --------------------------------------------------------------

def test_no_hits_short_circuits_without_calling_the_model():
    # If this touched the network the test would fail offline -- that is the point.
    answer = generate("when is the final round?", [])
    assert answer.text == NO_ANSWER
    assert answer.grounded is False


def test_citations_come_from_the_retrieved_hits():
    hits = [{"text": "...", "source": "wingspan-appendix.pdf", "page": 4, "score": 0.71}]
    assert Answer("Yes, you may. [1]", hits).citations() == [
        "[1] wingspan-appendix.pdf p.4  (score 0.710)"
    ]


def test_assemble_numbers_chunks_from_one():
    prompt = assemble("how many eggs?", [
        {"text": "first", "source": "a.pdf", "page": 1},
        {"text": "second", "source": "b.pdf", "page": 2},
    ])
    assert "[1] first" in prompt
    assert "[2] second" in prompt
    assert prompt.rstrip().endswith("Question: how many eggs?")


def test_a_refusal_carries_no_sources_even_when_chunks_cleared_the_floor():
    # Retrieval can return noise above min_score while answering nothing.
    # "I couldn't find that" above a Sources list sends the reader to check
    # a citation that was never used.
    hits = [{"text": "...", "source": "wingspan-rulebook.pdf", "page": 1, "score": 0.458}]
    refused = Answer(NO_ANSWER, hits=hits, grounded=False)
    assert refused.format() == NO_ANSWER
    assert "Sources" not in refused.format()


def test_a_real_answer_still_carries_its_sources():
    hits = [{"text": "...", "source": "wingspan-rulebook.pdf", "page": 8, "score": 0.71}]
    answered = Answer("Each bird has a limit. [1]", hits=hits, grounded=True)
    assert "Sources" in answered.format()