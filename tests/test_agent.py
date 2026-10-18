"""The agent loop, driven by a scripted model so none of this touches the proxy.

What matters here is not that the model is clever but that the loop stops,
that a malformed tool call is survivable, and that the citations can only
come from retrieval.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag.agent import Agent, AgentResult  # noqa: E402
from rag.answer import NO_ANSWER  # noqa: E402


# --- a scripted stand-in for the model --------------------------------------

def tool_call(name="search_docs", arguments='{"query": "egg limit"}', cid="call_1"):
    return SimpleNamespace(
        id=cid,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def reply(content=None, calls=None):
    message = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class ScriptedModel:
    """Replays a list of responses and records what it was asked."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, messages, tools):
        self.calls.append({"messages": list(messages), "tools_offered": tools is not None})
        if not self.responses:
            return reply("ran out of script")
        return self.responses.pop(0)


class FakeRetriever:
    def __init__(self, *result_sets):
        self.result_sets = list(result_sets)
        self.queries: list[str] = []

    def retrieve(self, query, game=None):
        self.queries.append(query)
        return self.result_sets.pop(0) if self.result_sets else []


def hit(cid="c0012", text="each bird has an egg limit", page=8, score=0.71):
    return {"id": cid, "text": text, "source": "wingspan-rulebook.pdf",
            "page": page, "game": "wingspan", "score": score}


# --- stopping ---------------------------------------------------------------

def test_a_reply_without_tool_calls_is_the_final_answer():
    model = ScriptedModel(reply("Two eggs per nest."))
    result = Agent(FakeRetriever(), chat=model).run("how many eggs?")
    assert result.answer == "Two eggs per nest."
    assert result.stopped_because == "model answered"
    assert result.trace == []


def test_the_agent_can_skip_retrieval_entirely():
    model = ScriptedModel(reply("You're welcome!"))
    retriever = FakeRetriever()
    Agent(retriever, chat=model).run("thanks!")
    assert retriever.queries == []   # the whole point of making it a tool


def test_a_tool_call_then_an_answer():
    model = ScriptedModel(reply(calls=[tool_call()]), reply("Each bird has a limit [c0012]."))
    retriever = FakeRetriever([hit()])
    result = Agent(retriever, chat=model).run("egg limit?")
    assert retriever.queries == ["egg limit"]
    assert result.answer == "Each bird has a limit [c0012]."
    assert result.trace[0]["hits"] == ["c0012"]


def test_max_steps_forces_an_answer_instead_of_looping():
    # A model that would call the tool forever.
    model = ScriptedModel(*[reply(calls=[tool_call(arguments=f'{{"query": "q{i}"}}',
                                                   cid=f"c{i}")]) for i in range(10)])
    result = Agent(FakeRetriever([hit()], [hit()], [hit()]), max_steps=3, chat=model).run("?")
    assert result.stopped_because == "max_steps (forced answer)"
    assert len(model.calls) == 3


def test_tools_are_withdrawn_on_the_last_step():
    model = ScriptedModel(reply(calls=[tool_call()]), reply(calls=[tool_call(cid="c2")]))
    Agent(FakeRetriever([hit()], [hit()]), max_steps=2, chat=model).run("?")
    assert model.calls[0]["tools_offered"] is True
    assert model.calls[1]["tools_offered"] is False


# --- guardrails -------------------------------------------------------------

def test_an_unknown_tool_is_refused_not_run():
    agent = Agent(FakeRetriever())
    content, hits = agent.dispatch("delete_everything", "{}", set())
    assert "Unknown tool" in content
    assert hits == []


def test_malformed_json_arguments_come_back_as_a_message():
    agent = Agent(FakeRetriever())
    content, hits = agent.dispatch("search_docs", "{not json", set())
    assert "valid JSON" in content
    assert hits == []


def test_a_missing_or_empty_query_is_refused():
    agent = Agent(FakeRetriever())
    for bad in ("{}", '{"query": ""}', '{"query": 42}', '{"query": "   "}'):
        content, hits = agent.dispatch("search_docs", bad, set())
        assert "non-empty string" in content
        assert hits == []


def test_a_repeated_query_is_blocked_rather_than_re_run():
    retriever = FakeRetriever([hit()], [hit()])
    agent = Agent(retriever)
    seen: set[str] = set()
    agent.dispatch("search_docs", '{"query": "egg limit"}', seen)
    content, hits = agent.dispatch("search_docs", '{"query": "  EGG   Limit "}', seen)
    assert "already searched" in content
    assert hits == []
    assert len(retriever.queries) == 1   # the second never reached retrieval


def test_a_bad_call_does_not_end_the_run():
    model = ScriptedModel(
        reply(calls=[tool_call(name="nonsense")]),
        reply(calls=[tool_call()]),
        reply("Recovered: the limit is on the card [c0012]."),
    )
    result = Agent(FakeRetriever([hit()]), max_steps=4, chat=model).run("?")
    assert result.stopped_because == "model answered"
    assert "Recovered" in result.answer


def test_empty_retrieval_tells_the_model_to_rephrase():
    agent = Agent(FakeRetriever([]))
    content, hits = agent.dispatch("search_docs", '{"query": "capital of france"}', set())
    assert "No passages matched" in content
    assert hits == []


# --- citations --------------------------------------------------------------

def test_citations_come_only_from_retrieved_hits():
    model = ScriptedModel(reply(calls=[tool_call()]), reply("See [c0012] and [c9999]."))
    result = Agent(FakeRetriever([hit()]), chat=model).run("?")
    # The model invented c9999; the sources list cannot contain it.
    assert result.citations() == ["[c0012] wingspan-rulebook.pdf p.8  (score 0.710)"]


def test_hits_accumulate_across_searches_without_duplicates():
    model = ScriptedModel(
        reply(calls=[tool_call(arguments='{"query": "first"}')]),
        reply(calls=[tool_call(arguments='{"query": "second"}', cid="c2")]),
        reply("done"),
    )
    retriever = FakeRetriever([hit("c0012")], [hit("c0012"), hit("c0034", page=9)])
    result = Agent(retriever, max_steps=4, chat=model).run("?")
    assert sorted(h["id"] for h in result.hits) == ["c0012", "c0034"]


def test_format_omits_the_sources_block_when_nothing_was_retrieved():
    assert AgentResult("Hello!").format() == "Hello!"


# --- a model that will not stop ---------------------------------------------

def test_a_tool_call_on_the_last_step_is_ignored_and_its_prose_kept():
    # Tools are withdrawn on the last step, but a model can emit a call
    # anyway. Honouring it would mean never stopping.
    model = ScriptedModel(
        reply(calls=[tool_call()]),
        reply("Best I can say from what I found [c0012].", calls=[tool_call(cid="c2")]),
    )
    result = Agent(FakeRetriever([hit()]), max_steps=2, chat=model).run("?")
    assert result.stopped_because == "max_steps (forced answer)"
    assert result.answer.startswith("Best I can say")


def test_a_last_step_tool_call_with_no_prose_falls_back_to_i_dont_know():
    model = ScriptedModel(reply(calls=[tool_call(cid="only")]))
    result = Agent(FakeRetriever([hit()]), max_steps=1, chat=model).run("?")
    assert result.answer == NO_ANSWER
    assert result.stopped_because == "max_steps (forced answer)"


def test_zero_steps_is_a_caller_error_not_a_crash():
    result = Agent(FakeRetriever(), max_steps=0, chat=ScriptedModel()).run("?")
    assert result.stopped_because == "no steps allowed"


def test_the_agent_also_drops_sources_under_a_refusal():
    model = ScriptedModel(reply(calls=[tool_call()]), reply(NO_ANSWER))
    result = Agent(FakeRetriever([hit()]), chat=model).run("what is the capital of france?")
    assert result.format() == NO_ANSWER
    assert "Sources" not in result.format()