"""An agent that decides whether to search the rulebooks, instead of always searching.

The RAG pipeline in cli.py retrieves once per question, whatever the question
is. That wastes a retrieval on "thanks", and it cannot recover when the first
search comes back with the wrong passages. Here retrieval is a tool the model
calls -- or skips -- and can call again with better words.

The loop is observe -> think -> act, and it stops on one of three things:

  natural stop   the model replies with no tool_calls; that message IS the
                 answer. This is the normal exit.
  max_steps      a backstop. Small models do not always stop cleanly, so on
                 the last step the tools are withdrawn and the model is made
                 to answer from what it already has -- an answer beats a
                 crash or an endless loop.
  no tools left  every available search has already been tried.

Safety, in the Lecture 10 sense, is separate from stopping: the dispatcher
allow-lists tool names, validates arguments before running anything, and
treats a bad call as a message back to the model rather than an exception.
A tool result is untrusted input -- it is data the model reads, never
instructions this code follows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from . import config
from .answer import NO_ANSWER
from .embeddings import client
from .retriever import Retriever

MAX_STEPS = 4
MAX_SNIPPET = 600          # characters of chunk text handed back per hit

SYSTEM = (
    "You answer board game rules questions for players mid-game.\n"
    "\n"
    "You have a search_docs tool over the rulebooks. Use it whenever the "
    "answer depends on what the rules actually say. Do NOT use it for "
    "greetings, thanks, or questions already answered by the conversation "
    "so far -- answer those directly.\n"
    "\n"
    "If a search comes back empty or off-target, you may search once more "
    "with different words -- the rulebook's wording often differs from the "
    "player's.\n"
    "\n"
    "Answer ONLY from what search_docs returned. Do not use other knowledge "
    "about this game, however confident you feel. Cite the chunk ids you "
    "used, like [c0012].\n"
    f'If the rulebooks do not answer the question, reply exactly: "{NO_ANSWER}"'
)

# The description is the control surface: it is all the model sees when
# deciding whether to call this. It says when NOT to search as plainly as
# when to search, because over-retrieval is the failure we are fixing.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": (
                "Search the board game rulebooks for passages relevant to a "
                "rules question. Use for any question about how the game "
                "works. Do not use for greetings, thanks, or anything the "
                "conversation has already answered."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "What to look for, in the rulebook's own "
                            "vocabulary where you can guess it."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


@dataclass
class AgentResult:
    answer: str
    trace: list[dict] = field(default_factory=list)
    hits: list[dict] = field(default_factory=list)
    stopped_because: str = ""

    def citations(self) -> list[str]:
        """Built from what retrieval actually returned, not from the prose."""
        return [
            f"[{h['id']}] {h['source']} p.{h['page']}  (score {h['score']:.3f})"
            for h in self.hits
        ]

    def format(self) -> str:
        # Same rule as answer.py: a refusal carries no sources, even when
        # searches did run and returned something.
        if not self.hits or NO_ANSWER.lower() in self.answer.lower():
            return self.answer
        return self.answer + "\n\nSources:\n" + "\n".join(self.citations())


def _assistant_message(msg) -> dict:
    """The SDK's object as a plain dict, so history is JSON all the way down."""
    return {
        "role": "assistant",
        "content": msg.content or "",
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.function.name,
                    "arguments": call.function.arguments,
                },
            }
            for call in msg.tool_calls
        ],
    }


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "No passages matched. Try different words, or say you could not find it."
    return "\n\n".join(
        f"[{h['id']}] ({h['source']} p.{h['page']}) "
        f"{' '.join(h['text'].split())[:MAX_SNIPPET]}"
        for h in hits
    )


class Agent:
    def __init__(self, retriever: Retriever, max_steps: int = MAX_STEPS, chat=None):
        self.retriever = retriever
        self.max_steps = max_steps
        # Injectable so the loop can be tested without the proxy.
        self._chat = chat or self._call_model

    def _call_model(self, messages: list[dict], tools: list | None):
        kwargs = {
            "model": config.CHAT_MODEL,
            "temperature": config.TEMPERATURE,
            "messages": messages,
        }
        if tools:
            kwargs["tools"] = tools
        return client().chat.completions.create(**kwargs)

    # --- the one place a tool actually runs ---------------------------------
    def dispatch(self, name: str, raw_arguments: str, seen: set[str]) -> tuple[str, list[dict]]:
        """Run an allow-listed tool. Every rejection is a message, not a raise.

        A model that sends a bad call should be told so and given another
        turn; crashing the loop would lose the work already done.
        """
        if name != "search_docs":
            return f"Unknown tool {name!r}. The only tool is search_docs.", []

        try:
            args = json.loads(raw_arguments or "{}")
        except json.JSONDecodeError:
            return "Arguments were not valid JSON. Send {\"query\": \"...\"}.", []
        if not isinstance(args, dict):
            return "Arguments must be a JSON object: {\"query\": \"...\"}.", []

        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            return "search_docs needs a non-empty string `query`.", []

        key = " ".join(query.lower().split())
        if key in seen:
            return (
                f"You already searched for {query!r}. Use different words, "
                "or answer from what you have."
            ), []
        seen.add(key)

        hits = self.retriever.retrieve(query)
        return _format_hits(hits), hits

    # --- observe -> think -> act -------------------------------------------
    def run(self, question: str) -> AgentResult:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ]
        seen: set[str] = set()
        trace: list[dict] = []
        hits_by_id: dict[str, dict] = {}

        for step in range(1, self.max_steps + 1):
            last = step == self.max_steps
            # On the final step the tools are withdrawn, which forces the
            # model to produce an answer rather than another tool call.
            resp = self._chat(messages, None if last else TOOLS)
            msg = resp.choices[0].message
            calls = list(getattr(msg, "tool_calls", None) or [])

            # On the last step the response is final whatever it contains.
            # Tools were not offered, but a model can still emit a tool call,
            # and honouring it here would mean never stopping.
            if last or not calls:
                content = (msg.content or "").strip()
                if not content:
                    # Asked for an answer, got another tool call and no prose.
                    content = NO_ANSWER
                return AgentResult(
                    content,
                    trace,
                    list(hits_by_id.values()),
                    "max_steps (forced answer)" if last else "model answered",
                )

            messages.append(_assistant_message(msg))
            for call in calls:
                content, hits = self.dispatch(
                    call.function.name, call.function.arguments, seen
                )
                for hit in hits:
                    hits_by_id.setdefault(hit["id"], hit)
                trace.append(
                    {
                        "step": step,
                        "tool": call.function.name,
                        "arguments": call.function.arguments,
                        "hits": [h["id"] for h in hits],
                    }
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": content}
                )

        # Only reachable with max_steps <= 0, which is a caller error.
        return AgentResult(NO_ANSWER, trace, list(hits_by_id.values()), "no steps allowed")