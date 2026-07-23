"""
Exercise — Agentic Memory (§4)
==============================

Build a minimal, framework-free agent that uses all four memory types:

    In-Context (Working) Memory  →  buffer of recent messages (what am I doing now?)
    Episodic Memory              →  log of past tasks + outcomes (what have I done?)
    Semantic Memory              →  key-value fact store (what do I know?)
    Procedural Memory            →  registry of named routines (how do I do things?)

Run from the project root:
    python -m agent_memory.agent_memory

Learning goals:
    - See that "memory" is just data structures + prompts
    - Understand what each memory type stores and when the agent reads/writes it
    - Trace how context window size limits working memory and drives the other types
"""

import json
from datetime import datetime

from google import genai
from google.genai import types
from shared.config import SETTINGS


# ─────────────────────────────────────────────────────────────────────────────
# ── MEMORY TYPE 1: IN-CONTEXT (WORKING) MEMORY ───────────────────────────────
# A capped message buffer — the agent's short-term attention span.
# Everything here is in the LLM's context window on every call.
# When it fills up, oldest messages are dropped (sliding window).
# ─────────────────────────────────────────────────────────────────────────────

class WorkingMemory:
    """
    Capped list of recent messages for the current session.

    Why a cap?  The context window is finite.  Keeping only the N most recent
    messages prevents silent truncation and makes token cost predictable.
    """

    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self._messages: list[dict] = []   # [{role, content}, ...]

    # TODO: implement add()
    # - Append {"role": role, "content": content} to self._messages
    # - If len > max_messages, drop the oldest entry (index 0)
    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_messages:
            self._messages.pop(0)

    # TODO: implement get()
    # - Return a copy of self._messages (so callers cannot mutate the buffer)
    def get(self) -> list[dict]:
        return list(self._messages)

    # TODO: implement clear()
    # - Reset the buffer to an empty list (new session, or context reset)
    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        return f"WorkingMemory({len(self._messages)}/{self.max_messages} messages)"


# ─────────────────────────────────────────────────────────────────────────────
# ── MEMORY TYPE 2: EPISODIC MEMORY ───────────────────────────────────────────
# A log of completed tasks with their outcomes and timestamps.
# Analogous to a diary: what happened, when, and what the result was.
# The agent consults this before starting a new task — "have I done this before?"
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Append-only log of task episodes.

    Each episode: {task, outcome, summary, timestamp}
    Searched by recency or keyword to give the agent relevant past experience.
    """

    def __init__(self):
        self._episodes: list[dict] = []

    # TODO: implement record()
    # - Append a dict: {task, outcome, summary, timestamp: datetime.now().isoformat()}
    # - outcome should be "success" or "failure"
    def record(self, task: str, outcome: str, summary: str) -> None:
        self._episodes.append({
            "task": task,
            "outcome": outcome,
            "summary": summary,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        })

    # TODO: implement search()
    # - Return all episodes where keyword appears (case-insensitive) in task or summary
    # - If keyword is empty, return the last N episodes (most recent first)
    def search(self, keyword: str = "", last_n: int = 3) -> list[dict]:
        if not keyword:
            return list(reversed(self._episodes[-last_n:]))
        kw = keyword.lower()
        return [e for e in self._episodes if kw in e["task"].lower() or kw in e["summary"].lower()]

    def all(self) -> list[dict]:
        return list(self._episodes)

    def __len__(self) -> int:
        return len(self._episodes)

    def __repr__(self) -> str:
        return f"EpisodicMemory({len(self._episodes)} episodes)"


# ─────────────────────────────────────────────────────────────────────────────
# ── MEMORY TYPE 3: SEMANTIC MEMORY ───────────────────────────────────────────
# A key-value fact store — persists knowledge that outlives a single session.
# "What is the user's preferred language?", "What is Pi?", "User prefers metric units."
# In production this is a vector DB; here we use a plain dict + substring search.
# ─────────────────────────────────────────────────────────────────────────────

class SemanticMemory:
    """
    Key-value store for persistent facts.

    Keys are short labels; values are the facts themselves.
    recall() does a simple substring search so the agent can query by topic.
    """

    def __init__(self):
        self._facts: dict[str, str] = {}

    # TODO: implement store()
    # - Save fact under key (overwrite if key already exists)
    def store(self, key: str, fact: str) -> None:
        self._facts[key] = fact

    # TODO: implement recall()
    # - Return all {key: fact} pairs where query appears in the key or the fact
    # - Case-insensitive match
    # - If query is empty, return all facts
    def recall(self, query: str = "") -> dict[str, str]:
        if not query:
            return dict(self._facts)
        q = query.lower()
        return {k: v for k, v in self._facts.items() if q in k.lower() or q in v.lower()}

    # TODO: implement forget()
    # - Remove the fact for key; do nothing if key is not present
    def forget(self, key: str) -> None:
        self._facts.pop(key, None)

    def __len__(self) -> int:
        return len(self._facts)

    def __repr__(self) -> str:
        return f"SemanticMemory({len(self._facts)} facts)"


# ─────────────────────────────────────────────────────────────────────────────
# ── MEMORY TYPE 4: PROCEDURAL MEMORY ─────────────────────────────────────────
# A registry of named routines — reusable system prompts and callable functions.
# "How do I summarise text?", "How do I translate?", "How do I classify sentiment?"
# The agent looks up a routine by name and slots it into the system prompt.
# ─────────────────────────────────────────────────────────────────────────────

class ProceduralMemory:
    """
    Named registry of routines.  Each routine has:
        - system_prompt : str   — the persona / instructions for the LLM
        - description   : str   — one-line summary shown in listings
    """

    def __init__(self):
        self._routines: dict[str, dict] = {}

    # TODO: implement register()
    # - Store {"system_prompt": system_prompt, "description": description} under name
    def register(self, name: str, system_prompt: str, description: str) -> None:
        self._routines[name] = {"system_prompt": system_prompt, "description": description}

    # TODO: implement get_prompt()
    # - Return the system_prompt for name, or None if not registered
    def get_prompt(self, name: str) -> str | None:
        routine = self._routines.get(name)
        return routine["system_prompt"] if routine else None

    # TODO: implement list_routines()
    # - Return a dict mapping name → description for all registered routines
    def list_routines(self) -> dict[str, str]:
        return {name: r["description"] for name, r in self._routines.items()}

    def __len__(self) -> int:
        return len(self._routines)

    def __repr__(self) -> str:
        return f"ProceduralMemory({len(self._routines)} routines)"


# ─────────────────────────────────────────────────────────────────────────────
# LLM HELPER
# ─────────────────────────────────────────────────────────────────────────────

def llm(client: genai.Client, system: str, messages: list[dict], max_tokens: int = 512) -> str:
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])]) for m in messages
    ]
    resp = client.models.generate_content(
        model=SETTINGS.model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.3,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# MEMORY-AWARE AGENT
# Ties all four memory types together in a single chat loop.
# ─────────────────────────────────────────────────────────────────────────────

class MemoryAgent:
    """
    A simple conversational agent that demonstrates all four memory types.

    Session flow:
        session_start()  — load episodic + semantic context into working memory
        chat(user_msg)   — respond using working memory; update memories after
        session_end()    — summarise session and persist to episodic memory
    """

    def __init__(self, client: genai.Client):
        self.client = client

        # ── Initialise the four memory types ─────────────────────────────────
        self.working   = WorkingMemory(max_messages=8)      # short-term buffer
        self.episodic  = EpisodicMemory()                   # past task log
        self.semantic  = SemanticMemory()                   # persistent facts
        self.procedural = ProceduralMemory()                # named routines

        # ── Register default routines in procedural memory ───────────────────
        # TODO: call self.procedural.register() to add at least two routines:
        #   "assistant"  — a helpful, concise general-purpose assistant
        #   "summariser" — distils long text into ≤3 bullet points
        self.procedural.register(
            "assistant",
            "You are a helpful, concise assistant. Answer clearly in 1-3 sentences.",
            "General-purpose conversational assistant",
        )
        self.procedural.register(
            "summariser",
            (
                "You are a summariser. Condense the provided text into at most 3 bullet points. "
                "Each bullet must be one sentence. Use '-' as the bullet character."
            ),
            "Distil text into ≤3 bullet points",
        )

        # ── Pre-load some facts into semantic memory ──────────────────────────
        # TODO: call self.semantic.store() to seed a few facts:
        #   "user_name"       → "Alex"
        #   "preferred_units" → "metric"
        #   "language"        → "English"
        self.semantic.store("user_name", "Alex")
        self.semantic.store("preferred_units", "metric")
        self.semantic.store("language", "English")

        self._active_routine = "assistant"   # current procedural routine name

    def session_start(self) -> None:
        """
        Prime working memory with persistent context before the first turn.

        Injects:
          1. Known facts from semantic memory → gives the agent background
          2. Recent episodes from episodic memory → reminds it what it has done
        """
        # TODO: build a context string from semantic and episodic memory and
        # add it as a "model" message so it sits at the top of working memory.
        facts = self.semantic.recall()
        episodes = self.episodic.search(last_n=3)

        lines = ["[Session context loaded from memory]"]
        if facts:
            lines.append("Known facts: " + "; ".join(f"{k}={v}" for k, v in facts.items()))
        if episodes:
            lines.append("Recent episodes:")
            for ep in episodes:
                lines.append(f"  • [{ep['outcome']}] {ep['task']}: {ep['summary']}")

        self.working.add("model", "\n".join(lines))
        print(f"  → Working memory primed ({len(self.working)} messages)")
        print(f"  → Semantic facts loaded: {len(facts)}")
        print(f"  → Episodic context loaded: {len(episodes)} recent episodes")

    def chat(self, user_message: str) -> str:
        """
        One conversation turn.

        Steps:
          1. Add user message to working memory
          2. Check if the message implies a routine switch (procedural lookup)
          3. Call LLM with current working memory + active routine's system prompt
          4. Add model reply to working memory
          5. Extract any new facts mentioned and store in semantic memory
        """
        # TODO: step 1 — add user_message to self.working (role "user")
        self.working.add("user", user_message)

        # TODO: step 2 — check for routine switch keywords
        # If "summarise" or "summary" is in the message, switch to "summariser"
        # Otherwise use "assistant"
        if any(kw in user_message.lower() for kw in ("summarise", "summarize", "summary")):
            self._active_routine = "summariser"
        else:
            self._active_routine = "assistant"

        # TODO: step 3 — get system prompt from procedural memory and call LLM
        system_prompt = self.procedural.get_prompt(self._active_routine) or ""
        reply = llm(self.client, system_prompt, self.working.get())

        # TODO: step 4 — add reply to working memory (role "model")
        self.working.add("model", reply)

        # TODO: step 5 — if the reply mentions the user's name or a preference,
        # store it. Here we do a simple demo: store the reply's first 80 chars
        # under a timestamped key so semantic memory grows over the session.
        snippet_key = f"turn_{len(self.working)}"
        self.semantic.store(snippet_key, reply[:80])

        return reply

    def session_end(self, task_description: str, outcome: str = "success") -> None:
        """
        Summarise the session and write an episode to episodic memory.

        In a real agent this would call the LLM to produce the summary;
        here we use the last model message as a proxy to keep the demo simple.
        """
        # TODO: get the last model message from working memory as the summary
        history = self.working.get()
        model_messages = [m["content"] for m in history if m["role"] == "model"]
        summary = model_messages[-1][:120] if model_messages else "No summary available."

        # TODO: record the episode
        self.episodic.record(task_description, outcome, summary)
        print(f"\n  → Episode recorded: [{outcome}] {task_description[:60]}")

        # TODO: clear working memory — session is over
        self.working.clear()
        print("  → Working memory cleared for next session.")


# ─────────────────────────────────────────────────────────────────────────────
# DEMO
# ─────────────────────────────────────────────────────────────────────────────

DEMO_TURNS = [
    "Hi! What's my name and what language do I prefer?",
    "What are the four types of agentic memory? Give me a brief overview.",
    "Summarise your previous answer about agentic memory.",
    "What units system do I use?",
]

DIVIDER = "─" * 65


def print_memory_state(agent: MemoryAgent) -> None:
    print(f"\n  {agent.working}")
    print(f"  {agent.episodic}")
    print(f"  {agent.semantic}")
    print(f"  {agent.procedural}")
    routines = agent.procedural.list_routines()
    for name, desc in routines.items():
        marker = "▶" if name == agent._active_routine else " "
        print(f"    {marker} [{name}] {desc}")


def main() -> None:
    client = genai.Client(api_key=SETTINGS.require_api_key())
    print("Agentic Memory Exercise (§4)")
    print(f"Model: {SETTINGS.model}")
    print(DIVIDER)

    agent = MemoryAgent(client)

    # ── Session 1 ─────────────────────────────────────────────────────────────
    print("\n[SESSION 1 START]")
    agent.session_start()
    print_memory_state(agent)

    for turn in DEMO_TURNS:
        print(f"\n{DIVIDER}")
        print(f"User: {turn}")
        reply = agent.chat(turn)
        print(f"Agent [{agent._active_routine}]: {reply}")
        print(f"\n  Active routine: {agent._active_routine}")

    agent.session_end("Demonstrated all four memory types to the user", outcome="success")
    print_memory_state(agent)

    # ── Session 2 — show episodic recall ─────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("[SESSION 2 START — new session, episodic memory persists]")
    agent.session_start()   # working memory re-primed from episodic + semantic

    followup = "What did we discuss in the previous session?"
    print(f"\n{DIVIDER}")
    print(f"User: {followup}")
    reply = agent.chat(followup)
    print(f"Agent: {reply}")

    agent.session_end("Follow-up session demonstrating episodic recall", outcome="success")

    # ── Final state ────────────────────────────────────────────────────────────
    print(f"\n{DIVIDER}")
    print("FINAL MEMORY STATE")
    print(DIVIDER)
    print(f"\n  Episodic log ({len(agent.episodic)} episodes):")
    for ep in agent.episodic.all():
        print(f"    [{ep['outcome']}] {ep['timestamp']}  {ep['task'][:55]}")

    facts = agent.semantic.recall()
    print(f"\n  Semantic facts ({len(facts)} total):")
    for k, v in list(facts.items())[:8]:    # show first 8 to keep output tidy
        print(f"    {k}: {v[:60]}")

    print(f"\n  Procedural routines: {list(agent.procedural.list_routines())}")
    print(f"\n{DIVIDER}")
    print("Done. Refer to §4 of docs/agentic-engineering-learning-path.md for the theory.")


if __name__ == "__main__":
    main()
