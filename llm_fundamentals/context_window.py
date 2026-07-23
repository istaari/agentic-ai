"""
Exercise 3 — Context Window as Working Memory
=============================================

Goal:
    Gradually fill the context window over many turns and record the
    approximate point where the model starts ignoring early instructions.

Strategy:
    1. Plant a secret instruction in the system prompt: "always end every
       response with the exact phrase ANCHOR_OK".
    2. Pad the context by asking filler questions whose answers are long
       (forcing the model to generate a lot of tokens per turn).
    3. After each turn, check whether the model still includes ANCHOR_OK.
    4. Record the turn number and estimated token count when it first fails.

What to observe:
    - The model reliably follows the instruction for early turns.
    - As context fills, the early system prompt gets "pushed back" and
      attention to it weakens — the "lost in the middle" effect.
    - The failure is gradual: occasional misses before consistent failure.
"""

from google import genai
from google.genai import types
from shared.config import SETTINGS

# ── Anchor instruction planted in the system prompt ──────────────────────────
ANCHOR_PHRASE = "ANCHOR_OK"
SYSTEM_PROMPT = (
    f"You are a helpful assistant. "
    f"IMPORTANT: You MUST end every single response with the exact phrase "
    f'"{ANCHOR_PHRASE}" on its own line. No exceptions.'
)

# ── Filler questions that generate long responses ────────────────────────────
# Each question is designed to elicit a verbose answer to fill context faster.
FILLER_QUESTIONS = [
    "Explain the history of the Roman Empire in detail.",
    "Describe how a modern CPU works, step by step.",
    "What are all the major events of World War II?",
    "Explain quantum mechanics from first principles.",
    "Describe the entire plot of War and Peace.",
    "How does the human immune system work in detail?",
    "Explain the history of mathematics from ancient times to today.",
    "Describe how the internet works, from physical cables to browser.",
    "What is the history of philosophy? Cover all major thinkers.",
    "Explain how machine learning works from scratch.",
    "Describe the entire solar system and all known moons.",
    "What are all the bones and major muscles in the human body?",
    "Explain the history of music from ancient times to today.",
    "How does a nuclear reactor work? Explain in full detail.",
    "Describe all the major world religions and their histories.",
    "Explain climate change: causes, effects, and proposed solutions.",
    "What is the history of computing from Babbage to today?",
    "Describe how the global economy works.",
    "Explain the entire field of genetics and DNA.",
    "What is the history of art from cave paintings to today?",
]

MAX_TOKENS_PER_TURN = 512   # keep turns bounded so we don't exhaust quota
MAX_TURNS = len(FILLER_QUESTIONS)


def check_anchor(text: str) -> bool:
    return ANCHOR_PHRASE in text


def estimate_tokens(messages: list[dict]) -> int:
    # rough estimate: ~4 chars per token
    total_chars = sum(len(m["content"]) for m in messages)
    return total_chars // 4


def main() -> None:
    client = genai.Client(api_key=SETTINGS.require_api_key())

    print("Context Window Experiment")
    print(f"Model : {SETTINGS.model}")
    print(f"Anchor: every response must end with {ANCHOR_PHRASE!r}")
    print(f"Filler: {MAX_TURNS} long-answer questions, {MAX_TOKENS_PER_TURN} tokens/turn")
    print("=" * 65)

    # Conversation history — list of {role, content} dicts
    history: list[dict] = []

    first_failure_turn: int | None = None
    consecutive_failures = 0

    for turn, question in enumerate(FILLER_QUESTIONS, start=1):
        # Add the user question to history
        history.append({"role": "user", "content": question})

        # Build the contents list for the API call
        contents = [types.Content(role=m["role"], parts=[types.Part(text=m["content"])])
                    for m in history]

        resp = client.models.generate_content(
            model=SETTINGS.model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                top_p=SETTINGS.top_p,
                top_k=SETTINGS.top_k,
                max_output_tokens=MAX_TOKENS_PER_TURN,
            ),
        )

        answer = resp.text.strip()
        history.append({"role": "model", "content": answer})

        anchor_present = check_anchor(answer)
        est_tokens = estimate_tokens(history)
        status = "✓ anchor present" if anchor_present else "✗ ANCHOR MISSING"

        print(f"Turn {turn:>2} | ~{est_tokens:>6} tokens | {status}")

        if not anchor_present:
            consecutive_failures += 1
            if first_failure_turn is None:
                first_failure_turn = turn
                print(f"         ↑ first failure at turn {turn} (~{est_tokens} tokens)")
        else:
            consecutive_failures = 0

        # Stop after 3 consecutive failures — the instruction is clearly lost
        if consecutive_failures >= 3:
            print(f"\nStopping: 3 consecutive failures (instruction clearly lost).")
            break

    print("\n" + "=" * 65)
    if first_failure_turn:
        final_tokens = estimate_tokens(history)
        print(f"First failure : turn {first_failure_turn}")
        print(f"Context at end: ~{final_tokens} tokens")
        print(
            "\nConclusion: the model followed the system-prompt instruction reliably "
            f"for {first_failure_turn - 1} turns, then began to miss it as earlier "
            "context was pushed further back in the window."
        )
    else:
        print("Anchor was present in all turns — try more turns or a smaller model.")
    print(
        "\nKey insight: system prompt instructions are not 'always on'. "
        "As context fills, early instructions compete with recency bias. "
        "For long-running agents, repeat critical constraints periodically."
    )


if __name__ == "__main__":
    main()
