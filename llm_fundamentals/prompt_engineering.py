"""
Exercise — Prompt Engineering as a Discipline (§2)
===================================================

Runs a live demonstration of all 8 techniques from §2 of the learning path.
Each demo calls the API twice — with and without the technique — so the
difference is visible directly in the output.

Run from the project root:
    python -m llm_fundamentals.prompt_engineering
"""

from google import genai
from google.genai import types
from shared.config import SETTINGS

DIVIDER = "─" * 65


def _call(
    client: genai.Client,
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 400,
) -> str:
    resp = client.models.generate_content(
        model=SETTINGS.model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system if system else None,
            temperature=temperature,
            max_output_tokens=max_tokens,
        ),
    )
    return resp.text.strip()


# ── §2.1  Chain-of-Thought ────────────────────────────────────────────────────

def demo_chain_of_thought(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.1  CHAIN-OF-THOUGHT")
    print(f"{'═' * 65}")

    question = (
        "A store sells 3 shirts for $25 each and 2 pants for $45 each. "
        "A customer buys all of them and pays with $200. How much change do they receive?"
    )

    print(f"\nQuestion: {question}\n")
    print(DIVIDER)
    print("[ WITHOUT CoT ]")
    print(_call(client, question, temperature=0.0))

    print(f"\n{DIVIDER}")
    print("[ WITH CoT — 'Think step by step' ]")
    print(_call(client, f"Think step by step.\n\n{question}", temperature=0.0))

    print("\n→ Each intermediate token conditions the next; arithmetic is computed via text.")


# ── §2.2  Few-Shot Learning ───────────────────────────────────────────────────

def demo_few_shot(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.2  FEW-SHOT LEARNING")
    print(f"{'═' * 65}")

    ticket = "The export button does nothing when I click it three times."
    print(f"\nNew ticket: {ticket!r}\n")

    print(DIVIDER)
    print("[ ZERO-SHOT — no examples ]")
    print(_call(
        client,
        f'Classify this support ticket as BUG, FEATURE_REQUEST, or QUESTION.\n\nTicket: "{ticket}"\nLabel:',
        temperature=0.0,
    ))

    print(f"\n{DIVIDER}")
    print("[ FEW-SHOT — 3 labeled examples ]")
    few_shot_prompt = (
        'Classify each support ticket as BUG, FEATURE_REQUEST, or QUESTION.\n\n'
        'Ticket: "The export button does nothing when I click it."\n'
        'Label: BUG\n\n'
        'Ticket: "Can you add dark mode to the dashboard?"\n'
        'Label: FEATURE_REQUEST\n\n'
        'Ticket: "How do I reset my password?"\n'
        'Label: QUESTION\n\n'
        f'Ticket: "{ticket}"\n'
        'Label:'
    )
    print(_call(client, few_shot_prompt, temperature=0.0))

    print("\n→ Examples shape output format without changing weights — 2 good ones beat 5 vague ones.")


# ── §2.3  System Prompt vs. User Prompt ──────────────────────────────────────

def demo_system_vs_user_prompt(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.3  SYSTEM PROMPT vs. USER PROMPT")
    print(f"{'═' * 65}")

    task  = "Explain what TCP does."
    rules = "You are a concise technical writer. Never use bullet points. Maximum 3 sentences."
    print(f"\nTask:  {task!r}")
    print(f"Rules: {rules!r}\n")

    print(DIVIDER)
    print("[ MIXED — rules crammed into user prompt ]")
    print(_call(client, f"{rules} {task}", temperature=0.3))

    print(f"\n{DIVIDER}")
    print("[ SEPARATED — rules in system prompt, task in user prompt ]")
    print(_call(client, task, system=rules, temperature=0.3))

    print("\n→ System prompt anchors constraints before the conversation; mixing dilutes both.")


# ── §2.4  Role / Persona Prompts ─────────────────────────────────────────────

def demo_persona(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.4  ROLE / PERSONA PROMPTS")
    print(f"{'═' * 65}")

    code = (
        "def get_user(user_id):\n"
        "    query = f'SELECT * FROM users WHERE id = {user_id}'\n"
        "    return db.execute(query)"
    )
    task = f"Review this code:\n\n```python\n{code}\n```"
    print(f"\nCode:\n{code}\n")

    print(DIVIDER)
    print("[ NO PERSONA ]")
    print(_call(client, task, temperature=0.3))

    print(f"\n{DIVIDER}")
    print("[ PERSONA — senior security engineer ]")
    print(_call(
        client,
        task,
        system="You are a senior security engineer. Identify vulnerabilities precisely. Be direct.",
        temperature=0.3,
    ))

    print("\n→ Persona activates the security-review token distribution from training data.")


# ── §2.5  Instruction Decomposition ──────────────────────────────────────────

def demo_instruction_decomposition(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.5  INSTRUCTION DECOMPOSITION")
    print(f"{'═' * 65}")

    article = (
        "The Q3 product review concluded with three decisions. "
        "Alice will redesign the onboarding flow by end of month. "
        "Bob must fix the payment timeout bug before the Friday release. "
        "The team agreed to defer the dark-mode feature to Q4, with Carol leading it."
    )
    print(f"\nArticle: {article!r}\n")

    print(DIVIDER)
    print("[ ONE BIG INSTRUCTION ]")
    print(_call(
        client,
        (
            f'Read this article, extract all action items, identify the owner of each, '
            f'and return a JSON array with fields: action_item, owner, deadline.\n\n'
            f'Article: "{article}"'
        ),
        temperature=0.0,
    ))

    print(f"\n{DIVIDER}")
    print("[ DECOMPOSED — ordered sub-steps ]")
    print(_call(
        client,
        (
            f'Article: "{article}"\n\n'
            f'Step 1: List every action item mentioned, verbatim.\n'
            f'Step 2: For each, identify the owner and deadline from the text.\n'
            f'Step 3: Return ONLY a JSON array. Each element: '
            f'{{"action_item": "...", "owner": "...", "deadline": "..."}}\n'
            f'        Use null for any field not stated in the text.'
        ),
        temperature=0.0,
    ))

    print("\n→ Sub-steps give each goal full attention; reduces compound failures.")


# ── §2.6  Structured Output ───────────────────────────────────────────────────

def demo_structured_output(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.6  STRUCTURED OUTPUT")
    print(f"{'═' * 65}")

    review = "This keyboard is incredibly comfortable but the backlight dies after an hour."
    print(f"\nReview: {review!r}\n")

    print(DIVIDER)
    print("[ FREE-FORM — prompt instructions only ]")
    print(_call(
        client,
        f'Analyse this product review. Give a sentiment (POSITIVE/NEGATIVE/MIXED), '
        f'a score from 1–10, and a one-sentence summary.\n\nReview: "{review}"',
        temperature=0.0,
    ))

    print(f"\n{DIVIDER}")
    print("[ SCHEMA-CONSTRAINED — response_mime_type + response_schema ]")
    schema_resp = client.models.generate_content(
        model=SETTINGS.model,
        contents=f'Analyse this product review.\n\nReview: "{review}"',
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema={
                "type": "object",
                "properties": {
                    "sentiment": {"type": "string", "enum": ["POSITIVE", "NEGATIVE", "MIXED"]},
                    "score":     {"type": "integer"},
                    "summary":   {"type": "string"},
                },
                "required": ["sentiment", "score", "summary"],
            },
            temperature=0.0,
            max_output_tokens=200,
        ),
    )
    print(schema_resp.text.strip())

    print("\n→ Schema enforcement: only valid JSON tokens are eligible at each position.")


# ── §2.7  Negative Space Prompting ───────────────────────────────────────────

def demo_negative_space(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.7  NEGATIVE SPACE PROMPTING")
    print(f"{'═' * 65}")

    article = (
        "Researchers at MIT published a study suggesting that daily coffee consumption "
        "might be linked to reduced risk of type-2 diabetes, though the authors caution "
        "that causation has not been established and confounding factors remain. "
        "The study tracked 10,000 participants over five years."
    )
    print(f"\nArticle: {article!r}\n")

    print(DIVIDER)
    print("[ WITHOUT negative constraints ]")
    print(_call(client, f'Summarize this article.\n\nArticle: "{article}"', temperature=0.3))

    print(f"\n{DIVIDER}")
    print("[ WITH negative constraints ]")
    print(_call(
        client,
        (
            f'Summarize this article.\n'
            f'Return only facts directly stated in the text.\n'
            f'Do NOT include opinions, speculation, or inference beyond what is written.\n'
            f'Do NOT exceed 2 sentences.\n\n'
            f'Article: "{article}"'
        ),
        temperature=0.3,
    ))

    print("\n→ Negative constraints prune known failure modes; always pair with positive instructions.")


# ── §2.8  Prompt Versioning ───────────────────────────────────────────────────

def demo_prompt_versioning(client: genai.Client) -> None:
    print(f"\n{'═' * 65}")
    print("§2.8  PROMPT VERSIONING")
    print(f"{'═' * 65}")
    print(
        "\nVersioning is a process discipline. This demo shows the same input\n"
        "through two prompt versions — and why the diff matters.\n"
    )

    ticket = "App crashes when I upload a profile picture larger than 5MB."

    PROMPT_V1 = 'Classify this ticket: "{ticket}"'
    PROMPT_V2 = (
        'Classify this support ticket as exactly one of: BUG, FEATURE_REQUEST, QUESTION, PERFORMANCE.\n'
        'Return ONLY the label, nothing else.\n\n'
        'Ticket: "{ticket}"'
    )

    print(f"Input ticket: {ticket!r}\n")

    print(DIVIDER)
    print("[ PROMPT v1 — vague, no label set, no format constraint ]")
    v1 = _call(client, PROMPT_V1.format(ticket=ticket), temperature=0.0)
    print(f"  Output: {v1}")

    print(f"\n{DIVIDER}")
    print("[ PROMPT v2 — explicit label set, format, added PERFORMANCE class ]")
    v2 = _call(client, PROMPT_V2.format(ticket=ticket), temperature=0.0)
    print(f"  Output: {v2}")

    print(
        "\n→ Each version should be committed to git with a message explaining WHY it changed.\n"
        "  A regression suite (e.g. 20 labeled tickets) catches when v3 breaks v2's gains."
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    client = genai.Client(api_key=SETTINGS.require_api_key())

    print("Prompt Engineering — Live Demonstrations (§2)")
    print(f"Model : {SETTINGS.model}")
    print("Each demo runs the same task with and without the technique.")

    demo_chain_of_thought(client)
    demo_few_shot(client)
    demo_system_vs_user_prompt(client)
    demo_persona(client)
    demo_instruction_decomposition(client)
    demo_structured_output(client)
    demo_negative_space(client)
    demo_prompt_versioning(client)

    print(f"\n{'═' * 65}")
    print("All 8 techniques demonstrated.")
    print("Refer to §2 of docs/agentic-engineering-learning-path.md for the theory.")


if __name__ == "__main__":
    main()
