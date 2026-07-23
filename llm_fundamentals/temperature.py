"""
Exercise 2 — Sampling Parameters
================================

Goal:
    Run the same prompt at temperature 0, 0.7, and 1.2 — 2 times each — and
    document how output stability changes.

What to observe:
    - temperature=0  : near-deterministic — responses should be near-identical
    - temperature=0.7: moderate variation — same intent, different wording
    - temperature=1.2: high variation — may diverge significantly or lose coherence
"""

from google import genai
from google.genai import types
from shared.config import SETTINGS

PROMPT = (
    "You are a creative writer. Write a one-sentence opening line for a science fiction "
    "novel where humanity discovers that time flows differently inside black holes, "
    "and a lone astronaut must decide whether to return home to a world that has aged "
    "centuries without them."
)
TEMPERATURES = [0.0, 0.7, 1.2]
RUNS_PER_TEMP = 2


def run_at_temperature(
    gemini_client: genai.Client,
    temperature: float,
    runs: int,
) -> list[str]:
    responses = []
    for _ in range(runs):
        resp = gemini_client.models.generate_content(
            model=SETTINGS.model,
            contents=PROMPT,
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=SETTINGS.top_p,
                top_k=SETTINGS.top_k,
                max_output_tokens=SETTINGS.max_output_tokens,
                seed=SETTINGS.seed,
            ),
        )
        responses.append(resp.text.strip())
    return responses


def are_all_identical(responses: list[str]) -> bool:
    return len(set(responses)) == 1


def unique_count(responses: list[str]) -> int:
    return len(set(responses))


def main() -> None:
    gemini_client = genai.Client(api_key=SETTINGS.require_api_key())

    print(f"Prompt: {PROMPT!r}")
    print(f"Model:  {SETTINGS.model}")
    print(f"Seed:   {SETTINGS.seed}  (set GEMINI_SEED in .env for reproducibility)")
    print("=" * 65)

    for temp in TEMPERATURES:
        print(f"\n── temperature={temp} ── ({RUNS_PER_TEMP} runs)")
        responses = run_at_temperature(gemini_client, temp, RUNS_PER_TEMP)

        for i, r in enumerate(responses, 1):
            print(f"  [{i}] {r}")

        identical = are_all_identical(responses)
        unique = unique_count(responses)
        print(f"\n  Unique responses: {unique}/{RUNS_PER_TEMP}  "
              f"{'(all identical ✓)' if identical else f'({unique} distinct variants)'}")

    print("\n" + "=" * 65)
    print("Observation guide:")
    print("  temperature=0.0 → expect 1 unique response (near-deterministic)")
    print("  temperature=0.7 → expect 1–2 unique responses (moderate variation)")
    print("  temperature=1.2 → expect 1–2 unique responses (high variation)")
    print("\nKey insight: stochasticity is not a bug — it is a design property.")
    print("Agents that need determinism should use temperature=0 AND a fixed seed.")


if __name__ == "__main__":
    main()
