# LLM Fundamentals — Understand What an LLM Actually Is

Exercises for [Section 1](../docs/agentic-engineering-learning-path.md#1-understand-what-an-llm-actually-is) of the learning path.

> Setup (API key, dependencies) is shared across all sections and lives at the
> [project root](../README.md). Do that once before running any exercise.

## Exercises

| File | Concept |
|---|---|
| `exercises/tokenizer.py` | §1.2 — count tokens, find surprising costs |
| `exercises/temperature.py` | §1.3 — sweep temperature, measure output stability |
| `exercises/context_window.py` | §1.5 — fill context, find where early instructions drop |
| `exercises/number_comparison.py` | §1.2 — probe `9.11 > 9.9`: tokenization vs. reasoning |

Each file is an empty stub — implement `main()`, then run it from the project root:

```bash
python -m llm_fundamentals.exercises.tokenizer
```

## Using the shared config

```python
from google import genai
from shared.config import SETTINGS

client = genai.Client(api_key=SETTINGS.require_api_key())
response = client.models.generate_content(
    model=SETTINGS.model,
    contents="Hello",
)
print(response.text)
```
