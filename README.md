# Agentic Engineering — Exercises

Hands-on Python exercises accompanying the learning path in
[`docs/agentic-engineering-learning-path.md`](docs/agentic-engineering-learning-path.md).
Each section of the guide has a matching package here. All sections share one
Gemini configuration in `shared/config.py`.

## Setup

```bash
python3 -m venv .venv                 # create a virtual environment
source .venv/bin/activate             # activate it
pip3 install -r requirements.txt      # install dependencies
cp .env.example .env                  # then add your GEMINI_API_KEY
```

Get a free API key at <https://aistudio.google.com/apikey>.

## Running an exercise

Run from the project root as a module (this puts the root on the import path,
so `from shared.config import ...` resolves with no setup):

```bash
python -m llm_fundamentals.exercises.tokenizer
```

## Layout

```
agentic-ai/
├── requirements.txt        # dependencies
├── .env.example            # template for GEMINI_API_KEY (+ optional overrides)
├── docs/                   # the learning path + cheatsheets
├── shared/
│   └── config.py           # THE single Gemini config — imported everywhere
│
├── llm_fundamentals/       # Section 1 — Understand What an LLM Actually Is
│   ├── README.md
│   └── exercises/
│       ├── tokenizer.py
│       ├── temperature.py
│       ├── context_window.py
│       └── number_comparison.py
└── …                       # future sections follow the same pattern
```

## The shared config

Each exercise imports `SETTINGS` and builds its own Gemini client:

```python
from google import genai
from shared.config import SETTINGS

client = genai.Client(api_key=SETTINGS.require_api_key())
resp = client.models.generate_content(
    model=SETTINGS.model,                   # from .env, default gemini-2.5-flash
    contents="Hello",
)
```
