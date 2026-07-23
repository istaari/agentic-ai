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
python -m llm_fundamentals.temperature          # §1 — LLM fundamentals
python -m llm_fundamentals.prompt_engineering   # §2 — Prompt engineering (8 techniques)
python -m agent_patterns.react_agent            # §3 — Planning + ReAct + Tool Use + Reflection
python -m agent_memory.agent_memory             # §4 — Working, Episodic, Semantic, Procedural memory
```

```
agentic-ai/
├── requirements.txt        # dependencies
├── .env.example            # template for GEMINI_API_KEY (+ optional overrides)
├── docs/                   # the learning path + cheatsheets
├── shared/
│   └── config.py           # THE single Gemini config — imported everywhere
│
├── llm_fundamentals/       # §1 — Understand What an LLM Actually Is
│   ├── temperature.py
│   ├── context_window.py
│   └── prompt_engineering.py
├── agent_patterns/         # §3 — Core Agent Patterns
│   └── react_agent.py      # Planning → ReAct → Tool Use → Reflection
└── agent_memory/           # §4 — Agentic Memory
    └── agent_memory.py     # WorkingMemory, EpisodicMemory, SemanticMemory, ProceduralMemory
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
