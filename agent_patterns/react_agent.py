"""
Exercise — Core Agent Patterns (§3)
====================================

Build a minimal, framework-free agent that layers all four patterns:

    Planning    →  breaks a user goal into sub-tasks
    ReAct       →  Thought / Action / Observation loop per step
    Tool use    →  calls real (mock) functions to ground actions in data
    Reflection  →  validates the final answer before returning it

Run from the project root:
    python -m agent_patterns.react_agent

Learning goals:
    - See that "an agent" is just a loop + a prompt + function dispatch
    - Understand where each pattern lives in the code
    - Trace how planning output feeds the ReAct loop, and how reflection
      wraps the whole thing

The four patterns are clearly labelled in the code with # ── Pattern ── markers.
"""

import json
import re

from google import genai
from google.genai import types
from shared.config import SETTINGS

# ─────────────────────────────────────────────────────────────────────────────
# TOOLS  (deterministic functions — no LLM involvement)
# ─────────────────────────────────────────────────────────────────────────────

PRODUCT_DB = {
    "laptop-pro":   {"name": "Laptop Pro",   "price": 1200, "stock": 5,  "rating": 4.7},
    "laptop-air":   {"name": "Laptop Air",   "price": 850,  "stock": 12, "rating": 4.5},
    "tablet-x":     {"name": "Tablet X",     "price": 499,  "stock": 0,  "rating": 4.2},
    "keyboard-k1":  {"name": "Keyboard K1",  "price": 79,   "stock": 30, "rating": 4.8},
}


def search_products(query: str) -> list[dict]:
    """Return products whose name contains the query (case-insensitive)."""
    q = query.lower()
    return [p for p in PRODUCT_DB.values() if q in p["name"].lower()]


def get_product_details(product_id: str) -> dict:
    """Return full details for a product ID, or an error dict."""
    if product_id in PRODUCT_DB:
        return PRODUCT_DB[product_id]
    return {"error": f"Product '{product_id}' not found. "
                     f"Valid IDs: {', '.join(PRODUCT_DB)}"}


def calculate(expression: str) -> str:
    """Safely evaluate a simple arithmetic expression and return the result."""
    # Only allow digits, operators, spaces, and dots — no builtins, no imports
    if not re.fullmatch(r"[\d\s\+\-\*\/\.\(\)]+", expression):
        return f"Error: unsafe expression '{expression}'"
    try:
        result = eval(expression, {"__builtins__": {}})  # noqa: S307
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def check_stock(product_id: str) -> dict:
    """Return stock availability for a product."""
    p = PRODUCT_DB.get(product_id)
    if p is None:
        return {"error": f"Unknown product ID '{product_id}'"}
    return {"product": p["name"], "in_stock": p["stock"] > 0, "quantity": p["stock"]}


# Tool registry — maps name → function + description
TOOLS = {
    "search_products": {
        "fn": search_products,
        "description": (
            "Search for products by name keyword. "
            "Input: {query: string}. "
            "Returns a list of matching products with price, stock, and rating."
        ),
    },
    "get_product_details": {
        "fn": get_product_details,
        "description": (
            "Get full details for a specific product. "
            "Input: {product_id: string}. "
            "Valid IDs: laptop-pro, laptop-air, tablet-x, keyboard-k1."
        ),
    },
    "calculate": {
        "fn": calculate,
        "description": (
            "Evaluate a simple arithmetic expression. "
            "Input: {expression: string}. Example: '1200 * 0.9' → '1080.0'."
        ),
    },
    "check_stock": {
        "fn": check_stock,
        "description": (
            "Check if a product is in stock. "
            "Input: {product_id: string}. Returns in_stock boolean and quantity."
        ),
    },
}

# Build a human-readable tool list to include in prompts
TOOL_DESCRIPTIONS = "\n".join(
    f"  - {name}: {info['description']}" for name, info in TOOLS.items()
)


# ─────────────────────────────────────────────────────────────────────────────
# TOOL DISPATCHER
# ─────────────────────────────────────────────────────────────────────────────

def call_tool(name: str, args: dict) -> str:
    """Execute a named tool with the given args and return a string observation."""
    if name not in TOOLS:
        return f"Error: unknown tool '{name}'. Available: {', '.join(TOOLS)}"
    try:
        result = TOOLS[name]["fn"](**args)
        return json.dumps(result, indent=2)
    except TypeError as e:
        return f"Error calling {name}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# LLM HELPER
# ─────────────────────────────────────────────────────────────────────────────

def llm(client: genai.Client, system: str, messages: list[dict]) -> str:
    """Single LLM call. messages = [{role, content}, ...]"""
    contents = [
        types.Content(role=m["role"], parts=[types.Part(text=m["content"])]) for m in messages
    ]
    resp = client.models.generate_content(
        model=SETTINGS.model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system,
            temperature=0.2,
            max_output_tokens=1024,
        ),
    )
    return resp.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# ── PATTERN 1: PLANNING ───────────────────────────────────────────────────────
# Ask the model to decompose the goal into ordered sub-tasks before acting.
# ─────────────────────────────────────────────────────────────────────────────

PLANNER_SYSTEM = f"""You are a planning agent. Given a user goal, break it into
ordered sub-tasks. Each sub-task should be achievable with one or two tool calls.

Available tools:
{TOOL_DESCRIPTIONS}

Return ONLY a JSON array of step strings. Example:
["Search for laptops", "Get details for the cheapest one", "Check its stock"]

No explanation. No markdown. Just the JSON array."""


def plan(client: genai.Client, goal: str) -> list[str]:
    """Break the goal into a list of sub-task strings."""
    response = llm(client, PLANNER_SYSTEM, [{"role": "user", "content": goal}])
    # Strip markdown fences if the model adds them
    cleaned = re.sub(r"```[a-z]*\n?", "", response).strip("` \n")
    try:
        steps = json.loads(cleaned)
        if isinstance(steps, list):
            return steps
    except json.JSONDecodeError:
        pass
    # Fallback: treat each line as a step
    return [line.strip("- ").strip() for line in response.splitlines() if line.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# ── PATTERN 2 & 3: ReAct + Tool Use ──────────────────────────────────────────
# Execute a single step via Thought → Action → Observation.
# The model emits JSON with {thought, action, args} or {thought, answer}.
# ─────────────────────────────────────────────────────────────────────────────

REACT_SYSTEM = f"""You are an execution agent. You complete one task at a time
using tools. At each turn you must respond with ONLY valid JSON in one of two forms:

To use a tool:
  {{"thought": "...", "action": "<tool_name>", "args": {{...}}}}

When you have enough information to answer:
  {{"thought": "...", "answer": "..."}}

Available tools:
{TOOL_DESCRIPTIONS}

Rules:
- Never make up data. Always use a tool to get real values.
- If a tool returns an error, try a different approach.
- Stop as soon as you have enough information to answer the step."""


def react_step(client: genai.Client, task: str, max_iterations: int = 6) -> str:
    """
    Run a ReAct loop for a single task step.
    Returns the step answer as a string.
    """
    history: list[dict] = [{"role": "user", "content": task}]

    for iteration in range(1, max_iterations + 1):
        raw = llm(client, REACT_SYSTEM, history)
        history.append({"role": "model", "content": raw})

        # Parse the model's JSON response
        cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            observation = f"Parse error — your response was not valid JSON: {raw[:200]}"
            history.append({"role": "user", "content": f"Observation: {observation}"})
            continue

        thought = parsed.get("thought", "")
        print(f"  [iter {iteration}] Thought: {thought}")

        # ── PATTERN 3: Tool Use ──────────────────────────────────────────────
        if "action" in parsed:
            tool_name = parsed["action"]
            tool_args = parsed.get("args", {})
            observation = call_tool(tool_name, tool_args)
            print(f"             Action: {tool_name}({tool_args})")
            print(f"             Observation: {observation[:120]}{'...' if len(observation) > 120 else ''}")
            history.append({"role": "user", "content": f"Observation: {observation}"})

        elif "answer" in parsed:
            return parsed["answer"]

    return "Max iterations reached — could not complete this step."


# ─────────────────────────────────────────────────────────────────────────────
# ── PATTERN 4: REFLECTION ────────────────────────────────────────────────────
# Validate the final assembled answer before returning it to the user.
# ─────────────────────────────────────────────────────────────────────────────

REFLECTION_SYSTEM = """You are a quality-review agent. Given:
- The original user goal
- A draft answer produced by an agent

Evaluate whether the answer:
1. Directly addresses the goal
2. Contains only facts (no speculation)
3. Is complete — nothing important is missing

Respond with ONLY valid JSON:
  {"verdict": "PASS" | "FAIL", "issues": ["..."], "improved_answer": "..."}

If verdict is PASS, improved_answer can repeat the draft unchanged.
If verdict is FAIL, improved_answer must fix the identified issues."""


def reflect(client: genai.Client, goal: str, draft: str) -> str:
    """Critique the draft answer and return the validated (possibly improved) answer."""
    prompt = f"Goal:\n{goal}\n\nDraft answer:\n{draft}"
    raw = llm(client, REFLECTION_SYSTEM, [{"role": "user", "content": prompt}])
    cleaned = re.sub(r"```[a-z]*\n?", "", raw).strip("` \n")
    try:
        parsed = json.loads(cleaned)
        verdict = parsed.get("verdict", "PASS")
        issues = parsed.get("issues", [])
        improved = parsed.get("improved_answer", draft)
        if verdict == "FAIL":
            print(f"\n  [reflection] FAIL — issues: {issues}")
            print(f"  [reflection] Improved answer produced.")
        else:
            print(f"\n  [reflection] PASS — answer looks good.")
        return improved
    except json.JSONDecodeError:
        # Reflection parse failed — return draft unchanged
        return draft


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATOR  (ties all four patterns together)
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(client: genai.Client, goal: str) -> str:
    """
    Full agent loop:
        1. Planning   — decompose goal into steps
        2. ReAct      — execute each step with tool calls
        3. Reflection — validate the assembled answer
    """
    print(f"\n{'═' * 65}")
    print(f"Goal: {goal}")
    print(f"{'═' * 65}")

    # ── 1. PLANNING ───────────────────────────────────────────────────────────
    steps = plan(client, goal)
    print(f"\n[plan] {len(steps)} steps:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    # ── 2. ReAct (per step) ───────────────────────────────────────────────────
    results: list[str] = []
    for i, step in enumerate(steps, 1):
        print(f"\n[step {i}/{len(steps)}] {step}")
        answer = react_step(client, step)
        results.append(f"Step {i} ({step}): {answer}")
        print(f"  → {answer}")

    # Assemble a draft answer from all step results
    draft = "\n".join(results)

    # ── 3. REFLECTION ─────────────────────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("[reflection] Validating final answer...")
    final = reflect(client, goal, draft)

    print(f"\n{'═' * 65}")
    print("FINAL ANSWER")
    print(f"{'═' * 65}")
    print(final)
    return final


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

DEMO_GOALS = [
    "Find the cheapest in-stock laptop and calculate 10% off its price.",
    "Which product has the highest rating, and is it available?",
]


def main() -> None:
    client = genai.Client(api_key=SETTINGS.require_api_key())
    print("Agent Patterns Exercise — Planning + ReAct + Tool Use + Reflection")
    print(f"Model: {SETTINGS.model}\n")
    print("Available demo goals:")
    for i, g in enumerate(DEMO_GOALS, 1):
        print(f"  {i}. {g}")
    print()

    goal = input("Enter a goal (or press Enter for goal 1): ").strip()
    if not goal:
        goal = DEMO_GOALS[0]

    run_agent(client, goal)


if __name__ == "__main__":
    main()
