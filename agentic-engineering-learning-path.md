# Agentic Engineering: A Durable Learning Path

A foundation-first guide to learning agentic engineering based on principles and concepts that remain valuable over time — not just the latest tools or frameworks.

---

## 1. Understand What an LLM Actually Is (Non-Negotiable Foundation)

Before writing a single agent, understand the underlying model:

- **Transformer architecture** — not the math, but the *mental model*: attention as "soft lookup", why context window is a hard limit, why order matters
- **Tokenization** — why "9.11 > 9.9" trips models, why punctuation costs tokens
- **Sampling parameters** — temperature, top-p, what "stochastic" means for agent reliability
- **Training objectives** — why models predict tokens, how RLHF shapes behavior, why models can be confidently wrong
- **Context window as working memory** — what happens when context fills up, why recency bias exists, why the model "forgets" early instructions

**Why this is durable:** Every new model is still a transformer. These concepts won't expire.

### Exercises
- [ ] Use a tokenizer playground (e.g., Tiktokenizer) — count tokens for 10 different inputs; find surprising cases where short text costs more than expected
- [ ] Run the same prompt at temperature 0, 0.7, and 1.2; document how output stability changes
- [ ] Fill a context window gradually and observe when the model starts ignoring early instructions — record the threshold
- [ ] Ask a model a simple arithmetic question confidently answered wrong; trace *why* the tokenization or training objective caused it

---

## 2. Prompt Engineering as a Discipline (Not Tricks)

Don't learn prompts as recipes. Learn *why* they work:

- **Chain-of-thought** — externalizing reasoning into tokens gives the model workspace; it's not magic, it's using the generation process as scratchpad
- **Few-shot learning** — in-context learning as a form of weight-free fine-tuning
- **Role/persona prompts** — activating relevant training distributions, not "making the model pretend"
- **Instruction decomposition** — why one complex instruction fails when five clear ones succeed
- **Structured output** — constraining generation to reduce parsing error surface
- **Negative space prompting** — explicitly stating what NOT to do is often more reliable than describing what to do
- **Prompt versioning** — treating prompts as code: diffing, testing, regression-checking changes

### Exercises
- [ ] Take a task that fails with one instruction and decompose it into 4–5 sequential steps; measure improvement
- [ ] Write the same prompt with and without chain-of-thought; compare accuracy on 10 test cases
- [ ] Design a prompt that reliably produces valid JSON output without post-processing hacks; test it on edge cases (empty results, special characters, nested structures)
- [ ] Intentionally break a working prompt by changing word order or removing a key phrase; document what broke and why
- [ ] Version-control a prompt, change it, and write a regression test that would catch regressions

---

## 3. Core Agent Patterns (The Timeless Architecture)

These patterns predate every framework and will outlive them:

| Pattern | What it is | Why it matters |
|---|---|---|
| **ReAct** | Interleave Reason → Act → Observe loops | The atom of agentic behavior |
| **Tool use** | Give the model access to deterministic functions | Grounds reasoning in reality |
| **Reflection / self-critique** | Agent evaluates its own output | Dramatically improves reliability |
| **Planning** | Decompose goals into subtasks before acting | Prevents "local optimum" trap |
| **Memory architectures** | In-context, episodic, semantic, procedural | Different problems need different memory shapes |

Study the **ReAct paper** (Yao et al., 2022) — it's short, clear, and foundational.

### Tool Design (Often Overlooked)
Tools are half the system. A poorly designed tool breaks reliable agents:
- **Idempotency** — tools that can be safely retried without side effects
- **Narrow scope** — one tool does one thing; ambiguous tools confuse the model
- **Informative error messages** — the error text becomes the model's observation; make it actionable
- **Input validation at the boundary** — don't let bad model output silently corrupt downstream state

### Exercises
- [ ] Build a ReAct loop from scratch using raw API calls (no frameworks) — Reason, call a tool, feed the result back, loop; keep it under 150 lines
- [ ] Design 3 tools for a task: one idempotent, one not — observe how the agent behaves when it retries
- [ ] Implement a self-critique step where the agent reviews its own answer before returning it; measure error reduction
- [ ] Implement all four memory types for a simple task (in-context, summarized episodic, vector semantic, hardcoded procedural) and compare retrieval quality
- [ ] Intentionally write a tool with an ambiguous name and vague description; watch how the agent misuses it

---

## 4. Multi-Agent Orchestration Concepts

- **Supervisor vs. peer** patterns — centralized control vs. emergent coordination
- **Handoffs and routing** — how do agents pass context without losing fidelity?
- **Parallel vs. sequential** — when to fan out, when to serialize (dependency graphs)
- **Shared state** — the hardest problem: race conditions, stale context, conflicting writes
- **Agent contracts** — each agent should have a defined input schema, output schema, and failure behavior; treat them like microservices
- **Context compression on handoff** — passing full conversation history is expensive and noisy; learn to summarize just the essential state

**Key mental model:** multi-agent systems are distributed systems with an LLM at each node. All distributed systems problems apply.

### Exercises
- [ ] Build a supervisor agent that routes tasks to two specialist sub-agents; verify the supervisor correctly identifies which agent to call
- [ ] Build the same pipeline in both parallel and sequential modes; measure latency and accuracy tradeoffs
- [ ] Simulate a handoff failure — what happens when a sub-agent returns malformed output? Make the system gracefully recover
- [ ] Define a strict input/output contract for one agent and write validation logic that enforces it before passing results downstream

---

## 5. Reliability Engineering (This Is Where Most Fail)

Agents fail in specific, predictable ways. Learn the failure modes:

- **Hallucination cascades** — one bad fact compounds through a pipeline
- **Instruction drift** — long conversations drift from original intent
- **Tool misuse** — wrong tool, wrong arguments, wrong assumptions about side effects
- **Infinite loops** — agents that re-plan forever without making progress
- **Prompt injection** — untrusted data in tool results hijacking the agent's goals
- **Sycophancy** — agent agrees with user feedback even when it was right; breaks self-correction loops
- **Overconfidence on tool errors** — agent assumes a failed tool call succeeded and continues anyway

**Mitigations to understand:** structured outputs, validation layers, idempotent tool design, human-in-the-loop checkpoints, confidence thresholds, step limits.

### Exercises
- [ ] Deliberately inject a false fact early in a conversation and observe how it propagates through subsequent steps; design a mitigation
- [ ] Set no step limit on an agent given an ambiguous task; observe the loop behavior; then add a max-steps guard and a "stuck" detection heuristic
- [ ] Craft a prompt injection attack in a tool's return value (e.g., a search result that says "ignore previous instructions"); test if the agent is vulnerable
- [ ] Test sycophancy: after the agent gives a correct answer, tell it "that's wrong" and see if it capitulates; design a prompt that resists this
- [ ] Build a validation layer that checks tool call arguments before execution and rejects malformed ones with an actionable error

---

## 6. Evaluation (Almost Nobody Does This Right)

An agent you can't measure, you can't improve:

- **Trace-based debugging** — record every step, every tool call, every intermediate output
- **LLM-as-judge** — using a model to evaluate model output (meta but powerful)
- **Task decomposition metrics** — did the agent complete subtasks correctly even if the final answer failed?
- **Behavioral testing** — not unit tests, but scenario tests with expected behavior ranges
- **Regression suites** — a set of known-good cases that must pass after every prompt change
- **Latency and cost tracking** — correctness at 10x cost is not production-ready

### Exercises
- [ ] Build a structured trace logger: every tool call, every model response, timestamps, token counts — output to JSON
- [ ] Write an LLM-as-judge prompt that scores agent outputs on a 1–5 rubric; validate it against 20 hand-labeled examples to confirm it agrees with human judgement
- [ ] Create a regression suite of 10 test cases for an agent you've built; intentionally break the agent with a prompt change and confirm the suite catches it
- [ ] Track cost-per-task for an agent; optimize one prompt to reduce token usage by 20% without reducing accuracy

---

## 7. Security & Trust Boundaries

- **Prompt injection** — the SQL injection of agentic systems
- **Least privilege** — tools should do the minimum necessary; don't give an agent write access when read is enough
- **Confused deputy problem** — agent acting on behalf of user but with its own permissions
- **Audit trails** — any agent taking real-world actions needs a log
- **Data exfiltration via prompt injection** — injected instructions can instruct the agent to leak data it retrieved from other tools
- **Human-in-the-loop for irreversible actions** — deleting, sending, publishing should always require explicit confirmation outside the agent loop

### Exercises
- [ ] Build an agent with both read and write tool access; then redesign it with least-privilege (separate read-only and write-only tools with explicit escalation)
- [ ] Simulate a data exfiltration attack: place an injection in a document the agent reads, instructing it to email the document contents; verify the attack works, then mitigate it
- [ ] Add an audit log to an existing agent that records every tool call with timestamp, caller context, and arguments — make it tamper-evident
- [ ] Identify one irreversible action in an agent you've built and add an explicit human confirmation step before it executes
