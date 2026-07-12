# Agentic Engineering: A Durable Learning Path

> A foundation-first guide built on principles that outlast any framework or tool.

---

## Table of Contents

| # | Topic |
|---|---|
| 1 | [Understand What an LLM Actually Is](#1-understand-what-an-llm-actually-is) |
| 2 | [Prompt Engineering as a Discipline](#2-prompt-engineering-as-a-discipline) |
| 3 | [Core Agent Patterns](#3-core-agent-patterns) |
| 4 | [Agentic Memory](#4-agentic-memory) |
| 5 | [RAG — Retrieval-Augmented Generation](#5-rag--retrieval-augmented-generation) |
| 6 | [MCP — Model Context Protocol](#6-mcp--model-context-protocol) |
| 7 | [Agent Communication Patterns](#7-agent-communication-patterns) |
| 8 | [Skills & Capabilities Architecture](#8-skills--capabilities-architecture) |
| 9 | [Multi-Agent Orchestration Architecture](#9-multi-agent-orchestration-architecture) |
| 10 | [Reliability Engineering](#10-reliability-engineering) |
| 11 | [Evaluation](#11-evaluation) |
| 12 | [Context Management Strategies](#12-context-management-strategies) |
| 13 | [Human-in-the-Loop (HITL) Patterns](#13-human-in-the-loop-hitl-patterns) |
| 14 | [Cost & Latency Optimization](#14-cost--latency-optimization) |
| 15 | [Security & Trust Boundaries](#15-security--trust-boundaries) |
| 16 | [LangGraph](#16-langgraph) |

---

## 1. Understand What an LLM Actually Is

> **Non-negotiable foundation.** Before writing a single agent, understand the underlying model.

| Concept | Why it matters |
|---|---|
| **Transformer architecture** | Mental model for attention, context limits, and why order matters |
| **Tokenization** | Why punctuation costs tokens; understanding token boundaries matters for prompts and cost |
| **Sampling parameters** | Temperature, top-p — what "stochastic" means for agent reliability |
| **Training objectives** | Why models predict tokens; how RLHF shapes behavior; why models can be confidently wrong |
| **Context window as working memory** | What happens when context fills; models attend better to start and end than the middle ("lost in the middle" effect) |

**Why these concepts are durable:** Every new model is still a transformer. These won't expire.

### Exercises
- [ ] Use a tokenizer playground — count tokens for 10 inputs; find cases where short text costs more than expected
- [ ] Run the same prompt at temperature `0`, `0.7`, `1.2` — document how output stability changes
- [ ] Gradually fill a context window and observe when the model starts ignoring early instructions
- [ ] Find an arithmetic question the model answers wrong confidently — trace why

---

## 2. Prompt Engineering as a Discipline

> Don't learn prompts as recipes. Learn *why* they work.

| Technique | What it does |
|---|---|
| **Chain-of-thought** | Externalizes reasoning into tokens — uses generation as scratchpad |
| **Few-shot learning** | Providing examples in-context shapes output format and style without changing model weights |
| **System prompt vs. user prompt** | System prompt sets stable behavior and identity; user prompt carries the task — mixing them degrades both |
| **Role/persona prompts** | Activates relevant training distributions |
| **Instruction decomposition** | One complex instruction fails where five clear ones succeed |
| **Structured output** | Constrains generation to reduce parsing errors |
| **Negative space prompting** | Explicitly stating what NOT to do can clarify intent, but positive instructions are generally more reliable — use both together |
| **Prompt versioning** | Treating prompts as code: diff, test, regression-check changes |

### Exercises
- [ ] Take a failing task and decompose it into 4–5 steps — measure improvement
- [ ] Write the same prompt with and without chain-of-thought — compare accuracy on 10 cases
- [ ] Design a prompt that reliably produces valid JSON — test on edge cases (empty results, special characters, nesting)
- [ ] Intentionally break a working prompt by changing word order — document what broke and why
- [ ] Version-control a prompt, change it, and write a regression test that catches the regression

---

## 3. Core Agent Patterns

> These patterns predate every framework and will outlive them.

| Pattern | What it is | Why it matters |
|---|---|---|
| **ReAct** | Reason → Act → Observe loop | The atom of agentic behavior |
| **Tool use** | Model calls deterministic functions | Grounds reasoning in reality |
| **Reflection** | Agent evaluates its own output | Improves reliability dramatically |
| **Planning** | Decompose goals before acting | Prevents "local optimum" trap |
| **Memory architectures** | In-context, episodic, semantic, procedural — covered in depth in §4 |

> Read the **ReAct paper** (Yao et al., 2022) — short, clear, foundational.

### Tool Design

Tools are half the system. A poorly designed tool breaks even a good agent.

- **Idempotency** — tools that can be safely retried without side effects
- **Narrow scope** — one tool does one thing; ambiguity confuses the model
- **Informative errors** — the error text *is* the model's observation; make it actionable
- **Input validation** — don't let bad model output silently corrupt downstream state

### Exercises
- [ ] Build a ReAct loop from scratch with raw API calls — no frameworks; keep it under 150 lines
- [ ] Design 2 tools: one idempotent, one not — observe how the agent behaves on retry
- [ ] Add a self-critique step where the agent reviews its answer before returning — measure error reduction
- [ ] Write a tool with an ambiguous name and vague description — watch how the agent misuses it

---

## 4. Agentic Memory

> Memory in agents is not one thing — it's four distinct systems. Confusing them leads to bad architecture.

| Type | Lives in | Used for | Cost |
|---|---|---|---|
| **In-context (working)** | Active prompt | Immediate state, recent observations | Token cost per request |
| **Episodic** | External DB / file | Past interaction history | Retrieval latency |
| **Semantic** | Vector database | Facts, knowledge — searched by similarity | Embedding + retrieval |
| **Procedural** | System prompt / tools | How to behave, what tools exist | Fixed at startup |

**Key concepts:**
- **Read vs. write timing** — getting this wrong creates stale or missing context
- **Memory scoping** — per-user, per-session, per-task, global; wrong scoping leaks context across users
- **Summarization strategies** — compressing episodic memory before it exceeds context limits
- **Memory as an injection surface** — stored memory can be poisoned; treat it with the same skepticism as external data
- **Forgetting as a feature** — TTL and explicit eviction are design choices, not afterthoughts

### Exercises
- [ ] Build an agent with episodic memory across 5 sessions — verify it recalls facts from session 1 in session 5
- [ ] Implement summarization that compresses history at 2000 tokens — verify no critical facts are lost
- [ ] Poison the memory store with a false fact — observe downstream effects; design a mitigation
- [ ] Build a scoping system for a multi-user agent — ensure user A's memory never leaks into user B's context

---

## 5. RAG — Retrieval-Augmented Generation

> RAG is a pattern for grounding agents in external knowledge they weren't trained on. Instead of relying on parametric memory (weights), retrieve relevant documents at query time and inject them into context.

| Concept | What to understand |
|---|---|
| **Chunking strategy** | Fixed-size vs. sentence-boundary vs. semantic vs. hierarchical — each has tradeoffs |
| **Document pre-processing** | Parsing quality (PDFs, tables, noise) before chunking determines the ceiling — garbage in, garbage out |
| **Embedding models** | Mismatch between embedding model and query style hurts recall |
| **Vector similarity search** | Cosine, dot product, ANN — don't treat retrieval as a black box |
| **Metadata filtering** | Filter by date, source, or category before/after vector search — on structured datasets this can significantly reduce irrelevant results |
| **Retrieval metrics** | Precision, recall, MRR — how to know if retrieval is actually helping |
| **Hybrid search** | Dense (vector) + sparse (BM25) often outperforms either alone |
| **Re-ranking** | Second-pass scoring of retrieved chunks before injection |
| **Context budget** | Retrieved chunks compete with instructions and history for context space |
| **RAG vs. fine-tuning** | RAG = dynamic knowledge; fine-tuning = stable behavior patterns |
| **Agentic RAG** | Agent decides *what* to retrieve, *when*, and *whether to retrieve again* — query reformulation, iterative deepening, retrieval chaining |

### Exercises
- [ ] Build a minimal RAG pipeline from scratch: chunk → embed → store → retrieve → inject — no frameworks
- [ ] Compare fixed-size vs. sentence-boundary chunking — measure precision on 10 test queries
- [ ] Implement hybrid search (BM25 + vector) — compare against pure vector on keyword-heavy queries
- [ ] Add a re-ranker that scores retrieved chunks 1–5 — measure whether final answer quality improves
- [ ] Exceed the context budget deliberately — observe degradation; implement budget-aware truncation
- [ ] Build an agentic RAG loop: retrieve → read → decide if more needed → reformulate → retrieve again

---

## 6. MCP — Model Context Protocol

> A standardized protocol for connecting agents to external tools and data. Think USB-C for agent integrations.

**Why it matters:** Without a standard, every agent-tool integration is a one-off. MCP separates tool *definition* from tool *consumption*.

| Concept | What to understand |
|---|---|
| **Server / client model** | Servers expose tools and resources; agents are clients that discover and call them |
| **Tool discovery** | `tools/list` at runtime — dynamic, not hardcoded |
| **Resources vs. tools** | Resources = data to read; tools = actions to invoke; distinction matters for authorization |
| **Prompts as first-class citizens** | Servers can expose prompt templates agents use without knowing the text |
| **Transport agnosticism** | stdio, HTTP/SSE — same protocol regardless of transport |
| **Capability negotiation** | Server and client exchange declarations on connect — version compatibility |
| **Authorization model** | Real-world actions need explicit user consent at the transport layer |

> **Durable principle:** Even if MCP is superseded, *protocol-based tool integration with dynamic discovery* will persist in whatever replaces it.

### Exercises
- [ ] Build a minimal MCP server exposing one tool — connect an agent and verify discovery works
- [ ] Expose both a resource and a tool — build an agent that uses the resource to decide which tool to call
- [ ] Add authorization: tool refuses without a valid token — test that an agent without the token is blocked
- [ ] Build a server with 5 tools; give the agent a task solvable with only 2 — verify correct selection via discovery

---

## 7. Agent Communication Patterns

> How agents talk to each other matters as much as what they individually do.

**Message passing vs. shared state:**
- **Message passing** — structured messages between agents; cleaner, auditable, easier to debug
- **Shared state** — agents read/write a common store; faster but prone to race conditions and stale reads

**Communication topologies:**

| Topology | Shape | Use when |
|---|---|---|
| **Pipeline** | A → B → C | Sequential tasks with clear dependencies |
| **Supervisor** | Hub → spokes | One coordinator, many specialists |
| **Peer-to-peer** | Any → any | Emergent, flexible — hard to reason about |
| **Blackboard** | All agents share a workspace | Tasks where agents contribute incrementally |
| **Event-driven / pub-sub** | Agents emit and subscribe to events | Async systems where agents react to state changes |

**Key concepts:**
- **Message schemas** — typed and validated, not free-form text; free-form is a reliability tax
- **Idempotent handling** — agents must handle duplicate messages safely; exactly-once delivery is impossible
- **Dead letter handling** — silent failure on unprocessable messages is the worst outcome
- **Backpressure** — slow downstream agents must not cause unbounded upstream queuing
- **Correlation IDs** — every request carries an ID that threads through all agent calls for traceability
- **Async vs. sync** — sync is simpler; async is more scalable but harder to debug; choose deliberately

### Exercises
- [ ] Build a 3-agent pipeline (A → B → C) with typed schemas — verify a malformed message from A is caught before C
- [ ] Implement the blackboard pattern: 3 agents contribute partial results; a 4th synthesizes the answer
- [ ] Add correlation IDs and build a trace view showing the full request path across all agents
- [ ] Simulate a dead letter: agent B can't process a message — implement a dead-letter queue and retry policy

---

## 8. Skills & Capabilities Architecture

> Skills are reusable, composable units of agent behavior. Designing them well is distinct from knowing how to use them.

| Concept | What to understand |
|---|---|
| **Skill as composability unit** | One skill does one thing well and combines with others — Unix philosophy for agents |
| **Skill discovery** | Static list vs. RAG over descriptions vs. protocol-based discovery (MCP) |
| **Skill vs. tool** | Tool = deterministic function; skill may contain LLM calls or a mini-agent loop |
| **Skill routing** | Description matching, classifier, explicit rules — each has tradeoffs |
| **Versioning & deprecation** | Skills evolve; agents must handle version changes gracefully |
| **Skill composition** | Chaining vs. parallel invocation vs. conditional branching |
| **Guardrails per skill** | Input/output validation and rate limits independent of the parent agent |

### Exercises
- [ ] Design a library of 6 skills for a domain — write descriptions precise enough for an LLM to route correctly
- [ ] Build a skill router: agent selects the right skill from descriptions only — test on 20 queries
- [ ] Implement skill chaining: output of skill A feeds skill B — handle the case where A fails
- [ ] Version a skill (v1 → v2, different output schema) — verify the agent handles both during migration

---

## 9. Multi-Agent Orchestration Architecture

> Section 7 covered *how* agents communicate. This section covers *how to structure the system*.

| Concept | What to understand |
|---|---|
| **Deterministic vs. LLM-driven control flow** | Code (state machine, graph) vs. LLM decides next step — most production systems use both: deterministic scaffolding with LLM reasoning inside each node |
| **Agent contracts** | Defined input schema, output schema, failure behavior — treat agents like microservices |
| **Handoffs & context compression** | Compress to essential state before handoff; passing full history is expensive and noisy |
| **Parallel vs. sequential fan-out** | Fan out when tasks are independent; serialize when steps depend on prior — think DAGs |
| **Supervisor pattern** | Coordinator routes to specialists; keeping coordinator narrow (routing-focused) improves reliability, though some designs include reasoning at the supervisor level |
| **Failure propagation** | Does one failure abort the pipeline or degrade gracefully? Define this explicitly |
| **Versioning & deployment** | Agents at different versions must have compatibility rules before you need them |

### Exercises
- [ ] Build a supervisor routing to two specialists — verify it correctly identifies which to call
- [ ] Run the same pipeline in parallel and sequential mode — measure latency and accuracy tradeoffs
- [ ] Implement a state machine governing which agent runs next — compare reliability vs. LLM-driven routing
- [ ] Simulate a handoff failure — implement graceful degradation instead of full pipeline abort
- [ ] Enforce input/output contracts at agent boundaries — measure how many runtime errors this prevents

---

## 10. Reliability Engineering

> Agents fail in specific, predictable ways. Learn the failure modes before you encounter them in production.

| Failure mode | What happens |
|---|---|
| **Hallucination cascades** | One bad fact compounds through the pipeline |
| **Instruction drift** | Long conversations drift from original intent |
| **Context window overflow** | Agent silently drops early instructions or tool results when context fills |
| **Tool misuse** | Wrong tool, wrong arguments, wrong assumptions about side effects |
| **Infinite loops** | Agent re-plans forever without making progress |
| **Prompt injection** | Untrusted tool results hijack the agent's goals |
| **Sycophancy** | Agent agrees with user feedback even when it was right — breaks self-correction |
| **Overconfidence on tool errors** | Agent assumes a failed call succeeded and continues anyway |

**Mitigations:** structured outputs, validation layers, idempotent tools, HITL checkpoints, confidence thresholds, step limits.

### Exercises
- [ ] Inject a false fact early — observe how it propagates; design a mitigation
- [ ] Remove step limits on an ambiguous task — observe the loop; then add a max-steps guard and "stuck" detection
- [ ] Craft a prompt injection in a tool's return value — test if the agent is vulnerable; then mitigate it
- [ ] Test sycophancy: give a correct answer then say "that's wrong" — see if it capitulates; fix the prompt
- [ ] Build a validation layer that rejects malformed tool call arguments with an actionable error

---

## 11. Evaluation

> An agent you can't measure, you can't improve.

| Concept | What to understand |
|---|---|
| **Trace-based debugging** | Record every step, every tool call, every intermediate output |
| **LLM-as-judge** | Using a model to evaluate model output — meta but powerful |
| **Evals vs. tests** | Tests are pass/fail (deterministic); evals accept a range of correct outputs (probabilistic) — agents need evals, not just tests |
| **Task decomposition metrics** | Did subtasks complete correctly even if the final answer failed? |
| **Behavioral testing** | Scenario tests with expected behavior ranges — not unit tests |
| **Regression suites** | Known-good cases that must pass after every prompt change |
| **Latency & cost tracking** | Correctness at 10x cost is not production-ready |

### Exercises
- [ ] Build a structured trace logger: tool calls, model responses, timestamps, token counts → JSON
- [ ] Write an LLM-as-judge prompt scoring outputs 1–5 — validate against 20 hand-labeled examples
- [ ] Create a 10-case regression suite — break the agent with a prompt change; confirm the suite catches it
- [ ] Track cost-per-task — optimize one prompt to cut token usage 20% without accuracy regression

---

## 12. Context Management Strategies

> Memory is about *persistence*. Context management is about *what's in the window right now*. They're different problems.

| Strategy | How it works | Tradeoff |
|---|---|---|
| **Sliding window** | Keep only N most recent turns | Simple; loses early context |
| **Selective retention** | Keep only "important" turns | Preserves signal; requires classification |
| **Summarization injection** | Compress older turns into a summary; inject at top | Preserves gist; loses detail |
| **System prompt separation** | Stable instructions in system prompt; dynamic state in turns | Cheaper, cached; requires discipline |
| **Context poisoning mitigation** | Periodic reset — extract key facts, wipe history, reinject clean summary | Removes noise; requires a reset trigger |
| **Needle-in-a-haystack awareness** | Place critical facts at start or end, not buried in the middle | Models attend unevenly across long contexts |

### Exercises
- [ ] Build a context manager that summarizes when token count exceeds a threshold — verify recalled facts survive
- [ ] Compare sliding window vs. selective retention on a 20-turn conversation — measure fact loss rate
- [ ] Bury a critical constraint in the middle of a long context — test adherence; move it to start/end and retest
- [ ] Implement a "context reset": extract key facts, wipe history, reinject a clean summary as new starting context

---

## 13. Human-in-the-Loop (HITL) Patterns

> Fully autonomous agents are the exception. Most production agents need structured human checkpoints.

| Pattern | What it is |
|---|---|
| **Approval gates** | Hard stop before irreversible actions — not a prompt, an actual pause |
| **Confidence thresholds** | Escalate to human when a secondary classifier or heuristic signals low reliability — note: LLM self-reported confidence is poorly calibrated and should not be trusted alone |
| **Correction loops** | Human feedback incorporated cleanly; agent continues without re-running completed steps |
| **Async HITL** | Agent checkpoints state, notifies human, resumes when approved |
| **Graceful degradation** | When human is unavailable: wait, abort, or reduce scope — never silently proceed |
| **Feedback loop integration** | HITL corrections feed back into improved prompts, evals, or training data — otherwise HITL is a safety net, not a learning mechanism |

> The decision of *what* requires human review is a product and risk decision, not a technical one. Document it explicitly per action type.

### Exercises
- [ ] Add an approval gate: before any destructive tool call, agent summarizes intent and waits for yes/no
- [ ] Implement confidence-based escalation: agent scores answer 1–10; below 6 flags for review — calibrate on 20 queries
- [ ] Build a correction loop: human feedback in natural language; agent incorporates it and continues cleanly
- [ ] Implement async HITL: checkpoint to disk, send notification, resume from checkpoint on approval

---

## 14. Cost & Latency Optimization

> An agent that works but costs 10x too much or takes 3x too long will not reach production.

| Technique | What it does |
|---|---|
| **Prompt caching** | Static prefixes (system prompts, instructions) cached by provider — structure prompts to maximize hits |
| **Model routing** | Cheap model for classification/routing; large model for complex reasoning only |
| **Batching** | Use async batch APIs (lower cost, higher latency) for non-urgent work; not the same as bundling requests in one call — most providers don't support that |
| **Streaming** | Tokens delivered as generated; perceived latency often matters more than actual |
| **Token budgets** | Explicit max-token limits per agent call; runaway generation is a cost and reliability risk |
| **Parallel execution** | Independent subtasks run concurrently; wall-clock = slowest, not sum |
| **Early termination** | Stop when intermediate results are sufficient; don't run the full pipeline unnecessarily |
| **Cost-quality profiling** | Quality gap between model tiers varies dramatically by task — measure for your use case |

### Exercises
- [ ] Profile token usage per step — find the top 2 cost drivers and optimize without accuracy regression
- [ ] Implement model routing: classify query complexity, route simple → cheap model, complex → capable model
- [ ] Restructure a system prompt: all static content at top, dynamic at bottom — verify cache hit rate improves
- [ ] Add token budget guards to each agent call — define graceful behavior when the budget is hit mid-task

---

## 15. Security & Trust Boundaries

> Agents that take real-world actions are a new attack surface. Treat them accordingly.

| Threat | What it is |
|---|---|
| **Direct prompt injection** | Malicious instructions in user input hijack agent goals |
| **Indirect prompt injection** | Malicious instructions embedded in documents, emails, or web pages the agent processes — harder to detect, more common in production |
| **Data exfiltration** | Injected instructions cause the agent to leak data retrieved from other tools |
| **Confused deputy** | Agent acts on behalf of user but with its own (broader) permissions |

**Architectural weaknesses to avoid:**
- **Overly broad permissions** — giving write access when read is enough; giving broad scope when narrow works

**Controls to implement:**
- **Structural separation** — use explicit delimiters (XML tags, JSON, clear labels) so instructions and untrusted content are architecturally distinct; stripping patterns is not reliable because injection vectors cannot be enumerated
- **Audit trails** — every tool call logged with timestamp, caller context, and arguments
- **Least privilege** — separate read-only and write tools with explicit escalation
- **HITL for irreversible actions** — delete, send, publish always require confirmation outside the agent loop

### Exercises
- [ ] Redesign an agent with least-privilege: separate read-only and write-only tools with explicit escalation
- [ ] Simulate data exfiltration: inject instructions in a document the agent reads to leak its contents — mitigate it
- [ ] Build a tamper-evident audit log: every tool call, timestamp, caller context, arguments
- [ ] Identify one irreversible action in an agent you built — add an explicit human confirmation step before it runs

---

## 16. LangGraph

> LangGraph is a framework for building stateful, graph-based agent workflows. Learn it *after* understanding the primitives — it implements concepts from §3, §7, §9, and §13 in a concrete, structured way.

**Core mental model:** LangGraph represents agent workflows as directed graphs — nodes are processing steps (LLM calls, tool calls, logic), edges are transitions between them, and state flows through the graph with full persistence.

### Core Concepts

| Concept | What to understand |
|---|---|
| **StateGraph** | The fundamental building block — defines a graph with typed shared state that all nodes can read and write |
| **Nodes** | Functions or runnables that receive state and return updates; can be LLM calls, tool executors, or plain Python |
| **Edges** | Connections between nodes; can be unconditional (always go to B after A) or conditional (route based on state) |
| **Conditional edges** | A function that inspects state and returns the name of the next node — this is where routing logic lives |
| **START / END** | `START` is the entrypoint constant used in `add_edge(START, "first_node")`; `END` signals graph completion — importing both from `langgraph.graph` |
| **State schema** | Typed definition of what the graph carries; usually a `TypedDict` or Pydantic model; all nodes read and return updates to this schema |
| **MessagesState** | Built-in state schema with a pre-configured append reducer for the `messages` key — the standard starting point for chat-based agents |
| **Reducers** | Control how state fields are merged when a node returns an update — default is overwrite; `Annotated[list, operator.add]` is the standard append-to-list pattern |
| **Compiled graph** | `graph.compile()` produces a runnable — checkpointers and interrupt configs are passed here, not at graph definition time |

### Persistence & Checkpointing

| Concept | What to understand |
|---|---|
| **Checkpointers** | Persist graph state after every node — enables pause, resume, and replay; built-in: `MemorySaver`, `SqliteSaver`, `PostgresSaver` |
| **Thread IDs** | Each run is identified by a thread — different threads are independent; same thread resumes from last checkpoint |
| **State replay** | Rewind a graph to any previous checkpoint and re-run from that point — critical for debugging and HITL corrections |
| **Time-travel debugging** | Use state history to inspect what the graph knew at each step; identify exactly where a failure occurred |

### Human-in-the-Loop in LangGraph

| Concept | What to understand |
|---|---|
| **`interrupt_before` / `interrupt_after`** | Compile-time config passed to `graph.compile()` — pauses execution before or after specific nodes; human resumes via `graph.invoke(None, config)` to replay from the checkpoint |
| **`interrupt()` function** | Called inside a node to pause mid-execution; the value passed to `interrupt()` is surfaced to the caller; resuming via `Command(resume=value)` returns `value` as the result of the `interrupt()` call inside the node — the node then continues from that point |
| **`Command(goto=..., update=...)`** | Used inside nodes to explicitly route to another node and optionally update state in one operation — commonly used for handoffs between agents |
| **Approval as a graph node** | Model approval gates as explicit nodes in the graph — makes the HITL flow visible in the graph topology |

### Multi-Agent Patterns in LangGraph

| Concept | What to understand |
|---|---|
| **Subgraphs** | A compiled graph used as a node inside another graph — each subgraph has its own state schema; the primary way to compose multi-agent systems |
| **Supervisor pattern** | An LLM node that decides which worker to call next via conditional edges; workers can be subgraphs or plain nodes |
| **Handoffs** | A node returns `Command(goto="other_agent")` to transfer control; the target agent resumes from shared state |
| **Shared vs. private state** | Parent and subgraph state schemas can share keys (overlapping) or be fully isolated; overlapping keys must use compatible reducers — design this deliberately |

### Streaming

| Concept | What to understand |
|---|---|
| **`stream_mode="values"`** | Emits the full state after each node completes |
| **`stream_mode="updates"`** | Emits only the state delta from each node — more efficient for large state |
| **`stream_mode="messages"`** | Streams LLM tokens as they generate — for user-facing, low-latency output |
| **Streaming in subgraphs** | Tokens and updates from nested subgraphs surface through the parent stream — requires `subgraphs=True` |

### What LangGraph Does NOT Do

Understanding the limits is as important as understanding the features:
- It does not make agents smarter — it structures the control flow around them
- It does not handle prompt design, tool quality, or evaluation — those are your responsibility
- The graph structure makes agent behavior more predictable and debuggable, but a bad prompt inside a node is still a bad prompt

### Exercises
- [ ] Build a ReAct agent using LangGraph's `StateGraph` — compare the structure to your raw ReAct loop from §3; note what LangGraph handles and what you still own
- [ ] Add `MemorySaver` checkpointing — pause mid-run, inspect the saved state, resume from the checkpoint
- [ ] Implement a conditional edge that routes to one of three nodes based on the current state — test all three branches
- [ ] Build a supervisor graph with two specialist subgraphs; verify state handoff between them is clean and typed
- [ ] Add `interrupt_before` to a node that calls a write tool — implement the human approval step using `Command(resume=...)`
- [ ] Use time-travel debugging: deliberately inject a bad state at step 2, observe the failure at step 4, rewind to step 2, fix the state, re-run
