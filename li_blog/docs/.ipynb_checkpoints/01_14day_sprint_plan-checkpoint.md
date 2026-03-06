# 14-Day Learning Sprint Plan

_Skills: LangGraph in Production · LLM Evaluation / LLMOps · asyncio + Concurrency_  
_Start date: 1 Mar 2026 | Pace: 2+ hrs/day | Style: Short concept → heavy hands-on_  
_Last updated: 27 Feb 2026_

---

## Overview

| Days | Skill | Goal |
|---|---|---|
| 1 – 5 | LangGraph in Production | Build real multi-agent workflows with state, memory, and deployment |
| 6 – 10 | LLM Evaluation / LLMOps | Evaluate RAG, trace LLM calls, add guardrails |
| 11 – 14 | asyncio + Python Concurrency | Write async-native agent code, parallel LLM calls |

LinkedIn posts are embedded throughout — post while you learn.

---

## Days 1–5: LangGraph in Production

### Day 1 — Mental Model: What is LangGraph and why does it exist?

**Concept (30 min)**
- Why LangChain chains break down for complex agents (no cycles, no shared state)
- LangGraph core primitives: `StateGraph`, nodes, edges, conditional edges
- The state object — what it is, why it's the heartbeat of the graph
- Compile → invoke vs stream
- Read: [LangGraph conceptual docs](https://langchain-ai.github.io/langgraph/concepts/)

**Hands-on (90 min)**
- Install: `pip install langgraph langchain-openai`
- Build a 2-node graph: `call_llm` → `format_output`
- Add a `TypedDict` state with at least 3 fields
- Run with both `.invoke()` and `.stream()`
- Inspect intermediate state at each node

**Output:** Working 2-node graph you can explain node-by-state-by-node

---

### Day 2 — Conditional Edges + Routing Logic

**Concept (20 min)**
- `add_conditional_edges()` — how routing decisions work
- `END` node — when and how to terminate
- Difference between deterministic and LLM-driven routing

**Hands-on (100 min)**
- Build a simple ReAct-style agent:
  - Node: `agent` (LLM decides: use tool or answer)
  - Node: `tools` (execute tool call)
  - Conditional edge: if tool_call → tools, else → END
- Use a real tool (e.g. a search function or calculator)
- Add basic logging to print state at each step

**Output:** A working ReAct agent in LangGraph you built from scratch

---

### Day 3 — Multi-Agent: Supervisor Pattern

**Concept (30 min)**
- What is the supervisor pattern — one orchestrator, many worker agents
- How to pass messages between agents via shared state
- When to use supervisor vs peer-to-peer vs hierarchical

**Hands-on (90 min)**
- Build a supervisor graph with 2 worker sub-graphs:
  - `ResearchAgent` — scrapes/retrieves info
  - `WriterAgent` — drafts output from retrieved info
  - `Supervisor` node — routes between them, decides when done
- Use `MessagesState` for message passing
- Test with a real prompt end-to-end

**Output:** A 3-agent supervisor system you can demo and explain

> 📝 **Write LinkedIn Post #1 tonight** — see `02_linkedin_posts.md`

---

### Day 4 — Memory, Persistence & Checkpointing

**Concept (30 min)**
- Short-term memory: state within a single run
- Long-term memory: persisting state across runs (checkpointers)
- `MemorySaver` (in-memory) vs `SqliteSaver` vs Postgres checkpointer
- Thread IDs and the `config` object

**Hands-on (90 min)**
- Add `MemorySaver` to your Day 3 multi-agent graph
- Run the same graph twice with the same thread_id — verify state persists
- Swap to `SqliteSaver` — inspect the database directly
- Add a "human-in-the-loop" interrupt: pause graph at a node, wait for user approval, then resume

**Output:** A persistent, interruptible agent graph

---

### Day 5 — Deploy: LangGraph on AWS

**Concept (20 min)**
- Serving options: FastAPI wrapper, LangGraph Cloud, self-hosted on EC2/Lambda
- Streaming responses over HTTP (SSE)
- Environment variable management for API keys in prod

**Hands-on (100 min)**
- Wrap your Day 3/4 graph in a FastAPI app:
  - `POST /invoke` — returns full result
  - `GET /stream` — streams state updates as SSE
- Dockerise: write a `Dockerfile`, build and run locally
- Deploy to EC2 (or ECS if comfortable):
  - Push Docker image to ECR
  - Run on EC2, expose via port
- Test end-to-end from a REST client (curl or Postman)

**Output:** A deployed, callable LangGraph agent on AWS

> 📝 **Write LinkedIn Post #2** — see `02_linkedin_posts.md`

---

## Days 6–10: LLM Evaluation / LLMOps

### Day 6 — What Does "Evaluating an LLM" Actually Mean?

**Concept (30 min)**
- Why vibes-based testing fails in production
- Offline eval vs online eval vs human eval
- Key RAG metrics: Faithfulness, Answer Relevancy, Context Precision, Context Recall
- What RAGAS is and how it works

**Hands-on (90 min)**
- Install: `pip install ragas langchain-openai`
- Take one of your existing RAG pipelines (or build a minimal one with Qdrant)
- Generate a small test dataset: 5–10 question/answer/context triples
- Run RAGAS evaluation on it — get metric scores
- Deliberately break the retrieval and see scores drop

**Output:** A scored evaluation of a RAG pipeline you own

---

### Day 7 — Building a Custom Eval Framework

**Concept (20 min)**
- Why you can't rely on RAGAS alone for domain-specific use cases
- LLM-as-a-judge pattern — using a strong LLM to score outputs
- Structuring eval datasets: what to store, how to version them

**Hands-on (100 min)**
- Write a custom evaluator:
  - Input: question, LLM answer, ground truth
  - Prompt a judge LLM (GPT-4o or Claude) to score on: correctness, conciseness, tone
  - Return structured JSON scores
- Build a simple eval runner:
  - Load test cases from a JSON file
  - Run each through your LangGraph agent
  - Score each output
  - Print a summary report
- Version your test dataset in Git

**Output:** A reusable eval script you can run on any agent output

---

### Day 8 — Tracing and Observability with LangSmith

**Concept (20 min)**
- What LangSmith is — tracing every LLM call, token count, latency, errors
- Why observability matters: debugging, cost tracking, regression detection
- Alternative: Arize Phoenix (open-source) if you prefer self-hosted

**Hands-on (100 min)**
- Set up LangSmith account (free tier) + set env vars
- Instrument your existing LangGraph agent — zero code change, just env vars
- Run 10 invocations and inspect traces in the LangSmith UI:
  - Find the slowest node
  - Find the most expensive call (token usage)
  - Find a failed run and trace the exact error
- Tag runs with metadata: `{user_id, session_id, version}`
- Create a dataset in LangSmith from real production traces

**Output:** Fully traced LangGraph agent with structured run metadata

> 📝 **Write LinkedIn Post #3** — see `02_linkedin_posts.md`

---

### Day 9 — Guardrails: Controlling LLM Inputs and Outputs

**Concept (30 min)**
- What guardrails solve: hallucination, prompt injection, off-topic responses, PII leakage
- Options: Guardrails AI, NeMo Guardrails, custom validators, LLM-based filters
- Where to put guardrails in an agent graph (input node vs output node vs both)

**Hands-on (90 min)**
- Install: `pip install guardrails-ai`
- Use Guardrails AI to:
  - Validate LLM output is valid JSON matching a Pydantic schema
  - Add a PII detection rail (block responses containing emails/phone numbers)
  - Add a topic filter — reject off-topic questions before they hit the LLM
- Integrate one guardrail as a node in your LangGraph agent
- Test: deliberately trigger each guardrail and verify it fires

**Output:** A guardrailed LangGraph agent with 3 types of validation

---

### Day 10 — CI/CD for AI: Automated Eval on Every Change

**Concept (20 min)**
- What MLOps / LLMOps CI/CD looks like: trigger eval on PR, gate deployment on score threshold
- Regression testing for LLMs — how to prevent a prompt change from silently degrading quality
- Canary deployments for AI: shadow mode, A/B split

**Hands-on (100 min)**
- Write a GitHub Actions workflow (`.github/workflows/eval.yml`):
  - Trigger: on pull request
  - Steps: install deps → run eval script → fail PR if any score < threshold
- Write a regression test: compare current agent vs baseline agent on the same dataset
- Simulate a "bad prompt change" — verify the pipeline catches it
- Document your eval thresholds and what each metric means

**Output:** A GitHub Actions eval pipeline that gates merges on quality

> 📝 **Write LinkedIn Post #4** — see `02_linkedin_posts.md`

---

## Days 11–14: asyncio + Python Concurrency

### Day 11 — How asyncio Actually Works

**Concept (40 min)**
- The event loop — what it is, what it runs
- `async def`, `await`, coroutines — what Python actually does at runtime
- `asyncio.gather()` vs `asyncio.create_task()` vs `asyncio.run()`
- When asyncio helps (I/O bound) vs when it doesn't (CPU bound → use `ProcessPoolExecutor`)
- Common mistakes: blocking the event loop, forgetting `await`, mixing sync and async

**Hands-on (80 min)**
- Write 5 small focused exercises:
  1. `gather()` — fan out 5 async functions, collect results
  2. `create_task()` — fire-and-forget + cancel a task
  3. `asyncio.timeout()` — cancel a slow coroutine after N seconds
  4. `asyncio.Queue` — producer/consumer pattern
  5. Call a synchronous function safely from async: `loop.run_in_executor()`

**Output:** 5 working async snippets covering the core patterns

---

### Day 12 — Async LLM Calls: Parallel Requests and Streaming

**Concept (20 min)**
- Why async matters for LLM calls: latency, throughput, rate limits
- LangChain async interface: `ainvoke`, `astream`, `abatch`
- Streaming tokens over HTTP: async generators + SSE

**Hands-on (100 min)**
- Run 5 LLM calls in parallel using `asyncio.gather()` — compare to sequential timing
- Implement streaming in FastAPI: `StreamingResponse` + async generator
- Add rate-limit handling: semaphore to cap concurrent LLM calls
- Implement retry with exponential backoff using `tenacity` (async version)
- Benchmark: sync vs async for 10, 50, 100 concurrent requests

**Output:** Async FastAPI endpoint that streams LLM responses with concurrency control

> 📝 **Write LinkedIn Post #5** — see `02_linkedin_posts.md`

---

### Day 13 — Async Context Managers, Generators & Error Handling

**Concept (20 min)**
- `async with` — async context managers, why they matter (DB connections, HTTP clients)
- `async for` — async generators, iterating streams
- Proper error handling in async: `try/except` inside coroutines, `asyncio.gather(return_exceptions=True)`
- Avoiding silent failures in async code

**Hands-on (100 min)**
- Build an async HTTP client using `httpx.AsyncClient` as an async context manager
- Write an async generator that streams chunks from an LLM and yields them
- Handle partial failures: gather 10 tasks, some fail — collect successes, log failures
- Write a reusable `async_retry` decorator
- Build an async connection pool pattern (simulate DB connection reuse)

**Output:** A set of reusable async utilities applicable to any agent codebase

---

### Day 14 — Apply Everything: Async LangGraph Agent

**Concept (10 min)**
- LangGraph async support: `ainvoke`, `astream_events`
- How to run an async LangGraph graph from an async FastAPI endpoint

**Hands-on (110 min)**
- Refactor your Day 5 LangGraph agent to be fully async:
  - All nodes use `async def`
  - Tool calls are parallelised with `asyncio.gather()` where independent
  - FastAPI endpoint uses `async def` + `StreamingResponse`
  - Add semaphore-based concurrency control
- Run a load test (use `locust` or `httpx` async client) — compare sync vs async throughput
- Final cleanup: type hints, docstrings, pytest-asyncio tests for at least 2 nodes

**Output:** A production-grade async LangGraph agent with streaming, tested and load-verified

> 📝 **Write LinkedIn Post #6 (Sprint Wrap-Up)** — see `02_linkedin_posts.md`

---

## Summary: What You'll Have After 14 Days

| Deliverable | Skills Demonstrated |
|---|---|
| Multi-agent LangGraph system (deployed on AWS) | LangGraph, multi-agent design, AWS deployment |
| RAGAS + custom eval framework with CI/CD gate | LLM evaluation, test-driven AI, GitHub Actions |
| Traced + guardrailed production agent (LangSmith) | LLMOps, observability, safety |
| Fully async LangGraph agent with streaming + load test | asyncio, FastAPI, production engineering |
| 6 LinkedIn posts | Personal brand, market signal, thought leadership |

---

## Resources

| Topic | Resource |
|---|---|
| LangGraph | [LangGraph docs](https://langchain-ai.github.io/langgraph/) |
| LangGraph multi-agent patterns | [LangGraph How-To guides](https://langchain-ai.github.io/langgraph/how-tos/) |
| RAGAS | [ragas.io docs](https://docs.ragas.io) |
| LangSmith | [smith.langchain.com](https://smith.langchain.com) |
| Guardrails AI | [guardrailsai.com/docs](https://www.guardrailsai.com/docs) |
| asyncio | [Python asyncio docs](https://docs.python.org/3/library/asyncio.html) |
| asyncio deep dive | [Real Python asyncio guide](https://realpython.com/async-io-python/) |
| pytest-asyncio | [pytest-asyncio docs](https://pytest-asyncio.readthedocs.io) |
