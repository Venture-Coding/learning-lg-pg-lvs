# LinkedIn Posts — 14-Day Sprint Series

_Style: Story hook → Technical insight → Market/career signal_  
_Tone: First-person, direct, no corporate fluff_  
_Format note: LinkedIn doesn't render markdown — use line breaks, no headers_  
_Last updated: 27 Feb 2026_

---

## How to use this file

- Each post maps to a sprint milestone
- Draft is your starting point — personalise with your actual code/output
- Add a real screenshot, diagram, or code snippet as the post image
- Post in the evening of the day it's tagged to (or the morning after)
- Engage with comments for the first hour after posting — it signals the algorithm

---

## Post 1 — Day 3 | LangGraph: The Mental Model Shift

_Post after building the supervisor pattern on Day 3_

---

**[DRAFT]**

I've been using LangChain for over a year.

Last week I finally sat down to properly learn LangGraph — and within 3 days I understood why every serious agentic AI system I've read about uses it instead of plain chains.

The shift is this:

LangChain chains are pipelines. They go A → B → C. That's great for simple workflows.

But real agents need loops. They need to call a tool, check the result, decide whether to call another tool, loop back, and eventually stop.

That's what LangGraph gives you. It's a graph, not a pipeline. Your agent's state flows through nodes. Edges decide what runs next — including conditional edges that let the LLM itself make routing decisions.

Here's the simplest mental model:

→ StateGraph = the whole workflow
→ Node = a Python function that reads + writes state
→ Edge = the connection between nodes
→ Conditional edge = "run this function to decide where to go next"

The part that clicked hardest: the state object is shared. Every node can read everything. No more passing outputs as inputs manually.

I built a 3-agent supervisor system yesterday:
• ResearchAgent retrieves context
• WriterAgent drafts the output
• Supervisor decides which to call and when to stop

The whole thing is ~120 lines of Python.

If you're building anything agentic and still using plain chains — LangGraph is worth 3 days of your time.

---

📌 **Image suggestion:** A diagram of your 3-node supervisor graph (draw.io or just a whiteboard photo)  
📌 **Hashtags:** `#LangGraph #AgenticAI #LLM #Python #AIEngineering`

---

## Post 2 — Day 5 | I Deployed a LangGraph Agent to AWS

_Post after Day 5 deployment to EC2_

---

**[DRAFT]**

5 days ago I didn't have a single LangGraph agent in production.

Today I have one running on AWS, wrapped in FastAPI, streaming responses over HTTP.

Here's what the stack looks like:

→ LangGraph: multi-agent graph with supervisor + 2 worker agents
→ FastAPI: POST /invoke for full response, GET /stream for SSE streaming
→ Docker: containerised, 180MB image
→ AWS ECR + EC2: deployed, callable from anywhere

The thing most tutorials skip: streaming.

Most demos just call `.invoke()` and wait. But real users don't want to wait 10 seconds staring at a blank screen.

LangGraph has `.astream_events()` — it emits events as each node runs. Pipe those into a FastAPI `StreamingResponse` and you get real-time token streaming with full control over what gets sent when.

The code to wire this up is surprisingly small:

```python
@app.get("/stream")
async def stream(query: str):
    async def event_generator():
        async for event in graph.astream_events({"query": query}, version="v2"):
            if event["event"] == "on_chat_model_stream":
                yield f"data: {event['data']['chunk'].content}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

This exact pattern comes up in the JDs I'm targeting. Not a coincidence I'm building it.

What's your current deployment setup for LangGraph agents?

---

📌 **Image suggestion:** Screenshot of your EC2 endpoint being called from curl with streaming output  
📌 **Hashtags:** `#LangGraph #AWS #FastAPI #AgenticAI #Python #AIEngineer`

---

## Post 3 — Day 8 | Nobody Actually Evaluates Their RAG Pipeline

_Post after Day 8 RAGAS + LangSmith work_

---

**[DRAFT]**

I've built a few RAG pipelines. My evaluation strategy was:

"Does it return something sensible?"

That's not evaluation. That's vibes.

This week I properly learned RAGAS — a framework that evaluates RAG pipelines using 4 metrics:

→ **Faithfulness** — is the answer supported by the retrieved context, or is the LLM hallucinating?
→ **Answer Relevancy** — does the answer actually address the question?
→ **Context Precision** — are the retrieved chunks relevant, or is there noise?
→ **Context Recall** — did retrieval find everything needed to answer?

I ran RAGAS against one of my existing RAG setups. Faithfulness: 0.91. Context Precision: 0.62.

That 0.62 told me retrieval was pulling in irrelevant chunks 38% of the time. The LLM was compensating by ignoring them — but it was burning tokens and occasionally confusing the output.

One retrieval parameter change later: Context Precision hit 0.84.

I would never have found that without a metric.

The second thing I set up: LangSmith for tracing. Every LLM call, every node in the graph, every token — logged automatically with zero code changes. Just two environment variables.

Now I can see which invocations are slow, which are expensive, and when a new prompt change degrades quality.

This is what "production AI" actually means. Not just getting it to work — getting it to work *measurably*, consistently.

---

📌 **Image suggestion:** LangSmith trace screenshot showing a slow node, or a RAGAS scores comparison table  
📌 **Hashtags:** `#RAG #LLMEval #LangSmith #RAGAS #LLMOps #AIEngineering`

---

## Post 4 — Day 10 | I Broke My AI Agent on Purpose (Using CI/CD)

_Post after Day 10 GitHub Actions eval pipeline_

---

**[DRAFT]**

Yesterday I deliberately made my LLM agent worse.

Changed one line in the system prompt. Made it slightly more verbose.

My GitHub Actions pipeline caught it before it could merge.

Here's the setup:

Every pull request runs an automated eval:
→ Load 10 test question/answer pairs
→ Run each through the agent
→ Score outputs with RAGAS + a custom LLM-as-judge evaluator
→ If Faithfulness < 0.85 or Answer Relevancy < 0.80 → fail the PR

Takes 90 seconds. Costs maybe $0.15 in API calls.

The PR with my "more verbose" prompt had Answer Relevancy drop to 0.73. Blocked.

This is what separates AI teams that ship reliably from ones that are always fighting regressions.

The dirty secret of most AI projects: the eval only happens manually, at demo time, when someone notices something feels off.

By then the bad change has been in production for days.

3 things make this work:
1. A versioned test dataset (mine lives in Git as a JSON file)
2. A scored, reproducible eval script (not "does it sound right")
3. A deployment gate with clear thresholds

None of this is new software engineering. It's just applying basic CI/CD discipline to AI.

Took me one afternoon to set up. Worth it.

---

📌 **Image suggestion:** GitHub Actions PR check showing a failed eval with metric scores  
📌 **Hashtags:** `#LLMOps #AIEngineering #CI_CD #MLOps #GitHub #LLMEval`

---

## Post 5 — Day 12–13 | asyncio Made My Agent 6× Faster

_Post after Day 12 async LLM call benchmarking_

---

**[DRAFT]**

I ran a benchmark yesterday that hurt to look at.

Sequential LLM calls for 10 queries: **47 seconds**
Async parallel calls for the same 10 queries: **8 seconds**

Same model. Same prompts. Same machine. Six times faster.

The reason is simple: LLM API calls are I/O bound. While you're waiting for the response, your Python process is doing nothing. Async lets you fire all 10 calls, then collect the results as they arrive.

```python
import asyncio
from langchain_openai import ChatOpenAI

llm = ChatOpenAI()

async def call_llm(query: str) -> str:
    response = await llm.ainvoke(query)
    return response.content

async def batch_queries(queries: list[str]) -> list[str]:
    return await asyncio.gather(*[call_llm(q) for q in queries])
```

That's it. The whole change.

But there's one trap: if you do this naively with 100 queries, you'll hit rate limits instantly. The fix is a semaphore:

```python
sem = asyncio.Semaphore(5)  # max 5 concurrent requests

async def call_llm_safe(query: str) -> str:
    async with sem:
        response = await llm.ainvoke(query)
        return response.content
```

Now you get the speed benefit without hammering the API.

This pattern comes up everywhere in production agent systems — parallel tool calls, fan-out retrieval, batch evaluation. It's a fundamental skill gap I've been intentionally closing this week.

---

📌 **Image suggestion:** Terminal output showing the benchmark comparison, or a simple timing table  
📌 **Hashtags:** `#Python #asyncio #LLM #AIEngineering #FastAPI #Performance`

---

## Post 6 — Day 14 | What 14 Days of Deliberate Learning Looks Like

_Sprint wrap-up post — Day 14_

---

**[DRAFT]**

14 days ago I sat down with a specific goal: close the skill gaps between where I am and where I want to be hired.

Not a course. Not tutorials. Deliberate practice, daily shipping, and posting what I built.

Here's what I built:

**Week 1 — LangGraph**
→ A multi-agent supervisor system (3 agents, shared state, conditional routing)
→ With memory and human-in-the-loop interrupts
→ Deployed to AWS via Docker + FastAPI with streaming

**Week 2 — LLM Evaluation + LLMOps**
→ RAGAS evaluation on my RAG pipeline (found a real retrieval bug from the metrics)
→ Custom LLM-as-judge evaluator for domain-specific quality
→ LangSmith tracing across all agent runs
→ GitHub Actions CI/CD gate — PRs blocked if eval scores drop

**Deep Python — asyncio**
→ Async LangGraph agent (all nodes `async`, parallel tool calls)
→ Streaming FastAPI endpoint
→ 6× throughput improvement over sync baseline (benchmark included)

What I know now that I didn't 14 days ago:

The gap between "I've used LangChain" and "I build production agentic AI systems" isn't that large. It's specific: state management, evaluation discipline, observability, and async engineering.

These aren't exotic skills. They're what every mid-to-senior AI Engineer JD in Australia is asking for right now.

If you're an AI/ML engineer thinking about what to learn next — build something deployable. Measure it. Document the measurement. That's the job.

---

📌 **Image suggestion:** A collage — LangGraph diagram + LangSmith trace + GitHub Actions green check  
📌 **Hashtags:** `#AIEngineer #LangGraph #LLMOps #Learning #CareerDevelopment #Python #GenAI`

---

## Posting Schedule

| Post | Topic | Post On |
|---|---|---|
| #1 | LangGraph mental model | Day 3 evening (Mar 3) |
| #2 | LangGraph deployed to AWS | Day 5 evening (Mar 5) |
| #3 | RAG evaluation + LangSmith | Day 8 evening (Mar 8) |
| #4 | CI/CD eval pipeline | Day 10 evening (Mar 10) |
| #5 | asyncio benchmark | Day 13 evening (Mar 13) |
| #6 | Sprint wrap-up | Day 14 evening (Mar 14) |

## Tips

- Post between **7–9am AEST** or **6–8pm AEST** — highest engagement for AU tech audience  
- Reply to every comment in the **first 60 minutes** — this is the biggest algorithmic signal  
- Don't post and ghost. Even "thanks for the question, I'll write more about X" counts  
- Week 3 onward: send a connection request + message to 1–2 hiring managers at each of your top 3 target companies, referencing the posts
