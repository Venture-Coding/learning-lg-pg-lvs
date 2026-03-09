# AutoGen — When to Use It, How to Use It, What to Avoid

> **Audience:** Junior / mid-level engineers integrating AutoGen into a LangChain / LangGraph stack.  
> **TL;DR:** AutoGen has two killer features. Everything else should be done with LangChain. Using AutoGen outside these two cases adds cost and complexity with no benefit.

---

## Table of Contents

1. [The One Rule](#the-one-rule)
2. [Killer Feature 1 — Code Generation + Actual Execution](#killer-feature-1--code-generation--actual-execution)
3. [Killer Feature 2 — Multi-Agent Debate](#killer-feature-2--multi-agent-debate)
4. [What NOT to Use AutoGen For](#what-not-to-use-autogen-for)
5. [LLM Configuration (Vertex AI / Gemini)](#llm-configuration-vertex-ai--gemini)
6. [Integrating AutoGen Inside a LangGraph Node](#integrating-autogen-inside-a-langgraph-node)
7. [Pitfalls and How to Avoid Them](#pitfalls-and-how-to-avoid-them)
8. [Quick Decision Checklist](#quick-decision-checklist)

---

## The One Rule

```
Does this task require:
  a) Running arbitrary generated code and feeding stdout back to the LLM?   → AutoGen
  b) Multiple agents arguing/critiquing each other until consensus?          → AutoGen
  c) Anything else?                                                          → LangChain
```

AutoGen's `AssistantAgent` alone (without code execution or GroupChat) is just an expensive `llm.invoke()` call. Don't use it for routing, structured output, RAG, tool calling, or single-turn generation.

---

## Killer Feature 1 — Code Generation + Actual Execution

### What it does

```
AssistantAgent writes Python in a ```python block
        ↓
UserProxyAgent extracts the block, runs it in a subprocess
        ↓
stdout / stderr / exit-code sent back to AssistantAgent
        ↓
AssistantAgent fixes errors
        ↓
loop continues until tests pass  OR  max_consecutive_auto_reply reached
```

**No other mainstream framework does this.** LangChain `bind_tools` can call *pre-defined* functions but cannot run *arbitrary generated code* and feed stdout back.

### Reusable pattern

```python
import textwrap
from pathlib import Path
from autogen import AssistantAgent, UserProxyAgent

CODE_SANDBOX = Path("./autogen_sandbox")
CODE_SANDBOX.mkdir(exist_ok=True)

AUTOGEN_LLM_CONFIG = {
    "config_list": [
        {
            "model": "vertex_ai/gemini-2.0-flash",
            "api_type": "litellm",
            # Credentials from GOOGLE_APPLICATION_CREDENTIALS env var
        }
    ],
    "temperature": 0.1,
    "timeout": 120,
}


def run_code_task(task: str, max_turns: int = 8) -> dict:
    """
    Give AutoGen a coding task. It writes, runs, fixes, and returns
    the final working code + full conversation transcript.

    Args:
        task: Plain-English description of what to build/fix.
              Include any relevant code context as a string.
        max_turns: Safety cap — stops the loop after this many exchanges.

    Returns:
        {"code": str, "transcript": list[str], "success": bool}
    """
    # ── Always create a FRESH pair per task ──────────────────────────────────
    # Never reuse agents across tasks — stale conversation history causes
    # the agent to get confused by previous context.
    assistant = AssistantAgent(
        name="CodeWriter",
        llm_config=AUTOGEN_LLM_CONFIG,
        system_message=textwrap.dedent("""\
            You are an expert Python engineer.
            Rules:
            1. Write COMPLETE, RUNNABLE Python in a ```python block — no placeholders.
            2. Include pytest tests in a separate ```python block marked # TESTS.
            3. When you see execution output, fix any errors immediately.
            4. When ALL tests pass, end your message with exactly: TASK_COMPLETED
            5. Never use '...' or '# TODO' as placeholders.
        """),
    )

    proxy = UserProxyAgent(
        name="CodeRunner",
        human_input_mode="NEVER",           # fully autonomous
        max_consecutive_auto_reply=max_turns,
        is_termination_msg=lambda m: "TASK_COMPLETED" in (m.get("content") or ""),
        code_execution_config={
            "work_dir":        str(CODE_SANDBOX),
            "use_docker":      False,        # ⚠️ set True in production for isolation
            "timeout":         30,           # seconds per execution
            "last_n_messages": 3,            # only re-run code from recent messages
        },
    )

    chat_result = proxy.initiate_chat(
        recipient=assistant,
        message=task,
        summary_method="last_msg",
    )

    transcript = [
        f"[{m.get('name', '?')}]: {m.get('content', '')}"
        for m in chat_result.chat_history
    ]
    success = any("TASK_COMPLETED" in (m.get("content") or "") 
                  for m in chat_result.chat_history)

    return {
        "code":       chat_result.summary or "",
        "transcript": transcript,
        "success":    success,
    }


# Usage
if __name__ == "__main__":
    result = run_code_task(
        "Write a function `sliding_windows(lst, size, stride)` that returns "
        "overlapping windows of a list. Include edge-case pytest tests."
    )
    print("Success:", result["success"])
    print("Turns:  ", len(result["transcript"]))
    print(result["code"][:500])
```

### When to use it

| Task | Use AutoGen code executor? |
|---|---|
| Write a new utility function + tests | ✅ Yes |
| Fix a bug and verify tests pass | ✅ Yes |
| Run a data analysis script | ✅ Yes |
| Explain what a function does | ❌ No — single LLM call |
| Call a pre-defined API | ❌ No — LangChain `bind_tools` |
| Generate a SQL query | ❌ No — `with_structured_output` |

---

## Killer Feature 2 — Multi-Agent Debate

### What it does

Multiple `AssistantAgent` instances are placed in a `GroupChat`. Each has a different specialist system prompt (e.g. Architect, Performance Engineer, Security Analyst). They take turns responding to the same topic, critiquing each other's points. A designated consensus agent ends with `CONSENSUS:` to terminate.

**Why this matters:** A single LLM call on an architecture or security question produces a balanced, hedged answer. Forcing three agents with *adversarial roles* to argue produces concrete weaknesses and mitigations that a single LLM typically self-censors.

### Reusable pattern

```python
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager


def run_debate(
    topic: str,
    agent_specs: list[dict],   # list of {"name": str, "role": str}
    max_rounds: int = 9,
) -> dict:
    """
    Run a multi-agent debate on a topic.

    Each agent in agent_specs gets its own system prompt based on its role.
    The LAST agent in the list is the designated consensus synthesiser —
    it should end its final message with 'CONSENSUS:'.

    Args:
        topic:       The question or artefact (code, design doc) to debate.
        agent_specs: Ordered list of agent roles. Last one synthesises.
        max_rounds:  Total turns across all agents before forced termination.

    Returns:
        {"consensus": str, "transcript": list[str]}
    """
    agents = []
    for i, spec in enumerate(agent_specs):
        is_last = (i == len(agent_specs) - 1)
        suffix = (
            "\n\nIMPORTANT: After all agents have spoken, synthesise the key "
            "findings and end your message with 'CONSENSUS:' followed by your summary."
            if is_last else ""
        )
        agents.append(AssistantAgent(
            name=spec["name"],
            llm_config=AUTOGEN_LLM_CONFIG,
            system_message=spec["role"] + suffix,
        ))

    # Initiator only starts the conversation — never speaks again
    initiator = UserProxyAgent(
        name="Initiator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda m: "CONSENSUS:" in (m.get("content") or ""),
        code_execution_config=False,
    )

    group_chat = GroupChat(
        agents=[initiator] + agents,
        messages=[],
        max_round=max_rounds,
        speaker_selection_method="round_robin",   # deterministic — easiest to debug
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=AUTOGEN_LLM_CONFIG,
    )

    initiator.initiate_chat(recipient=manager, message=topic)

    transcript = [
        f"[{m.get('name', '?')}]: {m.get('content', '')[:600]}"
        for m in group_chat.messages
    ]
    consensus = ""
    for msg in reversed(group_chat.messages):
        content = msg.get("content", "")
        if "CONSENSUS:" in content:
            consensus = content[content.index("CONSENSUS:"):]
            break

    return {
        "consensus":  consensus or (transcript[-1] if transcript else "No consensus."),
        "transcript": transcript,
    }


# ── Pre-built debate configs ─────────────────────────────────────────────────

ARCHITECTURE_DEBATE = [
    {
        "name": "Architect",
        "role": "Senior software architect. Evaluate for: scalability, "
                "separation of concerns, coupling, extensibility. Be direct about weaknesses.",
    },
    {
        "name": "PerformanceEngineer",
        "role": "Performance engineer. Identify: latency hotspots, memory leaks, "
                "N+1 queries, missing caches. Cite specific lines or patterns.",
    },
    {
        "name": "Pragmatist",
        "role": "Pragmatic senior engineer. Challenge over-engineering. "
                "Ask: what is the simplest solution? What is the maintenance cost?",
    },
]

SECURITY_DEBATE = [
    {
        "name": "SecurityAnalyst",
        "role": "AppSec expert (OWASP Top 10, NIST). Find injection risks, "
                "broken auth, data exposure, insecure deps. Cite CWE numbers.",
    },
    {
        "name": "Attacker",
        "role": "Offensive security thinker. Describe CONCRETE exploit scenarios: "
                "exact inputs, privilege escalation paths, data exfil routes.",
    },
    {
        "name": "Defender",
        "role": "Defensive reviewer. Propose concrete mitigations with code snippets "
                "for each attack scenario raised.",
    },
]


# Usage
if __name__ == "__main__":
    result = run_debate(
        topic="Review this architecture: a single FastAPI gateway routes all requests "
              "to specialised sub-agents. At 10k req/min, what breaks first?",
        agent_specs=ARCHITECTURE_DEBATE,
    )
    print(result["consensus"])
```

### When to use it

| Task | Use AutoGen debate? |
|---|---|
| Architecture / design review | ✅ Yes — adversarial critique finds real issues |
| Security threat modelling | ✅ Yes — attacker + defender roles |
| Code quality review with multiple perspectives | ✅ Yes |
| Answering a factual question | ❌ No — single LLM is faster and cheaper |
| Routing a request | ❌ No — classification, use `with_structured_output` |
| Summarising a document | ❌ No — single LCEL chain |

---

## What NOT to Use AutoGen For

These are the most common mistakes junior devs make when they first see AutoGen.

### ❌ Routing / intent classification

```python
# BAD — AssistantAgent is overkill, adds latency, no benefit
agent = AssistantAgent(name="Router", llm_config=config)
proxy = UserProxyAgent(name="Proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1)
proxy.initiate_chat(agent, message=f"Classify this query: {query}")

# GOOD — one LLM call, schema-validated output
from pydantic import BaseModel
from langchain_google_vertexai import ChatVertexAI

class RouteDecision(BaseModel):
    task_type: str
    file_scope: list[str]

llm = ChatVertexAI(model_name="gemini-2.0-flash", temperature=0.0)
decision: RouteDecision = llm.with_structured_output(RouteDecision).invoke(messages)
```

### ❌ Tool calling

```python
# BAD — AutoGen's function_map requires custom serialisation boilerplate
# and doesn't benefit from Gemini's native function-calling protocol

# GOOD — LangChain bind_tools passes full JSON schemas natively
from langchain.tools import tool

@tool
def web_search(query: str) -> str:
    """Search the web."""
    ...

tools_llm = llm.bind_tools([web_search])
response = tools_llm.invoke(messages)
# response.tool_calls = [{"name": "web_search", "args": {"query": "..."}}]
```

### ❌ Structured JSON output

```python
# BAD — parsing free-text from AssistantAgent output is fragile
agent = AssistantAgent(...)
# "Please respond in JSON format: {..." — unreliable

# GOOD — Pydantic enforces schema at the protocol level
class BugReport(BaseModel):
    findings: list[str]
    severity: str

report: BugReport = llm.with_structured_output(BugReport).invoke(messages)
```

### ❌ Simple single-turn generation

```python
# BAD — creates two agents, a conversation loop, serialisation overhead
# for what is literally one LLM call
assistant = AssistantAgent(name="Writer", llm_config=config)
proxy = UserProxyAgent(name="User", human_input_mode="NEVER", max_consecutive_auto_reply=1)
proxy.initiate_chat(assistant, message="Summarise this document: ...")

# GOOD
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

chain = ChatPromptTemplate.from_messages([
    ("system", "You are a technical summariser."),
    ("human", "Summarise: {document}"),
]) | llm | StrOutputParser()

summary = chain.invoke({"document": text})
```

---

## LLM Configuration (Vertex AI / Gemini)

AutoGen uses its own client, separate from LangChain. Point both at the same endpoint via litellm to avoid duplicated model loading and credential setup.

```python
import os
from langchain_google_vertexai import ChatVertexAI

GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "my-project")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ID     = "gemini-2.0-flash"

# LangChain client — for routing, RAG, structured output
lc_llm = ChatVertexAI(
    model_name=MODEL_ID,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    temperature=0.0,
    max_output_tokens=4096,
)

# AutoGen client — for code execution and debate nodes
# Uses litellm under the hood → same Vertex AI endpoint
AUTOGEN_LLM_CONFIG = {
    "config_list": [
        {
            "model":    f"vertex_ai/{MODEL_ID}",
            "api_type": "litellm",
            # Credentials: set GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json
            # OR run on GCP with a service account — ADC picks it up automatically
        }
    ],
    "temperature": 0.1,   # slight warmth helps code generation creativity
    "timeout":     120,   # generous — code execution can be slow
}

# For Gemini Developer API (non-Vertex) instead:
# AUTOGEN_LLM_CONFIG = {
#     "config_list": [{"model": "gemini/gemini-2.0-flash", "api_type": "litellm"}],
#     "temperature": 0.1,
# }
# os.environ["GEMINI_API_KEY"] = "your-key"
```

---

## Integrating AutoGen Inside a LangGraph Node

AutoGen runs synchronously inside a LangGraph node. LangGraph treats the entire AutoGen conversation loop (however many turns) as one atomic node execution. The checkpoint is written after the node completes.

```python
from typing import Annotated
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AIMessage, HumanMessage
from typing_extensions import TypedDict


class State(TypedDict):
    query:      str
    result:     str
    transcript: list[str]
    messages:   Annotated[list, add_messages]


def node_autogen_code(state: State) -> dict:
    """
    AutoGen runs inside here. LangGraph sees this as one step.
    The full write→run→fix loop is invisible to LangGraph's state machine —
    only the final result and transcript are stored in state.
    """
    result = run_code_task(state["query"])   # from the pattern above
    return {
        "result":     result["code"],
        "transcript": result["transcript"],
        "messages":   [AIMessage(content=f"[CodeExecutor] success={result['success']}")],
    }


def build_graph():
    g = StateGraph(State)
    g.add_node("code_executor", node_autogen_code)
    g.add_edge(START, "code_executor")
    g.add_edge("code_executor", END)

    # SqliteSaver: every node's state is checkpointed to disk.
    # Use AsyncPostgresSaver for production multi-user deployments.
    saver = SqliteSaver.from_conn_string("checkpoints.db")
    return g.compile(checkpointer=saver)


graph = build_graph()

# Same session_id = persistent memory across calls
result = graph.invoke(
    {"query": "Write a binary search function with tests", "result": "", 
     "transcript": [], "messages": [HumanMessage(content="write binary search")]},
    config={"configurable": {"thread_id": "user-alice-session-1"}},
)
```

---

## Pitfalls and How to Avoid Them

### Pitfall 1 — Reusing agent instances across tasks

```python
# BAD — stale conversation history bleeds into the next task
assistant = AssistantAgent(...)   # created once at module level
proxy = UserProxyAgent(...)

proxy.initiate_chat(assistant, message="Task A: write sort function")
proxy.initiate_chat(assistant, message="Task B: write binary search")
# ⚠️ Task B now sees Task A's conversation in context

# GOOD — fresh pair per task
def run_code_task(task: str):
    assistant = AssistantAgent(...)   # new instance every call
    proxy = UserProxyAgent(...)
    return proxy.initiate_chat(assistant, message=task)
```

### Pitfall 2 — No execution timeout

```python
# BAD — generated code with an infinite loop hangs your server forever
code_execution_config={
    "work_dir": "./sandbox",
    "use_docker": False,
}

# GOOD — always set a timeout
code_execution_config={
    "work_dir":  "./sandbox",
    "use_docker": False,
    "timeout":   30,   # seconds — kill the subprocess if it runs over
}
```

### Pitfall 3 — `use_docker=False` in production

```python
# BAD for production — generated code runs in YOUR process's Python environment
# A malicious or buggy generated script can read files, delete data, make network calls
code_execution_config={"work_dir": "./sandbox", "use_docker": False}

# GOOD for production — Docker container with no network, read-only mounts
code_execution_config={
    "work_dir":   "./sandbox",
    "use_docker": True,
    # AutoGen uses the 'python:3.11-slim' image by default
    # Configure volumes and network limits in Docker settings
}
```

### Pitfall 4 — Unbounded conversation loops

```python
# BAD — no cap means the loop can run 50 turns, burning tokens
proxy = UserProxyAgent(
    name="Runner",
    human_input_mode="NEVER",
    # max_consecutive_auto_reply not set — defaults to unlimited
)

# GOOD — always set a cap
proxy = UserProxyAgent(
    name="Runner",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=8,   # give up after 8 fix attempts
    is_termination_msg=lambda m: "TASK_COMPLETED" in (m.get("content") or ""),
)
```

### Pitfall 5 — GroupChat without a termination signal

```python
# BAD — debate runs all max_round turns regardless of quality
group_chat = GroupChat(agents=[...], messages=[], max_round=12)

# GOOD — designate one agent as consensus synthesiser;
# UserProxyAgent terminates on its signal
proxy = UserProxyAgent(
    name="Initiator",
    human_input_mode="NEVER",
    max_consecutive_auto_reply=0,
    is_termination_msg=lambda m: "CONSENSUS:" in (m.get("content") or ""),
    code_execution_config=False,
)
# Last agent in your spec list must end with "CONSENSUS: <summary>"
```

### Pitfall 6 — Using AutoGen for the gateway / router

```python
# BAD — three-agent debate to decide which sub-agent to call
# This is a classification task; AutoGen adds ~3s latency and 3× the tokens

# GOOD — one structured LLM call with a Pydantic output schema
decision = llm.with_structured_output(GatewayDecision).invoke(messages)
# Returns in ~300ms, schema-validated, zero conversation overhead
```

---

## Quick Decision Checklist

```
New task arrives →

  Does it need to WRITE code AND verify it runs?
  (output is verified by actually executing it)
      YES → AutoGen AssistantAgent + UserProxyAgent
      NO  ↓

  Does it need MULTIPLE expert perspectives arguing adversarially?
  (architecture review, security threat model, design trade-offs)
      YES → AutoGen GroupChat + GroupChatManager
      NO  ↓

  Does it need a structured JSON / Pydantic output?
      YES → LangChain with_structured_output
      NO  ↓

  Does it need to call pre-defined tools / APIs?
      YES → LangChain bind_tools
      NO  ↓

  Does it need to search documents?
      YES → LangChain BM25Retriever / Chroma + LCEL chain
      NO  ↓

  Single-turn text generation?
      YES → LangChain ChatPromptTemplate | llm | StrOutputParser()

  Need state machine + persistent memory across sessions?
      ALWAYS → LangGraph StateGraph + SqliteSaver / AsyncPostgresSaver
```

---

## Summary Table

| Capability | Best tool | Never use |
|---|---|---|
| Execute arbitrary generated code | AutoGen `UserProxyAgent` | LangChain tools |
| Write → run → fix loop | AutoGen `AssistantAgent` + `UserProxyAgent` | LangGraph loops |
| Multi-agent adversarial critique | AutoGen `GroupChat` | Chained single LLM calls |
| Routing / intent classification | LangChain `with_structured_output` + Pydantic | AutoGen |
| Tool / API calling | LangChain `bind_tools` | AutoGen `function_map` |
| Structured JSON output | LangChain `with_structured_output` | AutoGen free-text parsing |
| State machine + conditional routing | LangGraph `StateGraph` | AutoGen |
| Persistent checkpoint memory | LangGraph `SqliteSaver` | AutoGen chat history |
| Single-turn text generation | LangChain LCEL chain | AutoGen |
| RAG / retrieval | LangChain retriever + LCEL | AutoGen |
