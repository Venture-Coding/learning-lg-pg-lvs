"""
rlm_gateway_2.py  —  Recursive LM Gateway  (NO DSPy)
======================================================
Exact same architecture as rlm_gateway.py, rebuilt with:

  Gateway routing     → LangChain with_structured_output + Pydantic
  Sub-agent chains    → ChatPromptTemplate | ChatVertexAI | PydanticOutputParser
  Assert/retry loop   → instructor library  (best DSPy.Assert drop-in)
  State + memory      → LangGraph StateGraph + SqliteSaver checkpoint
  Retrieval           → direct retriever.invoke() (no dspy.Retrieve)
  Chunked summariser  → plain Python recursion

What you LOSE vs the DSPy version
----------------------------------
  ✗  Automated prompt optimisation (BootstrapFewShot / MIPROv2)
       → prompts must be written and tuned BY HAND
       → no framework replicates this; it is DSPy's unique feature
  ✗  dspy.Retrieve as a trainable step (k is now a hard constant)
  ✗  Joint end-to-end optimisation across chained modules

What you GAIN
-------------
  ✓  Simpler mental model — plain Python classes, no DSPy abstractions
  ✓  LangGraph checkpoint memory (persist across sessions, resume, audit)
  ✓  instructor's retry-with-feedback is production-grade and well-tested
  ✓  No compile step — deploy immediately
  ✓  Easier to debug — every prompt is explicit and visible

Framework alternatives to DSPy (for this pattern)
--------------------------------------------------
1. LangGraph + LangChain   ← this file — best for stateful agents + memory
2. LlamaIndex              ← best for RAG-heavy pipelines (RouterQueryEngine)
3. instructor              ← best pure drop-in for dspy.Assert structured retry
4. Outlines                ← best for schema enforcement at token level (local models)
5. Semantic Kernel         ← best for Azure/C# shops, plugin model
6. CrewAI                  ← simpler than LangGraph, role-based agents
7. AutoGen / AG2           ← conversational multi-agent, human-in-the-loop
   None of these replicate DSPy's optimizer.  That gap is real.
"""

from __future__ import annotations

import ast
import json
import os
import sqlite3
import textwrap
from pathlib import Path
from typing import Annotated, Any, Literal

import instructor                               # pip install instructor
from openai import OpenAI                       # instructor wraps any OpenAI-compat client
from pydantic import BaseModel, Field, field_validator

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver   # pip install langgraph[sqlite]
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "my-project")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
GATEWAY_MODEL = "gemini-2.0-flash"
AGENT_MODEL   = "gemini-2.0-flash"

# LangChain clients
gateway_llm = ChatVertexAI(
    model_name=GATEWAY_MODEL, project=GCP_PROJECT,
    location=GCP_LOCATION, temperature=0.0, max_output_tokens=512,
)
agent_llm = ChatVertexAI(
    model_name=AGENT_MODEL, project=GCP_PROJECT,
    location=GCP_LOCATION, temperature=0.0, max_output_tokens=4096,
)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=GCP_PROJECT, location=GCP_LOCATION,
)

# instructor client — wraps an OpenAI-compatible endpoint for retry-with-feedback
# For Vertex AI, instructor can wrap the LangChain client via a compatibility shim,
# OR you can use instructor's native Vertex support:
#   import instructor, anthropic  (for Claude) or instructor.from_vertexai(...)
# Here we show the pattern; swap the client for your actual provider.
#
# instructor is the best available replacement for dspy.Assert:
#   - retries automatically when Pydantic validation fails
#   - injects the validation error message back into the prompt as feedback
#   - configurable max_retries
#
# instructor.from_vertexai() is available in instructor >= 1.3
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY", ""))          # for Gemini API
    instructor_client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name=AGENT_MODEL),
        mode=instructor.Mode.GEMINI_JSON,
    )
    USE_INSTRUCTOR = True
except Exception:
    # Fallback: use LangChain with_structured_output (less retry sophistication)
    USE_INSTRUCTOR = False

MAX_FILE_TOKENS = 6000
CHUNK_LINES     = 100


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Pydantic models — same as dspy.Signatures but pure Pydantic
#     These replace ALL dspy.Signature definitions.
# ─────────────────────────────────────────────────────────────────────────────

TaskType = Literal[
    "explain_code", "find_bug", "write_test",
    "summarise_module", "cross_file_reason",
]

class GatewayDecision(BaseModel):
    task_type:  TaskType    = Field(description="Type of task")
    file_scope: list[str]   = Field(description="Max 5 file paths needed to answer")
    sub_query:  str         = Field(description="Refined self-contained question for sub-agent")
    reasoning:  str         = Field(description="One sentence explaining the routing choice")

class ExplainOutput(BaseModel):
    explanation:  str       = Field(description="Clear structured explanation of the code")
    key_symbols:  list[str] = Field(description="Central function/class names")

class BugOutput(BaseModel):
    findings:  list[str]    = Field(
        description="Each finding MUST start with 'Line N:'. Empty if no bugs."
    )
    severity:  str          = Field(description="critical / high / medium / low / none")

    @field_validator("findings")
    @classmethod
    def citations_required(cls, v: list[str]) -> list[str]:
        """
        This validator IS the dspy.Assert equivalent via instructor.
        instructor will automatically retry the LLM call with the
        ValidationError message injected as feedback if this fails.
        """
        for finding in v:
            if finding.strip() and not finding.strip().startswith("Line "):
                raise ValueError(
                    f"Every finding must start with 'Line N:' — got: '{finding[:60]}'. "
                    "Re-read the code and add exact line numbers to every finding."
                )
        return v

class TestOutput(BaseModel):
    test_code:   str        = Field(description="Complete runnable pytest file")
    test_cases:  list[str]  = Field(description="One-line description per test case")

    @field_validator("test_code")
    @classmethod
    def valid_python(cls, v: str) -> str:
        """instructor retries if this raises — equivalent to dspy.Assert on syntax."""
        try:
            ast.parse(v)
        except SyntaxError as e:
            raise ValueError(
                f"Generated test_code has a syntax error: {e}. "
                "Fix the syntax and return valid Python."
            ) from e
        return v

class ChunkSummary(BaseModel):
    summary: str            = Field(description="Dense technical summary max 150 words")

class ModuleSummaryOutput(BaseModel):
    final_summary: str      = Field(description="Coherent module summary max 300 words")
    public_api:    list[str]= Field(description="Public function/class one-line signatures")

class CrossFileOutput(BaseModel):
    answer:    str          = Field(description="Grounded answer with citations")
    citations: list[str]    = Field(description="file_path:line_range for each claim")

    @field_validator("citations")
    @classmethod
    def must_have_citations(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError(
                "Answer must include at least one citation in 'file_path:line_range' format. "
                "Re-read the snippets and cite specific locations."
            )
        return v


# ─────────────────────────────────────────────────────────────────────────────
# 2.  instructor-based retry wrapper
#     This is the production-grade dspy.Assert replacement.
#     instructor injects Pydantic ValidationError text back into the prompt
#     and retries up to max_retries times — same mechanism, explicit code.
# ─────────────────────────────────────────────────────────────────────────────

def call_with_retry(
    model_cls: type[BaseModel],
    messages:  list[dict],
    max_retries: int = 3,
) -> BaseModel:
    """
    Call the LLM and retry with validation-error feedback via instructor.

    If instructor is unavailable, falls back to LangChain with_structured_output
    (single attempt, no auto-retry with feedback).

    This replaces:
        dspy.Assert(condition, feedback)   — instructor does this automatically
        dspy.Suggest(condition, feedback)  — use max_retries=1 for soft version
    """
    if USE_INSTRUCTOR:
        return instructor_client.chat.completions.create(
            response_model=model_cls,
            messages=messages,
            max_retries=max_retries,
        )
    # Fallback: LangChain structured output (no automatic retry with feedback)
    structured = agent_llm.with_structured_output(model_cls)
    lc_messages = [
        HumanMessage(content=m["content"]) if m["role"] == "user"
        else SystemMessage(content=m["content"])
        for m in messages
    ]
    return structured.invoke(lc_messages)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Gateway — identical to DSPy version, just explicit Pydantic
# ─────────────────────────────────────────────────────────────────────────────

GATEWAY_SYSTEM = (
    "You are a code assistant gateway. Given a user question and the repo file tree, decide:\n"
    "1. task_type   — what kind of task this is\n"
    "2. file_scope  — the MINIMUM set of files needed (max 5, empty for cross_file_reason)\n"
    "3. sub_query   — a refined, self-contained question for the sub-agent\n\n"
    "Rules:\n"
    "- NEVER include files that are clearly irrelevant.\n"
    "- For explain_code / find_bug → scope = the specific file mentioned.\n"
    "- For write_test → scope = the file containing the function.\n"
    "- For summarise_module → scope = the module's main file.\n"
    "- For cross_file_reason → scope = [] (sub-agent uses RAG).\n"
)

gateway_structured = gateway_llm.with_structured_output(GatewayDecision)

def gateway_route(user_query: str, file_tree: str) -> GatewayDecision:
    return gateway_structured.invoke([
        SystemMessage(content=GATEWAY_SYSTEM),
        HumanMessage(content=f"Repo file tree:\n{file_tree}\n\nUser question: {user_query}"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 4.  File loader — same as DSPy version
# ─────────────────────────────────────────────────────────────────────────────

def load_scoped_files(
    file_paths: list[str],
    repo_root: Path,
    max_tokens_per_file: int = MAX_FILE_TOKENS,
) -> list[str]:
    loaded = []
    for rel_path in file_paths:
        abs_path = repo_root / rel_path
        if not abs_path.exists():
            loaded.append(f"# [{rel_path}] — file not found")
            continue
        content = abs_path.read_text(encoding="utf-8", errors="replace")
        max_chars = max_tokens_per_file * 4
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n... [truncated — {rel_path}]"
        loaded.append(f"# === {rel_path} ===\n{content}")
    return loaded

def get_file_tree(repo_root: Path, max_files: int = 2000) -> str:
    paths = sorted(
        str(p.relative_to(repo_root))
        for p in repo_root.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") or part in ("__pycache__", "node_modules")
                    for part in p.parts)
    )[:max_files]
    return "\n".join(paths)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Sub-agents — plain Python classes with explicit prompt templates
#     Replace: dspy.ChainOfThought(Signature)
#     With:    ChatPromptTemplate | agent_llm  (LangChain LCEL chain)
#              + call_with_retry for validation/assert behaviour
#
#     THE HARD TRUTH: these prompts must now be written and maintained BY HAND.
#     There is no optimizer. If quality drops, you edit the prompt and redeploy.
# ─────────────────────────────────────────────────────────────────────────────

# ── 5a. CodeExplainer ────────────────────────────────────────────────────────

EXPLAIN_SYSTEM = """\
You are an expert code reviewer. Explain what the provided code does.
Cover: purpose, key functions/classes, data flow, side-effects, gotchas.
Be precise and technical. Use the exact names that appear in the code.
"""

def run_explainer(question: str, code_files: list[str]) -> ExplainOutput:
    code_text = "\n\n".join(code_files)
    return call_with_retry(
        ExplainOutput,
        messages=[
            {"role": "system",  "content": EXPLAIN_SYSTEM},
            {"role": "user",    "content": f"Question: {question}\n\nCode:\n{code_text}"},
        ],
    )


# ── 5b. BugFinder ────────────────────────────────────────────────────────────
# The @field_validator on BugOutput.findings IS the dspy.Assert equivalent.
# instructor injects the ValueError message back and retries automatically.

BUG_SYSTEM = """\
You are a security and correctness code auditor.
Find bugs, logic errors, off-by-one errors, null/zero-division risks,
race conditions, or security issues.

CRITICAL: Every finding MUST start with "Line N:" citing the exact line number.
If there are no bugs, return an empty findings list with severity="none".
"""

def run_bug_finder(
    question: str, code_files: list[str], git_diff: str = ""
) -> BugOutput:
    code_text = "\n\n".join(code_files)
    diff_text = f"\n\nGit diff:\n{git_diff}" if git_diff else ""
    return call_with_retry(
        BugOutput,
        messages=[
            {"role": "system",  "content": BUG_SYSTEM},
            {"role": "user",    "content": f"Question: {question}\n\nCode:\n{code_text}{diff_text}"},
        ],
        max_retries=3,   # retry up to 3× if line-citation validator fires
    )


# ── 5c. TestWriter ───────────────────────────────────────────────────────────
# @field_validator on TestOutput.test_code checks syntax — instructor retries.

TEST_SYSTEM = """\
You are a test engineering expert. Write pytest unit tests.
Rules:
- Only import stdlib, pytest, and the module under test.
- Use parametrize for edge cases.
- Every test function must have a docstring.
- Return a COMPLETE, RUNNABLE test file — no placeholders or ellipsis.
"""

def run_test_writer(question: str, code_files: list[str]) -> TestOutput:
    code_text = "\n\n".join(code_files)
    return call_with_retry(
        TestOutput,
        messages=[
            {"role": "system",  "content": TEST_SYSTEM},
            {"role": "user",    "content": f"Write tests for: {question}\n\nSource:\n{code_text}"},
        ],
        max_retries=3,   # syntax validator retries on SyntaxError
    )


# ── 5d. ModuleSummariser — recursive chunked compression ─────────────────────
# Replaces: dspy.ChainOfThought(ChunkSummarySignature) + MergeSummarySignature
# Same algorithm, explicit prompt strings instead of dspy.Signature docstrings.

CHUNK_SYSTEM = """\
Summarise this code chunk. Preserve: class/function names, parameters,
return types, side-effects, raised exceptions, important constants.
Max 150 words. Be dense and technical.
"""

MERGE_SYSTEM = """\
You are given a list of chunk summaries from a Python module.
Merge them into one coherent module summary (max 300 words) and
list the public API functions/classes with their one-line signatures.
"""

def run_summariser(module_name: str, code_files: list[str]) -> ModuleSummaryOutput:
    chunk_summaries: list[str] = []

    for file_content in code_files:
        lines = file_content.splitlines()
        chunks = [
            "\n".join(lines[i : i + CHUNK_LINES])
            for i in range(0, len(lines), CHUNK_LINES)
        ]
        for chunk in chunks:
            result = call_with_retry(
                ChunkSummary,
                messages=[
                    {"role": "system",  "content": CHUNK_SYSTEM},
                    {"role": "user",    "content": chunk},
                ],
            )
            chunk_summaries.append(result.summary)

    return call_with_retry(
        ModuleSummaryOutput,
        messages=[
            {"role": "system",  "content": MERGE_SYSTEM},
            {"role": "user",    "content": (
                f"Module: {module_name}\n\n"
                "Chunk summaries:\n" +
                "\n---\n".join(chunk_summaries)
            )},
        ],
    )


# ── 5e. CrossFileReasoner — BM25 retrieval + grounded answer ─────────────────

CROSS_SYSTEM = """\
Answer the question using ONLY the retrieved code snippets below.
Cite EVERY claim with "file_path:line_range" format.
If the snippets are insufficient, say so explicitly — do not hallucinate.
"""

_cross_file_retriever: BM25Retriever | None = None

def _get_retriever(repo_root: Path) -> BM25Retriever:
    global _cross_file_retriever
    if _cross_file_retriever:
        return _cross_file_retriever
    docs: list[Document] = []
    for py_file in repo_root.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = code.splitlines()
        for start in range(0, len(lines), 60):
            end = min(start + 80, len(lines))
            docs.append(Document(
                page_content="\n".join(lines[start:end]),
                metadata={"source": str(py_file.relative_to(repo_root)),
                          "start_line": start + 1, "end_line": end},
            ))
    _cross_file_retriever = BM25Retriever.from_documents(docs, k=6)
    return _cross_file_retriever

def run_cross_file_reasoner(question: str, repo_root: Path) -> CrossFileOutput:
    retriever = _get_retriever(repo_root)
    docs = retriever.invoke(question)
    snippets = "\n\n".join(
        f"# {d.metadata['source']} "
        f"lines {d.metadata['start_line']}-{d.metadata['end_line']}\n{d.page_content}"
        for d in docs
    )
    return call_with_retry(
        CrossFileOutput,
        messages=[
            {"role": "system",  "content": CROSS_SYSTEM},
            {"role": "user",    "content": f"Question: {question}\n\nSnippets:\n{snippets}"},
        ],
        max_retries=3,   # citations validator retries if empty
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LangGraph  —  state machine + checkpoint memory
#
#  This is the biggest structural addition vs the DSPy version.
#  LangGraph gives you:
#    - Persistent state across sessions (SqliteSaver / AsyncPostgresSaver)
#    - Resume interrupted pipelines
#    - Full audit trail of every node execution
#    - Human-in-the-loop pause points
#    - Streaming intermediate outputs
#
#  The DSPy version had none of this — it was a simple function call.
#  LangGraph checkpoint memory is a genuine UPGRADE over DSPy here.
# ─────────────────────────────────────────────────────────────────────────────

class GatewayState(TypedDict):
    # Input
    user_query:   str
    repo_root:    str
    git_diff:     str

    # Gateway output
    task_type:    str
    file_scope:   list[str]
    sub_query:    str

    # Sub-agent output
    result_text:        str
    result_citations:   list[str]
    result_key_symbols: list[str]

    # Message history (persisted in checkpoint)
    messages: Annotated[list, add_messages]


def node_gateway(state: GatewayState) -> dict:
    """Node 1: Route the query — reads file tree only (~8k tokens max)."""
    file_tree = get_file_tree(Path(state["repo_root"]))
    decision  = gateway_route(state["user_query"], file_tree)
    return {
        "task_type":  decision.task_type,
        "file_scope": decision.file_scope,
        "sub_query":  decision.sub_query,
        "messages":   [AIMessage(content=(
            f"[Gateway] task={decision.task_type} "
            f"scope={decision.file_scope} reason={decision.reasoning}"
        ))],
    }


def node_load_files(state: GatewayState) -> dict:
    """Node 2: Load ONLY the scoped files into state — never the full repo."""
    if state["task_type"] == "cross_file_reason":
        return {"messages": [AIMessage(content="[Loader] cross_file_reason — skipping file load")]}
    files = load_scoped_files(state["file_scope"], Path(state["repo_root"]))
    # Store file contents as a message so they're in the checkpoint
    file_summary = f"[Loader] Loaded {len(files)} file(s): {state['file_scope']}"
    return {
        "messages": [AIMessage(content=file_summary)],
        # Pass files via state extension (add to dict directly)
        "_loaded_files": files,    # ephemeral — not in TypedDict, used in next node
    }


def node_sub_agent(state: GatewayState) -> dict:
    """Node 3: Run the appropriate sub-agent with the scoped context."""
    task     = state["task_type"]
    query    = state["sub_query"]
    root     = Path(state["repo_root"])
    git_diff = state.get("git_diff", "")

    # Retrieve loaded files from message history (last loader message carries them)
    # In production: store in a dedicated state field with Optional[list[str]]
    loaded: list[str] = state.get("_loaded_files", [])

    if task == "explain_code":
        out = run_explainer(query, loaded)
        return {
            "result_text":        out.explanation,
            "result_key_symbols": out.key_symbols,
            "messages": [AIMessage(content=f"[Explainer] {out.explanation[:200]}...")],
        }

    elif task == "find_bug":
        out = run_bug_finder(query, loaded, git_diff)
        findings_text = "\n".join(out.findings) if out.findings else "No bugs found."
        return {
            "result_text":      findings_text,
            "result_citations": out.findings,
            "messages": [AIMessage(content=f"[BugFinder] severity={out.severity}\n{findings_text}")],
        }

    elif task == "write_test":
        out = run_test_writer(query, loaded)
        return {
            "result_text":        out.test_code,
            "result_key_symbols": out.test_cases,
            "messages": [AIMessage(content=f"[TestWriter] {len(out.test_cases)} test cases")],
        }

    elif task == "summarise_module":
        out = run_summariser(query, loaded)
        return {
            "result_text":        out.final_summary,
            "result_key_symbols": out.public_api,
            "messages": [AIMessage(content=f"[Summariser] {out.final_summary[:200]}...")],
        }

    elif task == "cross_file_reason":
        out = run_cross_file_reasoner(query, root)
        return {
            "result_text":      out.answer,
            "result_citations": out.citations,
            "messages": [AIMessage(content=f"[CrossFile] {out.answer[:200]}...")],
        }

    return {"result_text": "Unknown task type", "messages": []}


def build_graph(checkpoint_db: str = "gateway_checkpoints.db") -> tuple:
    """
    Build and compile the LangGraph with SqliteSaver checkpoint memory.

    SqliteSaver persists the full state after every node.
    Switch to AsyncPostgresSaver for production:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        saver = AsyncPostgresSaver.from_conn_string("postgresql://...")

    The checkpoint gives you:
    - Resume an interrupted pipeline with the same thread_id
    - Full audit log of every node's input/output
    - Multi-session memory: same user, different queries, shared history
    """
    graph = StateGraph(GatewayState)
    graph.add_node("gateway",    node_gateway)
    graph.add_node("load_files", node_load_files)
    graph.add_node("sub_agent",  node_sub_agent)

    graph.add_edge(START,        "gateway")
    graph.add_edge("gateway",    "load_files")
    graph.add_edge("load_files", "sub_agent")
    graph.add_edge("sub_agent",  END)

    # SqliteSaver: checkpoint every node's state to a local SQLite DB
    saver = SqliteSaver.from_conn_string(checkpoint_db)
    compiled = graph.compile(checkpointer=saver)
    return compiled, saver


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

_agent, _saver = build_graph()


def ask(
    user_query: str,
    repo_root:  str | Path = ".",
    git_diff:   str = "",
    session_id: str = "default",   # LangGraph thread-level memory key
    verbose:    bool = False,
) -> dict:
    """
    session_id maps to a LangGraph thread.
    Using the same session_id across calls gives the agent persistent memory:
    it can see all previous queries + responses for that session.

    Different from the DSPy version, which had no memory between calls.
    """
    config = {"configurable": {"thread_id": session_id}}

    initial_state: GatewayState = {
        "user_query": user_query,
        "repo_root":  str(repo_root),
        "git_diff":   git_diff,
        "task_type":  "",
        "file_scope": [],
        "sub_query":  "",
        "result_text":        "",
        "result_citations":   [],
        "result_key_symbols": [],
        "messages": [HumanMessage(content=user_query)],
    }

    final = _agent.invoke(initial_state, config=config)

    if verbose:
        print(f"[Gateway] task={final['task_type']}  scope={final['file_scope']}")

    return {
        "task_type":    final["task_type"],
        "file_scope":   final["file_scope"],
        "result":       final["result_text"],
        "citations":    final.get("result_citations", []),
        "key_symbols":  final.get("result_key_symbols", []),
        "session_id":   session_id,
    }


def get_session_history(session_id: str) -> list[dict]:
    """
    Retrieve all checkpointed states for a session.
    This is checkpoint memory — inspect exactly what happened at every node.
    Equivalent to LangChain's ConversationHistory but persisted and auditable.
    """
    config = {"configurable": {"thread_id": session_id}}
    history = []
    for state in _agent.get_state_history(config):
        history.append({
            "step":      state.metadata.get("step"),
            "node":      state.metadata.get("source"),
            "task_type": state.values.get("task_type"),
            "result":    state.values.get("result_text", "")[:100],
        })
    return history


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo = Path(".")

    queries = [
        ("Explain what reciprocal_rank_fusion does", "session_alice"),
        ("Find bugs in the calculator tool",          "session_alice"),  # same session → has memory of prev query
        ("Write tests for hybrid_retrieve",           "session_bob"),
        ("Summarise the dspy_retrieve_and_memory module", "session_bob"),
        ("How does LangGraph state flow connect to DSPy?","session_alice"),
    ]

    for q, sid in queries:
        print("\n" + "=" * 60)
        print(f"QUERY [{sid}]: {q}")
        out = ask(q, repo_root=repo, session_id=sid, verbose=True)
        print(f"TASK:   {out['task_type']}")
        print(f"SCOPE:  {out['file_scope']}")
        print(f"RESULT: {out['result'][:400]}")
        if out["citations"]:
            print(f"CITED:  {out['citations'][:3]}")

    # Inspect checkpoint memory for a session
    print("\n--- Session history for session_alice ---")
    for entry in get_session_history("session_alice"):
        print(entry)
