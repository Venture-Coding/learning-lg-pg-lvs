"""
langgraph_autogen_langchain.py
===============================
LangChain + LangGraph + AutoGen — one file, optimal division of labour.

The only honest rule for which framework to use where:
  LangChain → structured I/O, tool binding, RAG, LCEL chains
  LangGraph  → state machine, conditional routing, checkpoint memory
  AutoGen    → ONLY for its two genuine killer features:
                 1. Code generation + ACTUAL execution (write → run → fix loop)
                 2. Multi-agent debate (agents critique each other until consensus)
               Everything else: don't use AutoGen, use LangChain.

Architecture
------------
LangGraph manages the outer state machine and persistence.
LangChain handles the gateway routing and RAG sub-agent.
AutoGen handles two specific node types:
  - autogen_code_executor : AssistantAgent writes code, UserProxyAgent runs it,
                            loop continues until tests pass or max_turns reached.
                            This is AutoGen's primary value — no other framework
                            actually executes code and feeds stdout back.
  - autogen_debate        : Multiple specialised AssistantAgents argue, critique,
                            and refine until they reach consensus. Useful for
                            architecture decisions, security reviews, trade-off analysis.

Task routing
------------
  explain_code       → LangChain LCEL (no AutoGen needed)
  summarise_module   → LangChain LCEL (no AutoGen needed)
  cross_file_reason  → LangChain RAG  (no AutoGen needed)
  write_code         → AutoGen code executor  ← actual execution
  fix_bug            → AutoGen code executor  ← actual execution + test runner
  run_analysis       → AutoGen code executor  ← runs data analysis scripts
  architecture_review→ AutoGen debate         ← multi-agent critique
  security_review    → AutoGen debate         ← multi-agent critique

requirements (add to requirements.txt):
  pyautogen>=0.9.0        (AG2 — the maintained AutoGen fork)
  langgraph>=0.2.0
  langgraph[sqlite]       (checkpoint memory)
  langchain>=0.3.0
  langchain-google-vertexai>=2.0.0
  langchain-community>=0.3.0
  pydantic>=2.0.0
"""

from __future__ import annotations

import ast
import json
import os
import textwrap
from pathlib import Path
from typing import Annotated, Any, Literal

# ── AutoGen (AG2 fork — pip install pyautogen) ───────────────────────────────
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# ── LangChain ────────────────────────────────────────────────────────────────
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from pydantic import BaseModel, Field

# ── LangGraph ─────────────────────────────────────────────────────────────────
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# 1.  Config
# ─────────────────────────────────────────────────────────────────────────────

GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "my-project")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ID     = "gemini-2.0-flash"

# LangChain LLM (for gateway + RAG nodes)
lc_llm = ChatVertexAI(
    model_name=MODEL_ID, project=GCP_PROJECT,
    location=GCP_LOCATION, temperature=0.0, max_output_tokens=4096,
)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=GCP_PROJECT, location=GCP_LOCATION,
)

# AutoGen LLM config — AutoGen communicates with the model via its own client.
# Use litellm "vertex_ai/" prefix so AutoGen and LangChain hit the same endpoint.
# AutoGen needs an explicit list format for its config_list.
AUTOGEN_LLM_CONFIG: dict = {
    "config_list": [
        {
            "model":  f"vertex_ai/{MODEL_ID}",
            "api_type": "litellm",           # AutoGen 0.9+ supports litellm natively
            # Vertex AI credentials picked up from GOOGLE_APPLICATION_CREDENTIALS env var
        }
    ],
    "temperature": 0.1,
    "timeout":     120,
}

# Code execution sandbox directory — UserProxyAgent writes and runs files here
CODE_SANDBOX = Path("./autogen_sandbox")
CODE_SANDBOX.mkdir(exist_ok=True)

MAX_FILE_TOKENS  = 6000
CHUNK_LINES      = 100


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LangGraph State
# ─────────────────────────────────────────────────────────────────────────────

TaskType = Literal[
    "explain_code",
    "summarise_module",
    "cross_file_reason",
    "write_code",          # → AutoGen code executor
    "fix_bug",             # → AutoGen code executor
    "run_analysis",        # → AutoGen code executor
    "architecture_review", # → AutoGen debate
    "security_review",     # → AutoGen debate
]

class AgentState(TypedDict):
    user_query:      str
    repo_root:       str
    task_type:       str
    file_scope:      list[str]
    sub_query:       str
    loaded_files:    list[str]   # file contents for scoped files
    result:          str         # final answer / generated code
    citations:       list[str]
    autogen_transcript: list[str]  # full AutoGen conversation log
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Gateway — LangChain with_structured_output (NOT AutoGen)
#     AutoGen adds zero value here; this is a classification task.
# ─────────────────────────────────────────────────────────────────────────────

class GatewayDecision(BaseModel):
    task_type:  TaskType  = Field(description="The type of task")
    file_scope: list[str] = Field(description="Minimal file paths needed, max 5. Empty for cross_file_reason.")
    sub_query:  str       = Field(description="Refined self-contained question/instruction for the sub-agent")
    reasoning:  str       = Field(description="One sentence explaining routing choice")

GATEWAY_SYSTEM = """\
You are a code assistant gateway. Given a user request and a repo file tree, decide:
  task_type:
    explain_code        — user wants to understand existing code
    summarise_module    — user wants an overview of a module/package
    cross_file_reason   — question spans multiple files, needs RAG search
    write_code          — user wants NEW code written and tested
    fix_bug             — user wants a bug fixed and verified (runs tests)
    run_analysis        — user wants a script run against data
    architecture_review — user wants design/architecture critique (multi-agent)
    security_review     — user wants security analysis (multi-agent)

  file_scope: minimal file paths. Empty list for cross_file_reason, write_code.
  sub_query:  precise, self-contained instruction for the sub-agent.
"""

_gateway = lc_llm.with_structured_output(GatewayDecision)

def node_gateway(state: AgentState) -> dict:
    """Read file tree only (~5-10k tokens). Never load file contents here."""
    root      = Path(state["repo_root"])
    file_tree = "\n".join(
        str(p.relative_to(root))
        for p in sorted(root.rglob("*"))
        if p.is_file()
        and not any(part.startswith(".") or part in ("__pycache__", "node_modules")
                    for part in p.parts)
    )[:8000]

    decision: GatewayDecision = _gateway.invoke([
        SystemMessage(content=GATEWAY_SYSTEM),
        HumanMessage(content=f"Repo:\n{file_tree}\n\nRequest: {state['user_query']}"),
    ])
    return {
        "task_type":  decision.task_type,
        "file_scope": decision.file_scope,
        "sub_query":  decision.sub_query,
        "messages":   [AIMessage(content=(
            f"[Gateway] task={decision.task_type} "
            f"scope={decision.file_scope}\nreason={decision.reasoning}"
        ))],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 4.  File loader — scoped, token-budgeted
# ─────────────────────────────────────────────────────────────────────────────

def _load_files(file_scope: list[str], repo_root: Path) -> list[str]:
    out = []
    for rel in file_scope:
        p = repo_root / rel
        if not p.exists():
            out.append(f"# [{rel}] not found")
            continue
        content = p.read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_FILE_TOKENS * 4:
            content = content[: MAX_FILE_TOKENS * 4] + "\n# ... truncated"
        out.append(f"# === {rel} ===\n{content}")
    return out

def node_load_files(state: AgentState) -> dict:
    if state["task_type"] in ("cross_file_reason", "write_code", "run_analysis",
                               "architecture_review", "security_review"):
        return {"loaded_files": [], "messages": [AIMessage(content="[Loader] skipped")]}
    files = _load_files(state["file_scope"], Path(state["repo_root"]))
    return {
        "loaded_files": files,
        "messages": [AIMessage(content=f"[Loader] {len(files)} file(s) loaded")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5.  LangChain sub-agents (for tasks that don't need execution or debate)
#     Plain LCEL chains — transparent, fast, no AutoGen overhead.
# ─────────────────────────────────────────────────────────────────────────────

# 5a. Explain / Summarise — LCEL chain (prompt | llm | str parser)
_explain_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "You are an expert code reviewer. Explain the code precisely and technically. "
                   "Cover: purpose, data flow, key symbols, side-effects, gotchas."),
        ("human",  "Question: {question}\n\nCode:\n{code}"),
    ])
    | lc_llm
    | StrOutputParser()
)

def node_explain(state: AgentState) -> dict:
    answer = _explain_chain.invoke({
        "question": state["sub_query"],
        "code":     "\n\n".join(state["loaded_files"]) or "(no files loaded)",
    })
    return {
        "result":   answer,
        "messages": [AIMessage(content=f"[Explain] {answer[:200]}...")],
    }


# 5b. Cross-file RAG — BM25 retrieval + LCEL generation
_cross_chain = (
    ChatPromptTemplate.from_messages([
        ("system", "Answer using ONLY the provided code snippets. "
                   "Cite every claim as 'file:line_range'. Say 'insufficient context' if unsure."),
        ("human",  "Question: {question}\n\nSnippets:\n{snippets}"),
    ])
    | lc_llm
    | StrOutputParser()
)

_bm25_retriever: BM25Retriever | None = None

def _get_bm25(repo_root: Path) -> BM25Retriever:
    global _bm25_retriever
    if _bm25_retriever:
        return _bm25_retriever
    docs: list[Document] = []
    for f in repo_root.rglob("*.py"):
        try:
            lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
        except Exception:
            continue
        for start in range(0, len(lines), 60):
            end = min(start + 80, len(lines))
            docs.append(Document(
                page_content="\n".join(lines[start:end]),
                metadata={"source": str(f.relative_to(repo_root)),
                          "start_line": start + 1, "end_line": end},
            ))
    _bm25_retriever = BM25Retriever.from_documents(docs, k=6)
    return _bm25_retriever

def node_cross_file(state: AgentState) -> dict:
    retriever = _get_bm25(Path(state["repo_root"]))
    docs = retriever.invoke(state["sub_query"])
    snippets = "\n\n".join(
        f"# {d.metadata['source']} L{d.metadata['start_line']}-{d.metadata['end_line']}\n"
        f"{d.page_content}"
        for d in docs
    )
    answer = _cross_chain.invoke({"question": state["sub_query"], "snippets": snippets})
    citations = [f"{d.metadata['source']}:{d.metadata['start_line']}-{d.metadata['end_line']}"
                 for d in docs]
    return {
        "result":    answer,
        "citations": citations,
        "messages":  [AIMessage(content=f"[CrossFile] {answer[:200]}...")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  AutoGen: Code Executor node  ⭐ Killer feature #1
#
#  This is why you use AutoGen:
#    AssistantAgent writes Python code in a markdown code block.
#    UserProxyAgent extracts it, runs it in a subprocess sandbox,
#    captures stdout/stderr/exit-code, and sends it back.
#    AssistantAgent sees the output and fixes errors.
#    Loop continues until all tests pass OR max_turns is reached.
#
#  No other framework in this stack actually EXECUTES code.
#  LangChain tools can call pre-defined functions, but they cannot
#  run arbitrary generated code and feed stdout back to the LLM.
#
#  Use cases: write_code, fix_bug, run_analysis
# ─────────────────────────────────────────────────────────────────────────────

def _build_code_executor(task_description: str) -> tuple[AssistantAgent, UserProxyAgent]:
    """
    Create a fresh AssistantAgent + UserProxyAgent pair per invocation.
    Fresh pair = clean conversation history = no cross-contamination between tasks.
    """
    assistant = AssistantAgent(
        name="CodeWriter",
        llm_config=AUTOGEN_LLM_CONFIG,
        system_message=textwrap.dedent("""\
            You are an expert Python engineer.
            When asked to write or fix code:
            1. Write complete, runnable Python in a ```python block.
            2. Include pytest tests in the same block or a separate ```python block clearly marked # TESTS.
            3. After the execution result is shown, if there are errors, fix them.
            4. When all tests pass, end your message with exactly: TASK_COMPLETED
            5. Never leave placeholder comments like '# TODO' or '...'.
        """),
    )

    # UserProxyAgent: the agent that actually executes code.
    # human_input_mode="NEVER" → fully autonomous (no human confirmation needed).
    # code_execution_config → where and how to run the code.
    # is_termination_msg → stops the loop when CodeWriter says TASK_COMPLETED.
    proxy = UserProxyAgent(
        name="CodeRunner",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=8,    # max iterations before giving up
        is_termination_msg=lambda msg: "TASK_COMPLETED" in (msg.get("content") or ""),
        code_execution_config={
            "work_dir":        str(CODE_SANDBOX),
            "use_docker":      False,    # set True for production sandboxing
            "timeout":         30,       # seconds per execution
            "last_n_messages": 3,        # re-run only if recent messages have code
        },
        system_message=(
            "You run the code written by CodeWriter. "
            "Report exact stdout, stderr, and exit code. "
            "Do not modify the code yourself."
        ),
    )
    return assistant, proxy


def node_autogen_code_executor(state: AgentState) -> dict:
    """
    AutoGen code executor node.

    The full conversation loop (write → run → fix → run → ...) happens
    inside this single LangGraph node. LangGraph sees it as one atomic step.
    The transcript is stored in state for audit/debugging.
    """
    context_block = ""
    if state.get("loaded_files"):
        context_block = "\n\nRelevant code context:\n" + "\n\n".join(state["loaded_files"])

    task_message = f"{state['sub_query']}{context_block}"

    assistant, proxy = _build_code_executor(task_message)

    # initiate_chat starts the autonomous conversation loop.
    # It blocks until termination condition is met or max_turns reached.
    chat_result = proxy.initiate_chat(
        recipient=assistant,
        message=task_message,
        summary_method="last_msg",   # use last message as summary
    )

    # Extract the last assistant message as the result
    transcript = [
        f"[{msg.get('name', 'agent')}]: {msg.get('content', '')}"
        for msg in chat_result.chat_history
    ]
    final_code = chat_result.summary or ""

    return {
        "result":               final_code,
        "autogen_transcript":   transcript,
        "messages": [AIMessage(content=(
            f"[AutoGen:CodeExecutor] {len(transcript)} turns. "
            f"Result preview: {final_code[:200]}..."
        ))],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 7.  AutoGen: Multi-agent Debate node  ⭐ Killer feature #2
#
#  Three specialised agents are placed in a GroupChat.
#  They take turns responding — each critiques the previous response from
#  their own specialist perspective.
#  GroupChatManager runs the round-robin and synthesises a final consensus.
#
#  Use cases: architecture_review, security_review
#
#  Plain LangChain/LangGraph cannot replicate this natively:
#  you'd have to manually chain N LLM calls and pass outputs forward.
#  AutoGen's GroupChat handles the turn management and termination automatically.
# ─────────────────────────────────────────────────────────────────────────────

def _build_debate_agents(task_type: str) -> tuple[list[AssistantAgent], UserProxyAgent]:
    """
    Build specialist debate agents based on task type.
    Each agent has a fixed perspective — this forces adversarial critique.
    """
    if task_type == "architecture_review":
        agents = [
            AssistantAgent(
                name="Architect",
                llm_config=AUTOGEN_LLM_CONFIG,
                system_message=(
                    "You are a senior software architect. Evaluate designs for: "
                    "scalability, separation of concerns, coupling, extensibility. "
                    "Be constructive but direct about weaknesses."
                ),
            ),
            AssistantAgent(
                name="PerformanceEngineer",
                llm_config=AUTOGEN_LLM_CONFIG,
                system_message=(
                    "You are a performance engineer. Evaluate for: "
                    "latency, throughput, memory usage, caching strategy, "
                    "database query patterns. Cite specific bottlenecks."
                ),
            ),
            AssistantAgent(
                name="Pragmatist",
                llm_config=AUTOGEN_LLM_CONFIG,
                system_message=(
                    "You are a pragmatic senior engineer. Push back on over-engineering. "
                    "Ask: is this the simplest solution that works? What is the maintenance burden? "
                    "Summarise the final consensus recommendation at the end with 'CONSENSUS:'."
                ),
            ),
        ]
    else:  # security_review
        agents = [
            AssistantAgent(
                name="SecurityAnalyst",
                llm_config=AUTOGEN_LLM_CONFIG,
                system_message=(
                    "You are an application security expert (OWASP, NIST). "
                    "Identify injection risks, auth flaws, data exposure, "
                    "insecure dependencies. Cite CWE numbers where applicable."
                ),
            ),
            AssistantAgent(
                name="Attacker",
                llm_config=AUTOGEN_LLM_CONFIG,
                system_message=(
                    "You think like an attacker. Given the code, describe concrete "
                    "exploit scenarios: exact input sequences, privilege escalation paths, "
                    "data exfiltration routes. Be specific, not generic."
                ),
            ),
            AssistantAgent(
                name="SecurityReviewer",
                llm_config=AUTOGEN_LLM_CONFIG,
                system_message=(
                    "You are a defensive security reviewer. After hearing the attack scenarios, "
                    "propose concrete mitigations with code snippets. "
                    "Summarise severity + mitigations as 'CONSENSUS:' at the end."
                ),
            ),
        ]

    # Initiator — kicks off the debate, doesn't participate in argument
    initiator = UserProxyAgent(
        name="Initiator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,   # never auto-replies — only starts the conversation
        is_termination_msg=lambda msg: "CONSENSUS:" in (msg.get("content") or ""),
        code_execution_config=False,    # no code execution in debate
    )
    return agents, initiator


def node_autogen_debate(state: AgentState) -> dict:
    """
    AutoGen multi-agent debate node.

    Three specialist agents take turns critiquing the design/code.
    The designated consensus agent ends with 'CONSENSUS:' when it has
    synthesised agreement — GroupChat terminates on this signal.
    """
    context_block = ""
    if state.get("loaded_files"):
        context_block = "\n\nCode / Design under review:\n" + "\n\n".join(state["loaded_files"])

    topic = f"{state['sub_query']}{context_block}"

    debate_agents, initiator = _build_debate_agents(state["task_type"])

    group_chat = GroupChat(
        agents=[initiator] + debate_agents,
        messages=[],
        max_round=9,                     # 3 agents × 3 rounds each
        speaker_selection_method="round_robin",  # deterministic turn order
    )
    manager = GroupChatManager(
        groupchat=group_chat,
        llm_config=AUTOGEN_LLM_CONFIG,
    )

    # Start the debate — blocks until CONSENSUS or max_round
    initiator.initiate_chat(recipient=manager, message=topic)

    # Extract full transcript + consensus
    transcript = [
        f"[{msg.get('name', 'agent')}]: {msg.get('content', '')[:500]}"
        for msg in group_chat.messages
    ]
    # Find the CONSENSUS section in any agent's message
    consensus = ""
    for msg in reversed(group_chat.messages):
        content = msg.get("content", "")
        if "CONSENSUS:" in content:
            consensus = content[content.index("CONSENSUS:") :]
            break

    return {
        "result":             consensus or transcript[-1] if transcript else "No consensus reached.",
        "autogen_transcript": transcript,
        "messages": [AIMessage(content=(
            f"[AutoGen:Debate] {len(transcript)} turns across {len(debate_agents)} agents. "
            f"Consensus: {consensus[:200]}..."
        ))],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LangGraph: wire everything together
# ─────────────────────────────────────────────────────────────────────────────

LANGCHAIN_TASKS = {"explain_code", "summarise_module", "cross_file_reason"}
AUTOGEN_CODE_TASKS = {"write_code", "fix_bug", "run_analysis"}
AUTOGEN_DEBATE_TASKS = {"architecture_review", "security_review"}

def route_after_load(state: AgentState) -> str:
    """Conditional edge: route to correct sub-agent after file loading."""
    t = state["task_type"]
    if t in ("explain_code", "summarise_module"):
        return "explain"
    if t == "cross_file_reason":
        return "cross_file"
    if t in AUTOGEN_CODE_TASKS:
        return "autogen_code"
    if t in AUTOGEN_DEBATE_TASKS:
        return "autogen_debate"
    return "explain"   # safe default


def build_graph(checkpoint_db: str = "agent_checkpoints.db"):
    """
    Compile the LangGraph.

    SqliteSaver gives persistent checkpoint memory:
      - Every node's input/output is saved after execution
      - Resume interrupted pipelines with the same thread_id
      - Multi-session memory: same user, different queries, shared history
      - Full audit trail — inspect every step after the fact
    """
    g = StateGraph(AgentState)

    g.add_node("gateway",       node_gateway)
    g.add_node("load_files",    node_load_files)
    g.add_node("explain",       node_explain)
    g.add_node("cross_file",    node_cross_file)
    g.add_node("autogen_code",  node_autogen_code_executor)
    g.add_node("autogen_debate",node_autogen_debate)

    g.add_edge(START,        "gateway")
    g.add_edge("gateway",    "load_files")

    g.add_conditional_edges(
        "load_files",
        route_after_load,
        {
            "explain":       "explain",
            "cross_file":    "cross_file",
            "autogen_code":  "autogen_code",
            "autogen_debate":"autogen_debate",
        },
    )

    # All sub-agents converge to END
    for node in ("explain", "cross_file", "autogen_code", "autogen_debate"):
        g.add_edge(node, END)

    saver = SqliteSaver.from_conn_string(checkpoint_db)
    return g.compile(checkpointer=saver), saver


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Public API
# ─────────────────────────────────────────────────────────────────────────────

_graph, _checkpointer = build_graph()


def ask(
    query:      str,
    repo_root:  str | Path = ".",
    session_id: str = "default",
    verbose:    bool = False,
) -> dict:
    """
    Run the full pipeline for a query.

    session_id → LangGraph thread_id.
    Same session_id across calls = persistent memory of all prior turns.
    AutoGen sub-agents get a fresh pair per call (no cross-task contamination).
    """
    config = {"configurable": {"thread_id": session_id}}

    state: AgentState = {
        "user_query":         query,
        "repo_root":          str(repo_root),
        "task_type":          "",
        "file_scope":         [],
        "sub_query":          "",
        "loaded_files":       [],
        "result":             "",
        "citations":          [],
        "autogen_transcript": [],
        "messages":           [HumanMessage(content=query)],
    }

    final = _graph.invoke(state, config=config)

    if verbose:
        print(f"\n[task={final['task_type']}  scope={final['file_scope']}]")
        if final.get("autogen_transcript"):
            print(f"[AutoGen: {len(final['autogen_transcript'])} turns]")

    return {
        "task_type":   final["task_type"],
        "file_scope":  final["file_scope"],
        "result":      final["result"],
        "citations":   final.get("citations", []),
        "transcript":  final.get("autogen_transcript", []),
        "session_id":  session_id,
    }


def session_history(session_id: str) -> list[dict]:
    """Retrieve LangGraph checkpoint history for a session."""
    config = {"configurable": {"thread_id": session_id}}
    return [
        {
            "step":      s.metadata.get("step"),
            "node":      s.metadata.get("source"),
            "task_type": s.values.get("task_type"),
            "result":    s.values.get("result", "")[:120],
        }
        for s in _graph.get_state_history(config)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Demo — one query per task type
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    repo = Path(".")

    scenarios = [
        # LangChain LCEL path — fast, no AutoGen overhead
        ("Explain what the reciprocal_rank_fusion function does",             "alice"),
        ("How does the LangGraph state connect to DSPy modules?",             "alice"),

        # AutoGen code executor — writes code, runs it, fixes errors in a loop
        ("Write a function that chunks a list into overlapping windows "
         "with configurable size and stride. Include pytest tests.",          "bob"),
        ("The calculator tool crashes on sqrt expressions. Fix it and "
         "verify all edge cases pass.",                                        "bob"),

        # AutoGen multi-agent debate — adversarial critique
        ("Review the gateway routing architecture for scalability: "
         "could it become a bottleneck at 10k requests/minute?",              "carol"),
        ("Security review: can the code execution sandbox in AutoGen "
         "be exploited for arbitrary file read?",                             "carol"),
    ]

    for query, sid in scenarios:
        print("\n" + "=" * 70)
        print(f"SESSION={sid}  QUERY: {query[:80]}")
        out = ask(query, repo_root=repo, session_id=sid, verbose=True)
        print(f"TASK:   {out['task_type']}")
        print(f"RESULT: {out['result'][:500]}")
        if out["citations"]:
            print(f"CITED:  {out['citations'][:3]}")
        if out["transcript"]:
            print(f"TRANSCRIPT TURNS: {len(out['transcript'])}")
            # Show first and last AutoGen turns
            print(f"  First: {out['transcript'][0][:120]}")
            print(f"  Last:  {out['transcript'][-1][:120]}")

    # Show persistent checkpoint memory for alice's session
    print("\n--- alice session checkpoint history ---")
    for entry in session_history("alice"):
        print(entry)
