import sys
from typing import Optional, Literal, Union

import mlflow
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel

mlflow.langchain.autolog()
# ── LLM setup (same as notebook) ──────────────────────────────────────────
PROD_ROOT = "/Users/s748779/CEP_AI/anzsic_mapping_v1"
if PROD_ROOT not in sys.path:
    sys.path.insert(0, PROD_ROOT)

from prod.adapters.gcp_auth import GCPAuthManager
from prod.adapters.gemini_langchain_llm import GeminiLangChainLLMAdapter
from prod.config.settings import get_settings

settings   = get_settings()
auth       = GCPAuthManager(settings)
adapter    = GeminiLangChainLLMAdapter(settings, auth=auth)
vertex_llm = adapter.get_raw_llm()

# ── Prompts ────────────────────────────────────────────────────────────────
PROMPTS = {
    "RESEARCH": (
        "Conduct thorough research on this query and provide fodder points "
        "to be used by the writer agent."
    ),
    "WRITER": (
        "Write a brief write-up on the shared query with help of fodder "
        "points provided."
    ),
    "SUPERVISOR": (
        "Look at the output in the state and decide if the next step should "
        "be to direct to:\n"
        "  1. ResearchAgent - if the topic needs research or writeup needs improvement.\n"
        "  2. WriterAgent   - if the research content seems enough to write on the given topic.\n"
        "  3. FINISH        - if previous output by WriterAgent is satisfactory.\n"
        "Strictly output text content either 'ResearchAgent' OR 'WriterAgent' OR 'FINISH'."
    ),
}

# ── State & structured output ──────────────────────────────────────────────
class SupervisorState(MessagesState):
    next_call: Optional[str] = None

class SupervisorOutput(BaseModel):
    next_call: Literal["ResearchAgent", "WriterAgent", "FINISH"]

# ── Helper ────────────────────────────────────────────────────────────────
def _as_text(msg_content: Union[str, list]) -> str:
    if isinstance(msg_content, str):
        return msg_content
    if isinstance(msg_content, list):
        for part in msg_content:
            if isinstance(part, dict) and part.get("type") == "text":
                return part.get("text") or part.get("content") or ""
    return ""

# ── Nodes ─────────────────────────────────────────────────────────────────
def research_node(state: SupervisorState):
    query = state["messages"][0].content
    result = vertex_llm.invoke([
        SystemMessage(content=PROMPTS["RESEARCH"]),
        HumanMessage(content=query),
    ])
    return {"messages": result}

def writer_node(state: SupervisorState):
    research_content = _as_text(state["messages"][-1].content)
    result = vertex_llm.invoke([
        SystemMessage(content=PROMPTS["WRITER"]),
        HumanMessage(content=f"Use the following research fodder to write:\n\n{research_content}"),
    ])
    return {"messages": result}

def supervisor_node(state: SupervisorState):
    history = "\n\n".join(
        f"[{m.type.upper()}]: {m.content}" for m in state["messages"]
    )
    supervisor_llm = vertex_llm.with_structured_output(SupervisorOutput)
    result = supervisor_llm.invoke([
        SystemMessage(content=PROMPTS["SUPERVISOR"]),
        HumanMessage(content=f"Conversation so far:\n\n{history}\n\nWhat should be the next step?"),
    ])
    return {"next_call": result.next_call}

def router_edge(state: SupervisorState):
    mapping = {
        "ResearchAgent": "research_node",
        "WriterAgent":   "writer_node",
        "FINISH":        END,
    }
    return mapping[state["next_call"]]

# ── Build graph ───────────────────────────────────────────────────────────
builder = StateGraph(SupervisorState)
builder.add_node("research_node",  research_node)
builder.add_node("writer_node",    writer_node)
builder.add_node("supervisor_node", supervisor_node)
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", router_edge)
builder.add_edge("research_node", "supervisor_node")
builder.add_edge("writer_node",   "supervisor_node")
graph = builder.compile()

# ── Required by MLflow models-from-code ───────────────────────────────────
mlflow.models.set_model(graph)
