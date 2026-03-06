"""
Agentic Hybrid RAG: DSPy + LangGraph + LangChain (ChatVertexAI / ChatGoogleGenerativeAI)
==========================================================================================

Architecture:
  LLM Layer      → ChatVertexAI / ChatGoogleGenerativeAI (Gemini)
  DSPy Layer     → ONLY for HybridRAGGenerate (answer + citations) – where optimisation matters
  LangChain      → Routing / query-analysis / tool-calling via with_structured_output + bind_tools
  Retrieval      → Hybrid: Dense (Chroma + VertexAI Embeddings) + Sparse (BM25) → RRF fusion
  Agent Layer    → LangGraph state machine wiring it all together
  Tools          → LangChain @tool decorators called by the agent when needed

Honest division of labour
--------------------------
- Routing (query analysis, tool selection):
    Use LangChain with_structured_output + Pydantic.
    Gemini's native JSON/function-calling mode is more reliable here.
    DSPy adds nothing unless you run an optimiser.

- Answer generation (HybridRAGGenerate):
    Use DSPy ChainOfThought + BootstrapFewShot / MIPROv2.
    Answer quality is measurable (RAGAS, faithfulness, citation F1).
    The optimiser auto-generates few-shot demos and rewrites field
    descriptions to improve grounding – this is where DSPy earns its place.

- Tool execution:
    Use LangChain bind_tools + native Gemini function calling.
    Purpose-built, schema-validated, no custom dict parsing needed.

Run:
    pip install -r requirements.txt
    export GOOGLE_CLOUD_PROJECT=my-project   # for Vertex AI
    # OR
    export GOOGLE_API_KEY=my-key             # for Google Generative AI (Gemini API)
    python hybrid_rag_agent.py
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Annotated, Any, Optional

from pydantic import BaseModel, Field

# ─── DSPy ────────────────────────────────────────────────────────────────────
import dspy

# ─── LangChain / LangGraph ────────────────────────────────────────────────────
from langchain.tools import tool
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings

# Uncomment for Gemini Developer API (non-Vertex) instead:
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LLM + Embeddings  (choose ONE block below)
# ─────────────────────────────────────────────────────────────────────────────

# --- Option A: Vertex AI (recommended for GCP) --------------------------------
GCP_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT", "my-gcp-project")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ID = "gemini-2.0-flash"

langchain_llm = ChatVertexAI(
    model_name=MODEL_ID,
    project=GCP_PROJECT,
    location=GCP_LOCATION,
    temperature=0.0,
    max_output_tokens=2048,
)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=GCP_PROJECT,
    location=GCP_LOCATION,
)

# --- Option B: Google Generative AI (Gemini Developer API) --------------------
# GOOGLE_API_KEY must be set in env
# langchain_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.0)
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  Configure DSPy to use the same Gemini backend via litellm
#     dspy.LM uses the litellm router under the hood — same model, no duplication
# ─────────────────────────────────────────────────────────────────────────────

# For Vertex AI use the "vertex_ai/" prefix (litellm convention)
dspy_lm = dspy.LM(
    model=f"vertex_ai/{MODEL_ID}",
    # If using Gemini Developer API instead:
    # model=f"gemini/{MODEL_ID}",
    temperature=0.0,
    max_tokens=2048,
    # Vertex AI needs project + location via env or explicit kwargs
    vertex_project=GCP_PROJECT,
    vertex_location=GCP_LOCATION,
)
dspy.configure(lm=dspy_lm)


# ─────────────────────────────────────────────────────────────────────────────
# 3a.  LangChain Pydantic schemas for ROUTING decisions
#      Use with_structured_output – Gemini native JSON/function-calling mode.
#      No DSPy needed here: simple decisions, no training data, no metric to
#      optimise.  with_structured_output is more reliable for strict JSON.
# ─────────────────────────────────────────────────────────────────────────────

class QueryAnalysisOutput(BaseModel):
    """Structured output for query analysis routing."""
    needs_tools: bool = Field(
        description="True if the question requires real-time data, calculations, or external APIs"
    )
    search_queries: list[str] = Field(
        description="1 to 3 retrieval-optimised sub-queries derived from the original question"
    )
    reasoning: str = Field(description="Short reasoning for the decisions above")


# ─────────────────────────────────────────────────────────────────────────────
# 3b.  DSPy Signature ONLY for answer generation
#      This is where optimisation (BootstrapFewShot / MIPROv2) genuinely helps:
#      answer quality, grounding, citation precision are all measurable metrics.
# ─────────────────────────────────────────────────────────────────────────────

class HybridRAGGenerate(dspy.Signature):
    """Generate a grounded, cited answer using the provided retrieved context."""

    query: str = dspy.InputField(desc="Original user question")
    context: list[str] = dspy.InputField(
        desc="Retrieved passages from the hybrid RAG pipeline"
    )
    tool_results: list[str] = dspy.InputField(
        desc="Results from tool calls (empty list if no tools were used)"
    )
    answer: str = dspy.OutputField(desc="Comprehensive, well-structured answer")
    citations: list[str] = dspy.OutputField(
        desc="List of passage snippets used as evidence (verbatim quotes)"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Module instances
#
#  - query_analyser_llm : LangChain structured_llm  (Pydantic schema)
#  - hybrid_rag         : DSPy ChainOfThought        (optimisable)
#  - tools_llm          : LangChain llm.bind_tools    (native function calling)
# ─────────────────────────────────────────────────────────────────────────────

# ── LangChain: query routing via native JSON mode ────────────────────────────
QUERY_ANALYSIS_SYSTEM = (
    "You are a query analysis assistant. Given a user question:\n"
    "1. Decide if real-time data, calculations, or external APIs are needed.\n"
    "2. Produce 1-3 retrieval-optimised sub-queries to look up relevant documents.\n"
    "Respond only in the structured format requested."
)
query_analyser_llm = langchain_llm.with_structured_output(QueryAnalysisOutput)


# ── DSPy: answer generation (the only place the optimiser is worth running) ──
class HybridRAG(dspy.Module):
    """Core RAG module optimised with DSPy.  Receives pre-retrieved context."""

    def __init__(self):
        self.cot = dspy.ChainOfThought(HybridRAGGenerate)

    def forward(
        self, query: str, context: list[str], tool_results: list[str] | None = None
    ) -> dspy.Prediction:
        return self.cot(
            query=query,
            context=context,
            tool_results=tool_results or [],
        )


hybrid_rag = HybridRAG()


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Hybrid Retrieval  (Dense + Sparse + RRF fusion)
# ─────────────────────────────────────────────────────────────────────────────

# Demo corpus – replace with your actual documents / loader
DEMO_DOCUMENTS = [
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs.",
        metadata={"source": "langgraph_docs", "id": "d1"},
    ),
    Document(
        page_content="DSPy is a framework for algorithmically optimising LLM prompts and weights.",
        metadata={"source": "dspy_docs", "id": "d2"},
    ),
    Document(
        page_content="Hybrid RAG combines dense vector search with sparse BM25 keyword search "
        "and merges results using Reciprocal Rank Fusion (RRF).",
        metadata={"source": "rag_survey", "id": "d3"},
    ),
    Document(
        page_content="ChatVertexAI is LangChain's integration for Google Cloud Vertex AI models "
        "including the Gemini family.",
        metadata={"source": "langchain_docs", "id": "d4"},
    ),
    Document(
        page_content="Reciprocal Rank Fusion (RRF) merges ranked lists: score = Σ 1/(k + rank_i) "
        "where k=60 is a smoothing constant.",
        metadata={"source": "rrf_paper", "id": "d5"},
    ),
    Document(
        page_content="A ReAct agent iterates: Thought → Action → Observation until the agent "
        "has enough information to produce a final answer.",
        metadata={"source": "react_paper", "id": "d6"},
    ),
]


def build_retrievers(docs: list[Document]):
    """Build dense (Chroma) and sparse (BM25) retrievers from a document list."""
    # Dense retriever  – Chroma in-memory with VertexAI embeddings
    dense_store = Chroma.from_documents(docs, embedding=embeddings)
    dense_retriever = dense_store.as_retriever(search_kwargs={"k": 5})

    # Sparse retriever – BM25 (no embeddings needed)
    sparse_retriever = BM25Retriever.from_documents(docs, k=5)

    return dense_retriever, sparse_retriever


dense_retriever, sparse_retriever = build_retrievers(DEMO_DOCUMENTS)


def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]], k: int = 60
) -> list[Document]:
    """
    Merge multiple ranked document lists with RRF.
    score(d) = Σ_i  1 / (k + rank_i(d))
    Returns documents sorted by descending fused score.
    """
    scores: dict[str, float] = defaultdict(float)
    doc_map: dict[str, Document] = {}

    for ranked in ranked_lists:
        for rank, doc in enumerate(ranked, start=1):
            doc_id = doc.metadata.get("id", doc.page_content[:60])
            scores[doc_id] += 1.0 / (k + rank)
            doc_map[doc_id] = doc

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)
    return [doc_map[d] for d in sorted_ids]


def hybrid_retrieve(queries: list[str], top_k: int = 5) -> list[str]:
    """
    Run hybrid retrieval for every sub-query, fuse with RRF, return top-k passages.
    """
    all_ranked: list[list[Document]] = []
    for q in queries:
        dense_docs = dense_retriever.invoke(q)
        sparse_docs = sparse_retriever.invoke(q)
        all_ranked.extend([dense_docs, sparse_docs])

    fused = reciprocal_rank_fusion(all_ranked)[:top_k]
    return [d.page_content for d in fused]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  LangChain Tools  +  tools_llm (bind_tools = native Gemini function calling)
#
#  Why NOT DSPy here:
#  LangChain's bind_tools passes the full JSON schema to Gemini as a
#  "function declaration".  The model fills structured args natively.
#  DSPy would have to parse free-text dict output – strictly worse.
# ─────────────────────────────────────────────────────────────────────────────

@tool
def web_search(query: str) -> str:
    """Search the web for real-time or recent information not in the local corpus.
    Returns a short summary of the top result."""
    # Replace with real Tavily / SerpAPI call:
    # from langchain_community.tools.tavily_search import TavilySearchResults
    # return TavilySearchResults(max_results=3).invoke(query)
    return f"[MOCK web_search]: Top result for '{query}': Placeholder web content."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely. E.g. '2 ** 10 + sqrt(9)'."""
    import ast, operator as op
    allowed_ops = {
        ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
        ast.Div: op.truediv, ast.Pow: op.pow, ast.USub: op.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.n
        if isinstance(node, ast.BinOp):
            return allowed_ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return allowed_ops[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression: {ast.dump(node)}")

    try:
        tree = ast.parse(expression, mode="eval")
        return str(_eval(tree.body))
    except Exception as e:
        return f"Error evaluating '{expression}': {e}"


@tool
def lookup_document_metadata(doc_id: str) -> str:
    """Retrieve metadata (source, date) for a document given its ID."""
    meta = {d.metadata.get("id"): d.metadata for d in DEMO_DOCUMENTS}
    return json.dumps(meta.get(doc_id, {"error": "document not found"}))


ALL_TOOLS = [web_search, calculator, lookup_document_metadata]

# LangChain LLM with tools bound – Gemini receives full JSON schemas as
# "function declarations" and returns structured ToolCall objects.
# No manual dict parsing, no DSPy needed.
tools_llm = langchain_llm.bind_tools(ALL_TOOLS)

# Map name → callable for dispatching tool calls
TOOL_REGISTRY: dict[str, Any] = {t.name: t for t in ALL_TOOLS}


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LangGraph State
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Core pipeline data
    query: str
    search_queries: list[str]
    retrieved_context: list[str]
    partial_answer: str
    tool_results: list[str]
    final_answer: str
    citations: list[str]

    # Control flow flags
    needs_tools: bool
    tool_calls_remaining: int   # prevents infinite tool loops

    # LangGraph message thread (for observability / memory)
    messages: Annotated[list, add_messages]


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LangGraph Nodes
# ─────────────────────────────────────────────────────────────────────────────

def node_query_analysis(state: AgentState) -> dict:
    """Node 1 – LangChain with_structured_output routes the query.
    Pydantic schema → Gemini JSON mode.  Simple, transparent, no optimiser needed.
    """
    result: QueryAnalysisOutput = query_analyser_llm.invoke([
        SystemMessage(content=QUERY_ANALYSIS_SYSTEM),
        HumanMessage(content=state["query"]),
    ])
    return {
        "search_queries": result.search_queries,
        "needs_tools": result.needs_tools,
        "messages": [
            AIMessage(content=f"[Query analysis] Reasoning: {result.reasoning}\n"
                              f"Sub-queries: {result.search_queries}\n"
                              f"Needs tools: {result.needs_tools}")
        ],
    }


def node_hybrid_retrieval(state: AgentState) -> dict:
    """Node 2 – Dense + Sparse retrieval fused with RRF."""
    passages = hybrid_retrieve(state["search_queries"], top_k=6)
    return {
        "retrieved_context": passages,
        "messages": [
            AIMessage(content=f"[Hybrid retrieval] Retrieved {len(passages)} passages.")
        ],
    }


def node_generate_answer(state: AgentState) -> dict:
    """Node 3 – DSPy RAG module generates a grounded answer."""
    result = hybrid_rag(
        query=state["query"],
        context=state["retrieved_context"],
        tool_results=state.get("tool_results", []),
    )
    return {
        "partial_answer": result.answer,
        "citations": result.citations,
        "messages": [AIMessage(content=f"[Draft answer] {result.answer}")],
    }


def node_tool_execution(state: AgentState) -> dict:
    """Node 4 – LangChain bind_tools drives native Gemini function calling.

    tools_llm.invoke returns an AIMessage whose .tool_calls list contains
    structured {name, args} dicts – no free-text parsing, no DSPy needed.
    """
    context_text = "\n".join(state.get("retrieved_context", []))
    ai_msg: AIMessage = tools_llm.invoke([
        SystemMessage(
            content="You are a helpful assistant. Use the provided tools to fill "
                    "gaps in the answer. Call at most one tool per turn."
        ),
        HumanMessage(
            content=f"Question: {state['query']}\n\n"
                    f"Context:\n{context_text}\n\n"
                    f"Partial answer:\n{state['partial_answer']}\n\n"
                    f"What tool (if any) should be called next?"
        ),
    ])

    if not ai_msg.tool_calls:
        return {
            "needs_tools": False,
            "messages": [AIMessage(content="[Tool] No further tool call needed.")],
        }

    # Execute the first (and only) tool call
    tc = ai_msg.tool_calls[0]
    tool_fn = TOOL_REGISTRY.get(tc["name"])
    if tool_fn is None:
        return {"needs_tools": False, "messages": [ai_msg]}

    try:
        tool_output = tool_fn.invoke(tc["args"])
    except Exception as exc:
        tool_output = f"Tool error: {exc}"

    updated_tool_results = state.get("tool_results", []) + [
        f"{tc['name']}({tc['args']}) → {tool_output}"
    ]
    remaining = state.get("tool_calls_remaining", 3) - 1

    return {
        "tool_results": updated_tool_results,
        "tool_calls_remaining": remaining,
        "needs_tools": remaining > 0,
        "messages": [
            AIMessage(
                content=f"[Tool call] {tc['name']}({tc['args']})\nResult: {tool_output}"
            )
        ],
    }


def node_final_answer(state: AgentState) -> dict:
    """Node 5 – Re-generate final answer after all tools have been used."""
    result = hybrid_rag(
        query=state["query"],
        context=state["retrieved_context"],
        tool_results=state.get("tool_results", []),
    )
    return {
        "final_answer": result.answer,
        "citations": result.citations,
        "messages": [AIMessage(content=f"[Final answer] {result.answer}")],
    }


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Conditional Edges (routers)
# ─────────────────────────────────────────────────────────────────────────────

def router_needs_tools(state: AgentState) -> str:
    """After draft generation: branch to tool execution or directly to END."""
    if state.get("needs_tools") and state.get("tool_calls_remaining", 0) > 0:
        return "tool_execution"
    return "finish"


def router_after_tool(state: AgentState) -> str:
    """After a tool call: keep calling tools or finalise the answer."""
    if state.get("needs_tools") and state.get("tool_calls_remaining", 0) > 0:
        return "tool_execution"
    return "final_answer"


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Build the LangGraph
# ─────────────────────────────────────────────────────────────────────────────

def build_agent_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("query_analysis",   node_query_analysis)
    graph.add_node("hybrid_retrieval", node_hybrid_retrieval)
    graph.add_node("generate_answer",  node_generate_answer)
    graph.add_node("tool_execution",   node_tool_execution)
    graph.add_node("final_answer",     node_final_answer)

    # Linear backbone
    graph.add_edge(START,             "query_analysis")
    graph.add_edge("query_analysis",  "hybrid_retrieval")
    graph.add_edge("hybrid_retrieval","generate_answer")

    # Branch after draft answer
    graph.add_conditional_edges(
        "generate_answer",
        router_needs_tools,
        {"tool_execution": "tool_execution", "finish": END},
    )

    # Loop or finalise after each tool call
    graph.add_conditional_edges(
        "tool_execution",
        router_after_tool,
        {"tool_execution": "tool_execution", "final_answer": "final_answer"},
    )

    graph.add_edge("final_answer", END)
    return graph


agent = build_agent_graph().compile()


# ─────────────────────────────────────────────────────────────────────────────
# 11.  (Optional) DSPy Optimisation with BootstrapFewShot
#      Run offline once, then save/load the compiled program.
# ─────────────────────────────────────────────────────────────────────────────

def optimise_dspy_modules(train_examples: list[dspy.Example]) -> None:
    """
    Optimise ONLY the HybridRAG module (answer generation).
    Routing and tool selection are handled by LangChain – no optimisation needed.

    Run once offline with labelled QA examples; reload the saved JSON at startup.
    Replace the word-overlap metric with RAGAS faithfulness / answer-relevancy
    for production-quality optimisation.

    Example training record shape:
        dspy.Example(
            query="What is RRF?",
            context=["RRF merges ranked lists ..."],
            tool_results=[],
            answer="Reciprocal Rank Fusion ...",
            citations=["RRF merges ranked lists ..."],
        ).with_inputs("query", "context", "tool_results")
    """
    from dspy.teleprompt import BootstrapFewShot

    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        # Simple word-overlap proxy – swap for RAGAS in production:
        # from ragas.metrics import faithfulness, answer_relevancy
        gold_words = set(example.answer.lower().split())
        pred_words = set(pred.answer.lower().split())
        if not gold_words:
            return 0.0
        return len(gold_words & pred_words) / len(gold_words)

    optimiser = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4)
    compiled_rag = optimiser.compile(hybrid_rag, trainset=train_examples)
    compiled_rag.save("hybrid_rag_optimised.json")
    print("DSPy HybridRAG module saved to hybrid_rag_optimised.json")


def load_optimised_modules(path: str = "hybrid_rag_optimised.json") -> None:
    """Load a previously optimised DSPy program."""
    if os.path.exists(path):
        hybrid_rag.load(path)
        print(f"Loaded optimised DSPy module from {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_agent(query: str) -> dict:
    """Run the full agentic hybrid RAG pipeline for a user query."""
    initial_state: AgentState = {
        "query": query,
        "search_queries": [],
        "retrieved_context": [],
        "partial_answer": "",
        "tool_results": [],
        "final_answer": "",
        "citations": [],
        "needs_tools": False,
        "tool_calls_remaining": 3,      # max tool calls per turn
        "messages": [HumanMessage(content=query)],
    }

    final_state = agent.invoke(initial_state)

    # If the graph ended via the "finish" branch (no tools), partial_answer IS the answer
    answer = final_state.get("final_answer") or final_state.get("partial_answer", "")
    return {
        "answer": answer,
        "citations": final_state.get("citations", []),
        "tool_results": final_state.get("tool_results", []),
        "messages": final_state.get("messages", []),
    }


if __name__ == "__main__":
    # ── Load optimised DSPy modules if available ──
    load_optimised_modules()

    # ── Example queries ──
    queries = [
        "How does DSPy integrate with LangGraph for a hybrid RAG pipeline?",
        "What is RRF and how does it merge retrieval results?",
        "What is 2 to the power of 10 plus the square root of 144?",   # triggers calculator tool
    ]

    for q in queries:
        print("\n" + "=" * 70)
        print(f"QUERY: {q}")
        print("=" * 70)
        result = run_agent(q)
        print(f"\nANSWER:\n{result['answer']}")
        if result["citations"]:
            print(f"\nCITATIONS:")
            for c in result["citations"]:
                print(f"  • {c}")
        if result["tool_results"]:
            print(f"\nTOOL RESULTS:")
            for t in result["tool_results"]:
                print(f"  • {t}")
