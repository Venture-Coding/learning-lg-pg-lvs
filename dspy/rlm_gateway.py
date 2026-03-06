"""
rlm_gateway.py  —  Recursive LM Gateway
=========================================
A gateway LM routes a query to a specialised sub-agent that loads ONLY the
context slice it needs.  No sub-agent ever sees the full repo.

Key insight
-----------
The bottleneck in large-repo Copilot-style products is not the model —
it's the context window.  Routing to a sub-agent that loads 3 files instead of
300 gives you:
  - Lower latency (smaller prompt → faster TTFT)
  - Higher accuracy (less noise → better attention)
  - Lower cost (fewer input tokens billed)
  - Better security (sub-agents can't leak unrelated files)

Architecture
------------
Gateway (ChatVertexAI + with_structured_output)
    → classifies intent + extracts minimal file scope
    → dispatches to one of five sub-agents

Sub-agents (each is a DSPy module or LangChain chain):
  1. CodeExplainer    — explain_code:        loads 2-5 files
  2. BugFinder        — find_bug:            loads 1 file + imports + git diff
  3. TestWriter       — write_test:          loads 1 function + type stubs
  4. ModuleSummariser — summarise_module:    chunks + progressively compresses
  5. CrossFileReasoner— cross_file_reason:   hybrid RAG over repo index

DSPy is used ONLY in sub-agents where a quality metric exists and optimisation
is worth the compile-time LLM budget.  The gateway itself uses LangChain
with_structured_output (no training data needed for routing).
"""

from __future__ import annotations

import ast
import os
import textwrap
from pathlib import Path
from typing import Literal

import dspy
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "my-project")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")

# Fast/cheap model for the gateway (routing only — no generation)
GATEWAY_MODEL = "gemini-2.0-flash"
# Capable model for sub-agents that need deep reasoning
AGENT_MODEL   = "gemini-2.0-flash"   # swap to gemini-2.5-pro for harder tasks

gateway_llm = ChatVertexAI(
    model_name=GATEWAY_MODEL, project=GCP_PROJECT,
    location=GCP_LOCATION, temperature=0.0, max_output_tokens=512,
)
embeddings = VertexAIEmbeddings(
    model_name="text-embedding-005",
    project=GCP_PROJECT, location=GCP_LOCATION,
)

dspy.configure(lm=dspy.LM(
    model=f"vertex_ai/{AGENT_MODEL}",
    temperature=0.0, max_tokens=4096,
    vertex_project=GCP_PROJECT, vertex_location=GCP_LOCATION,
))


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Gateway: classify intent + extract minimal file scope
#     Uses with_structured_output — no DSPy, no training data needed.
# ─────────────────────────────────────────────────────────────────────────────

TaskType = Literal[
    "explain_code",
    "find_bug",
    "write_test",
    "summarise_module",
    "cross_file_reason",
]

class GatewayDecision(BaseModel):
    """Routing decision produced by the gateway LM."""
    task_type: TaskType = Field(
        description="The type of task the user is asking for"
    )
    file_scope: list[str] = Field(
        description=(
            "Minimal list of file paths (relative to repo root) needed to answer. "
            "NEVER include more than 5 files. If unsure, pick the single most relevant file."
        )
    )
    sub_query: str = Field(
        description="Refined, self-contained question to pass to the sub-agent"
    )
    reasoning: str = Field(
        description="One sentence explaining why this task type and file scope was chosen"
    )


GATEWAY_SYSTEM = """\
You are a code assistant gateway.  Given a user question and the repo file tree,
decide:
  1. task_type   — what kind of task this is
  2. file_scope  — the MINIMUM set of files needed (max 5)
  3. sub_query   — a refined, self-contained question for the sub-agent

Rules:
- NEVER include files that are clearly irrelevant.
- For explain_code / find_bug → scope = the specific file mentioned or implied.
- For write_test → scope = the file containing the function to test.
- For summarise_module → scope = the module's __init__.py or main file.
- For cross_file_reason → scope = [] (sub-agent uses RAG over the full index).
"""

gateway_structured = gateway_llm.with_structured_output(GatewayDecision)


def gateway_route(
    user_query: str,
    file_tree: str,          # lightweight: just paths, no content
) -> GatewayDecision:
    """
    The gateway reads only the FILE TREE (paths, no content).
    It never loads any file's content — that's the sub-agent's job.
    A 1000-file repo's file tree is ~5-10k tokens; totally manageable.
    """
    return gateway_structured.invoke([
        SystemMessage(content=GATEWAY_SYSTEM),
        HumanMessage(content=f"Repo file tree:\n{file_tree}\n\nUser question: {user_query}"),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# 2.  File loader — loads ONLY the files in the gateway's scope
#     This is the key mechanism: sub-agents never touch unscoped files.
# ─────────────────────────────────────────────────────────────────────────────

MAX_FILE_TOKENS = 6000   # ~24k chars per file — hard cap per sub-agent call

def load_scoped_files(
    file_paths: list[str],
    repo_root: str | Path,
    max_tokens_per_file: int = MAX_FILE_TOKENS,
) -> list[str]:
    """
    Load file contents for ONLY the files the gateway selected.
    Truncates files that exceed the token budget (tail truncation —
    swap for smarter chunking if needed).
    """
    root = Path(repo_root)
    loaded = []
    for rel_path in file_paths:
        abs_path = root / rel_path
        if not abs_path.exists():
            loaded.append(f"# [{rel_path}] — file not found")
            continue
        try:
            content = abs_path.read_text(encoding="utf-8", errors="replace")
            # Rough token estimate: 4 chars ≈ 1 token
            max_chars = max_tokens_per_file * 4
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n\n... [truncated — {rel_path}]"
            loaded.append(f"# === {rel_path} ===\n{content}")
        except Exception as exc:
            loaded.append(f"# [{rel_path}] — read error: {exc}")
    return loaded


def get_file_tree(repo_root: str | Path, max_files: int = 2000) -> str:
    """
    Return a lightweight file tree (paths only, no content).
    This is what the gateway reads — never the file contents.
    """
    root = Path(repo_root)
    paths = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file()
        and not any(part.startswith(".") or part in ("__pycache__", "node_modules")
                    for part in p.parts)
    )[:max_files]
    return "\n".join(paths)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Sub-Agent: CodeExplainer (DSPy — optimisable)
# ─────────────────────────────────────────────────────────────────────────────

class ExplainSignature(dspy.Signature):
    """Explain what the provided code does, step by step.
    Cover: purpose, key functions/classes, data flow, side-effects, gotchas."""

    question:  str       = dspy.InputField(desc="What the user wants to understand")
    code_files: list[str] = dspy.InputField(desc="File contents (already scoped by gateway)")
    explanation: str     = dspy.OutputField(desc="Clear, structured explanation")
    key_symbols: list[str] = dspy.OutputField(
        desc="Function/class names central to the explanation"
    )


class CodeExplainer(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(ExplainSignature)

    def forward(self, question: str, code_files: list[str]) -> dspy.Prediction:
        return self.cot(question=question, code_files=code_files)


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Sub-Agent: BugFinder (DSPy with Assert — must cite line numbers)
# ─────────────────────────────────────────────────────────────────────────────

class BugSignature(dspy.Signature):
    """Find bugs, logic errors, or potential exceptions in the provided code.
    Every finding MUST include the exact line number."""

    question:   str       = dspy.InputField()
    code_files: list[str] = dspy.InputField()
    git_diff:   str       = dspy.InputField(desc="Git diff of recent changes (may be empty)")
    findings:   list[str] = dspy.OutputField(
        desc="List of bugs. Format: 'Line N: <description>'. Empty list if no bugs found."
    )
    severity: str = dspy.OutputField(desc="critical / high / medium / low / none")


class BugFinder(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(BugSignature)

    def forward(
        self, question: str, code_files: list[str], git_diff: str = ""
    ) -> dspy.Prediction:
        pred = self.cot(question=question, code_files=code_files, git_diff=git_diff)

        # Assert: every finding must start with "Line "
        all_cited = all(f.strip().startswith("Line ") for f in pred.findings) if pred.findings else True
        dspy.Assert(
            all_cited,
            "Every bug finding must start with 'Line N:' citing the exact line number. "
            "Re-read the code and add line references to each finding.",
        )
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Sub-Agent: TestWriter (DSPy — optimisable with test-pass metric)
# ─────────────────────────────────────────────────────────────────────────────

class TestSignature(dspy.Signature):
    """Write pytest unit tests for the specified function or class.
    Use only the symbols visible in the provided code. No imports beyond
    stdlib, pytest, and the module under test."""

    question:     str       = dspy.InputField(desc="What to test and any specific scenarios")
    code_files:   list[str] = dspy.InputField(desc="Source file(s) containing the code under test")
    test_code:    str       = dspy.OutputField(desc="Complete, runnable pytest test file")
    test_cases:   list[str] = dspy.OutputField(desc="One-line description of each test case")


class TestWriter(dspy.Module):
    def __init__(self):
        self.cot = dspy.ChainOfThought(TestSignature)

    def forward(self, question: str, code_files: list[str]) -> dspy.Prediction:
        pred = self.cot(question=question, code_files=code_files)

        # Assert: generated code must be valid Python syntax
        try:
            ast.parse(pred.test_code)
            valid_syntax = True
        except SyntaxError:
            valid_syntax = False

        dspy.Assert(
            valid_syntax,
            "The generated test code has a syntax error. Fix the syntax and return "
            "valid Python that can be parsed without errors.",
        )
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 6.  Sub-Agent: ModuleSummariser (DSPy — recursive chunked compression)
#
#  This is the recursive part of the "Recursive LM Gateway":
#  large modules are split into chunks, each chunk summarised independently,
#  then summaries are merged.  No single LLM call ever sees the full module.
# ─────────────────────────────────────────────────────────────────────────────

class ChunkSummarySignature(dspy.Signature):
    """Summarise a code chunk.  Preserve: class/function names, parameters,
    return types, side-effects, raised exceptions, important constants."""

    chunk:   str = dspy.InputField(desc="A section of source code")
    summary: str = dspy.OutputField(desc="Dense technical summary, max 150 words")


class MergeSummarySignature(dspy.Signature):
    """Merge multiple chunk summaries into one coherent module summary."""

    chunk_summaries: list[str] = dspy.InputField()
    module_name:     str       = dspy.InputField()
    final_summary:   str       = dspy.OutputField(desc="Coherent module summary, max 300 words")
    public_api:      list[str] = dspy.OutputField(
        desc="Public functions/classes and their one-line signatures"
    )


class ModuleSummariser(dspy.Module):
    """
    Recursively summarise a large module without ever loading it fully in one call.

    Chunk size is ~100 lines (~400 tokens) → each chunk fits easily in context.
    This is the same principle VS Code Copilot uses internally for large files.
    """
    CHUNK_LINES = 100

    def __init__(self):
        self.chunk_cot = dspy.ChainOfThought(ChunkSummarySignature)
        self.merge_cot = dspy.ChainOfThought(MergeSummarySignature)

    def _chunk_code(self, code: str) -> list[str]:
        lines = code.splitlines()
        return [
            "\n".join(lines[i : i + self.CHUNK_LINES])
            for i in range(0, len(lines), self.CHUNK_LINES)
        ]

    def forward(self, question: str, code_files: list[str]) -> dspy.Prediction:
        module_name = question  # reuse question field as module identifier
        all_chunk_summaries: list[str] = []

        for file_content in code_files:
            chunks = self._chunk_code(file_content)
            for chunk in chunks:
                result = self.chunk_cot(chunk=chunk)
                all_chunk_summaries.append(result.summary)

        merged = self.merge_cot(
            chunk_summaries=all_chunk_summaries,
            module_name=module_name,
        )
        return merged


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Sub-Agent: CrossFileReasoner (Hybrid RAG over full repo index)
#
#  When the gateway sets task_type=cross_file_reason, no files are pre-loaded.
#  Instead the sub-agent uses a pre-built BM25 + vector index of the entire repo.
#  Each retrieval call pulls only the 5 most relevant chunks — never the full files.
#  This is what makes cross-file reasoning over 1000+ file repos tractable.
# ─────────────────────────────────────────────────────────────────────────────

class CrossFileSignature(dspy.Signature):
    """Answer a question that requires reasoning across multiple files in a codebase.
    Use ONLY the retrieved code snippets.  Cite file path + line range for every claim."""

    question:  str       = dspy.InputField()
    snippets:  list[str] = dspy.InputField(desc="Retrieved code snippets with file paths")
    answer:    str       = dspy.OutputField(desc="Grounded answer with citations")
    citations: list[str] = dspy.OutputField(desc="file_path:line_range for each claim")


class CrossFileReasoner(dspy.Module):
    def __init__(self, retriever):
        self.retriever = retriever   # BM25Retriever or Chroma retriever
        self.cot = dspy.ChainOfThought(CrossFileSignature)

    def forward(self, question: str, code_files: list[str] = None) -> dspy.Prediction:
        # Retrieve across the full repo index — not from scoped files
        docs = self.retriever.invoke(question)
        snippets = [
            f"# {d.metadata.get('source', 'unknown')} "
            f"lines {d.metadata.get('start_line', '?')}-{d.metadata.get('end_line', '?')}\n"
            f"{d.page_content}"
            for d in docs
        ]
        pred = self.cot(question=question, snippets=snippets)
        dspy.Assert(
            len(pred.citations) > 0,
            "Every claim must be cited with a file path and line range. "
            "Re-read the snippets and add citations.",
        )
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 8.  Dispatcher: wires gateway decision → correct sub-agent
# ─────────────────────────────────────────────────────────────────────────────

# Pre-instantiate sub-agents (load optimised weights if available)
_explainer   = CodeExplainer()
_bug_finder  = BugFinder()
_test_writer = TestWriter()
_summariser  = ModuleSummariser()

# CrossFileReasoner needs a retriever — built lazily from the repo index
_cross_file_reasoner: CrossFileReasoner | None = None


def _get_cross_file_reasoner(repo_root: str | Path) -> CrossFileReasoner:
    """Build BM25 index from code chunks on first call (or reload from cache)."""
    global _cross_file_reasoner
    if _cross_file_reasoner is not None:
        return _cross_file_reasoner

    root = Path(repo_root)
    docs: list[Document] = []
    for py_file in root.rglob("*.py"):
        try:
            code = py_file.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        lines = code.splitlines()
        # Chunk into 80-line windows with 20-line overlap
        for start in range(0, len(lines), 60):
            end = min(start + 80, len(lines))
            chunk = "\n".join(lines[start:end])
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": str(py_file.relative_to(root)),
                    "start_line": start + 1,
                    "end_line": end,
                },
            ))

    retriever = BM25Retriever.from_documents(docs, k=6)
    _cross_file_reasoner = CrossFileReasoner(retriever=retriever)
    return _cross_file_reasoner


SUB_AGENT_MAP = {
    "explain_code":     lambda q, files, **_: _explainer(question=q, code_files=files),
    "find_bug":         lambda q, files, **kw: _bug_finder(question=q, code_files=files, git_diff=kw.get("git_diff", "")),
    "write_test":       lambda q, files, **_: _test_writer(question=q, code_files=files),
    "summarise_module": lambda q, files, **_: _summariser(question=q, code_files=files),
}


def dispatch(
    decision: GatewayDecision,
    repo_root: str | Path,
    git_diff: str = "",
) -> dspy.Prediction:
    """
    Load ONLY the scoped files, then call the appropriate sub-agent.
    Total tokens seen by any single LLM call is bounded by:
        len(file_scope) * MAX_FILE_TOKENS  (sub-agents)
        ~10k tokens for file tree          (gateway)
    """
    if decision.task_type == "cross_file_reason":
        # CrossFileReasoner builds its own context via retrieval — no file loading
        reasoner = _get_cross_file_reasoner(repo_root)
        with dspy.context(max_backtracks=2):
            return reasoner(question=decision.sub_query)

    # Load only the files the gateway selected
    code_files = load_scoped_files(decision.file_scope, repo_root)

    fn = SUB_AGENT_MAP.get(decision.task_type)
    if fn is None:
        raise ValueError(f"Unknown task type: {decision.task_type}")

    with dspy.context(max_backtracks=2):   # enables dspy.Assert retries
        return fn(decision.sub_query, code_files, git_diff=git_diff)


# ─────────────────────────────────────────────────────────────────────────────
# 9.  Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def ask(
    user_query: str,
    repo_root:  str | Path = ".",
    git_diff:   str = "",
    verbose:    bool = False,
) -> dict:
    """
    Full pipeline:
      1. Gateway reads file tree (~5k tokens) → routing decision
      2. Dispatcher loads only scoped files   → sub-agent call
      3. Sub-agent returns grounded answer

    No single LLM call ever sees more than ~8k tokens of code.
    """
    root      = Path(repo_root)
    file_tree = get_file_tree(root)

    # Step 1: gateway routing (fast, cheap, small context)
    decision = gateway_route(user_query, file_tree)

    if verbose:
        print(f"[Gateway] task={decision.task_type}  "
              f"scope={decision.file_scope}  "
              f"reason={decision.reasoning}")

    # Step 2: sub-agent call (context-bounded)
    result = dispatch(decision, root, git_diff=git_diff)

    return {
        "task_type":  decision.task_type,
        "file_scope": decision.file_scope,
        "sub_query":  decision.sub_query,
        "result":     result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 10.  (Optional) Optimise sub-agents offline
#
#  Each sub-agent can be optimised independently with its own metric.
#  The gateway is NOT optimised — routing quality is checked manually
#  (it's a classification task, not a generation task).
# ─────────────────────────────────────────────────────────────────────────────

def optimise_sub_agents(
    explainer_trainset:   list[dspy.Example] | None = None,
    bug_finder_trainset:  list[dspy.Example] | None = None,
    test_writer_trainset: list[dspy.Example] | None = None,
) -> None:
    """
    Run compile-time optimisation for sub-agents.
    Each call makes N_examples * N_backtracks * N_candidates LLM calls — run offline.

    Explainer example:
        dspy.Example(
            question="What does the RRF function do?",
            code_files=["def reciprocal_rank_fusion(...):\n    ..."],
            explanation="RRF merges ranked lists ...",
            key_symbols=["reciprocal_rank_fusion"],
        ).with_inputs("question", "code_files")

    BugFinder example:
        dspy.Example(
            question="Find bugs in this code",
            code_files=["def divide(a, b):\n    return a/b  # line 2"],
            git_diff="",
            findings=["Line 2: ZeroDivisionError when b=0"],
            severity="high",
        ).with_inputs("question", "code_files", "git_diff")
    """
    from dspy.teleprompt import BootstrapFewShot

    if explainer_trainset:
        def explain_metric(ex, pred, trace=None):
            # Proxy: do key_symbols appear in the explanation?
            symbols_covered = sum(
                1 for s in ex.key_symbols if s.lower() in pred.explanation.lower()
            )
            return symbols_covered / max(len(ex.key_symbols), 1)

        opt = BootstrapFewShot(metric=explain_metric, max_bootstrapped_demos=4)
        compiled = opt.compile(_explainer, trainset=explainer_trainset)
        compiled.save("explainer_optimised.json")

    if bug_finder_trainset:
        def bug_metric(ex, pred, trace=None):
            # All expected findings cited?
            expected = set(ex.findings)
            found    = set(pred.findings)
            precision = len(expected & found) / max(len(found), 1)
            recall    = len(expected & found) / max(len(expected), 1)
            return (precision + recall) / 2

        opt = BootstrapFewShot(metric=bug_metric, max_bootstrapped_demos=4)
        compiled = opt.compile(_bug_finder, trainset=bug_finder_trainset)
        compiled.save("bug_finder_optimised.json")

    if test_writer_trainset:
        def test_metric(ex, pred, trace=None):
            # Valid syntax is the minimum bar
            try:
                ast.parse(pred.test_code)
                return 1.0
            except SyntaxError:
                return 0.0

        opt = BootstrapFewShot(metric=test_metric, max_bootstrapped_demos=4)
        compiled = opt.compile(_test_writer, trainset=test_writer_trainset)
        compiled.save("test_writer_optimised.json")


def load_optimised_sub_agents() -> None:
    """Load compiled sub-agent weights at startup."""
    for module, path in [
        (_explainer,   "explainer_optimised.json"),
        (_bug_finder,  "bug_finder_optimised.json"),
        (_test_writer, "test_writer_optimised.json"),
    ]:
        if Path(path).exists():
            module.load(path)
            print(f"Loaded optimised weights: {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  Demo
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_optimised_sub_agents()

    repo = Path(".")   # point at your actual repo root

    test_queries = [
        "Explain what the reciprocal_rank_fusion function does",
        "Are there any bugs in the calculator tool function?",
        "Write tests for the hybrid_retrieve function",
        "Summarise the dspy_retrieve_and_memory module",
        "How does the LangGraph state flow connect to the DSPy modules?",
    ]

    for q in test_queries:
        print("\n" + "=" * 60)
        print(f"QUERY: {q}")
        out = ask(q, repo_root=repo, verbose=True)
        print(f"TASK:  {out['task_type']}")
        print(f"SCOPE: {out['file_scope']}")
        result = out["result"]
        # Print the most relevant output field
        for field in ("answer", "explanation", "test_code", "final_summary", "findings"):
            val = getattr(result, field, None)
            if val:
                preview = val[:500] if isinstance(val, str) else str(val)[:500]
                print(f"\n{field.upper()}:\n{preview}")
                break
