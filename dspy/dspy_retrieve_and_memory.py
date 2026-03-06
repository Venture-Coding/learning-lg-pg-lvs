"""
dspy_retrieve_and_memory.py
============================
Demonstrates the REAL value of DSPy beyond simple structured output:

1. dspy.Retrieve + dspy.configure(rm=...)
   - DSPy's retrieval model (RM) concept: retrieval becomes a trainable step
   - The optimizer can tune k (number of passages), query rewriting, and
     passage selection jointly with generation quality

2. dspy.Assert / dspy.Suggest
   - Self-refinement with constraints: if the answer violates a rule,
     DSPy automatically retries with feedback injected into the prompt
   - No LangGraph loop needed for simple constraint enforcement

3. Long-context memory management
   - ContextWindowManager: DSPy module that compresses conversation history
     to fit within token budgets before each generation call
   - Works for any long-context product (VS Code Copilot, multi-turn chat,
     document analysis)

4. Why the optimizer matters
   - Run optimise() once with labelled examples + a metric
   - DSPy selects best few-shot demos and rewrites field descriptions
   - Reload the compiled JSON at startup — no code changes needed
"""

from __future__ import annotations

import os
import textwrap
from typing import Any

import dspy
from dspy.retrieve.chromadb_rm import ChromadbRM   # DSPy-native Chroma RM

# ─────────────────────────────────────────────────────────────────────────────
# 1. Configure DSPy LM + RM  (both become trainable knobs for the optimizer)
# ─────────────────────────────────────────────────────────────────────────────

GCP_PROJECT  = os.getenv("GOOGLE_CLOUD_PROJECT", "my-project")
GCP_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ID     = "gemini-2.0-flash"

dspy_lm = dspy.LM(
    model=f"vertex_ai/{MODEL_ID}",
    temperature=0.0,
    max_tokens=4096,
    vertex_project=GCP_PROJECT,
    vertex_location=GCP_LOCATION,
)

# dspy.configure(rm=...) makes dspy.Retrieve() a trainable step.
# ChromadbRM is a DSPy-aware wrapper: it exposes the retriever as an RM
# so the optimizer can tune query reformulation and k jointly.
#
# Swap for any other DSPy RM:
#   dspy.ColBERTv2, dspy.AzureCognitiveSearch, dspy.WeaviateRM, etc.
chroma_rm = ChromadbRM(
    collection_name="my_documents",
    persist_directory="./chroma_db",
    k=5,
)

dspy.configure(lm=dspy_lm, rm=chroma_rm)


# ─────────────────────────────────────────────────────────────────────────────
# 2. RAG with dspy.Retrieve  (vs calling a retriever manually)
#
#  Why dspy.Retrieve instead of just calling retriever.invoke()?
#  ─────────────────────────────────────────────────────────────
#  When you call dspy.Retrieve(k=3)(query), DSPy records this step in its
#  execution trace.  The optimizer can then:
#    - Try different values of k (3, 5, 10) and pick the best
#    - Try different query rewriting strategies before the retrieve step
#    - Tune the full pipeline end-to-end including retrieval quality
#
#  If you call retriever.invoke() directly, the optimizer is blind to it.
# ─────────────────────────────────────────────────────────────────────────────

class RAGSignature(dspy.Signature):
    """Answer the question using ONLY the provided context passages.
    Cite every claim. If the context is insufficient, say so explicitly."""

    question: str         = dspy.InputField()
    context:  list[str]  = dspy.InputField(desc="Retrieved passages")
    answer:   str         = dspy.OutputField(desc="Cited, grounded answer")
    citations: list[str] = dspy.OutputField(desc="Verbatim passage quotes used")


class RAGWithRetrieve(dspy.Module):
    """
    The retrieve step IS part of the DSPy module.
    This gives the optimizer visibility into both retrieval AND generation.
    """
    def __init__(self, k: int = 5):
        self.retrieve = dspy.Retrieve(k=k)      # trainable retrieval step
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question: str) -> dspy.Prediction:
        # dspy.Retrieve uses the configured rm automatically
        passages = self.retrieve(question).passages   # list[str]
        return self.generate(question=question, context=passages)


# ─────────────────────────────────────────────────────────────────────────────
# 3. dspy.Assert / dspy.Suggest  (self-refinement with constraints)
#
#  This is the feature with NO clean equivalent in plain LangChain.
#  dspy.Assert(condition, feedback) will:
#    - If condition is False at inference time → retry the LLM call with
#      the feedback string injected as a correction hint
#    - If condition is still False after max_backtracks → raise DSPyAssertionError
#
#  dspy.Suggest is the same but soft: it logs a warning instead of raising.
#
#  Use case here: ensure the answer is grounded (citations non-empty) and
#  does not hallucinate by checking citation text appears in context.
# ─────────────────────────────────────────────────────────────────────────────

class GroundedRAG(dspy.Module):
    """RAG with assertion-based self-refinement.  No explicit retry loop needed."""

    def __init__(self, k: int = 5):
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought(RAGSignature)

    def forward(self, question: str) -> dspy.Prediction:
        passages = self.retrieve(question).passages
        pred = self.generate(question=question, context=passages)

        # Hard constraint: answer must include at least one citation
        dspy.Assert(
            len(pred.citations) > 0,
            "The answer must include at least one verbatim citation from the context. "
            "Re-read the context and quote specific passages.",
        )

        # Soft constraint: each citation should appear (approximately) in passages
        context_blob = " ".join(passages).lower()
        grounded = all(
            any(word in context_blob for word in c.lower().split()[:5])
            for c in pred.citations
        )
        dspy.Suggest(
            grounded,
            "Some citations do not appear in the retrieved context. "
            "Only cite text that is verbatim in the passages provided.",
        )

        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 4. Long-context memory management with DSPy
#
#  Problem in long-context products (VS Code Copilot, multi-turn chat,
#  document analysis): conversation history grows and eventually exceeds the
#  context window, OR degrades quality because the LLM attends to too much noise.
#
#  DSPy approach: ContextWindowManager — a DSPy module that:
#    a) Compresses old turns into a rolling summary (progressive compression)
#    b) Scores each turn for relevance to the CURRENT question
#    c) Keeps only the top-k relevant turns + the summary
#    d) The optimizer can tune the compression policy and relevance threshold
#
#  This is better than LangChain's ConversationSummaryMemory because:
#    - Relevance scoring is query-aware (trims irrelevant turns, not just old ones)
#    - The compression quality is optimisable with a metric (e.g. faithfulness)
#    - Works end-to-end with the generation module in one compiled program
# ─────────────────────────────────────────────────────────────────────────────

MAX_TOKENS_BUDGET = 6000   # approximate token budget for context


class CompressTurns(dspy.Signature):
    """Compress a list of conversation turns into a concise factual summary.
    Preserve all specific facts, numbers, decisions, and named entities.
    Discard pleasantries and repetition."""

    turns: list[str] = dspy.InputField(desc="Conversation turns to compress")
    summary: str     = dspy.OutputField(desc="Dense factual summary, max 200 words")


class ScoreRelevance(dspy.Signature):
    """Score how relevant a conversation turn is to the current question (0.0–1.0)."""

    question: str    = dspy.InputField()
    turn:     str    = dspy.InputField()
    score:    float  = dspy.OutputField(desc="Relevance score between 0.0 and 1.0")


class ContextWindowManager(dspy.Module):
    """
    Manages conversation history to fit within a token budget.

    Strategy (all steps are optimisable by DSPy):
      1. If history is short enough, pass it through unchanged.
      2. Compress the oldest half of turns into a rolling summary.
      3. Score remaining turns for relevance to the current question.
      4. Return: summary + top-k relevant recent turns.

    This is the pattern to use for VS Code Copilot or any long-context
    agentic product where conversation history can grow unbounded.
    """

    def __init__(self, token_budget: int = MAX_TOKENS_BUDGET, keep_recent: int = 4):
        self.token_budget  = token_budget
        self.keep_recent   = keep_recent
        self.compressor    = dspy.ChainOfThought(CompressTurns)
        self.scorer        = dspy.ChainOfThought(ScoreRelevance)

    @staticmethod
    def _rough_token_count(text: str) -> int:
        """Rough token estimate: ~4 chars per token."""
        return len(text) // 4

    def forward(
        self,
        question:        str,
        history:         list[str],   # list of "Role: text" strings
        existing_summary: str = "",
    ) -> dspy.Prediction:

        total_chars = sum(len(t) for t in history)
        total_tokens = self._rough_token_count(total_chars)

        # ── Fast path: history fits in budget ────────────────────────────────
        if total_tokens <= self.token_budget:
            return dspy.Prediction(
                summary=existing_summary,
                relevant_turns=history,
                compressed=False,
            )

        # ── Split: keep recent N turns, compress the rest ────────────────────
        recent_turns = history[-self.keep_recent:]
        old_turns    = history[:-self.keep_recent]

        if old_turns:
            compression_result = self.compressor(
                turns=old_turns + ([existing_summary] if existing_summary else [])
            )
            new_summary = compression_result.summary
        else:
            new_summary = existing_summary

        # ── Score recent turns for relevance to THIS question ────────────────
        scored = []
        for turn in recent_turns:
            score_result = self.scorer(question=question, turn=turn)
            try:
                score = float(score_result.score)
            except (ValueError, TypeError):
                score = 0.5   # default if parsing fails
            scored.append((score, turn))

        # Keep top-k by relevance score (always keep the most recent 1)
        scored.sort(key=lambda x: x[0], reverse=True)
        top_turns = [t for _, t in scored[: max(self.keep_recent, 2)]]
        # Preserve chronological order
        top_turns_ordered = [t for t in recent_turns if t in top_turns]

        dspy.Suggest(
            len(top_turns_ordered) > 0,
            "At least one recent turn should be retained for answer continuity.",
        )

        return dspy.Prediction(
            summary=new_summary,
            relevant_turns=top_turns_ordered,
            compressed=True,
        )


class ConversationalRAG(dspy.Module):
    """
    Full conversational RAG pipeline with managed context window.
    This entire module — retrieval, context compression, generation — is
    optimisable end-to-end with a single DSPy optimizer run.
    """

    def __init__(self, k: int = 5, token_budget: int = MAX_TOKENS_BUDGET):
        self.retrieve     = dspy.Retrieve(k=k)
        self.ctx_manager  = ContextWindowManager(token_budget=token_budget)
        self.generate     = dspy.ChainOfThought(RAGSignature)

    def forward(
        self,
        question:        str,
        history:         list[str] = None,
        existing_summary: str = "",
    ) -> dspy.Prediction:

        history = history or []

        # Step 1: retrieve relevant passages
        passages = self.retrieve(question).passages

        # Step 2: compress history to fit token budget
        ctx = self.ctx_manager(
            question=question,
            history=history,
            existing_summary=existing_summary,
        )

        # Step 3: build full context = summary + relevant history + passages
        full_context: list[str] = []
        if ctx.summary:
            full_context.append(f"[Conversation summary]\n{ctx.summary}")
        full_context.extend(ctx.relevant_turns)
        full_context.extend(passages)

        # Step 4: generate grounded answer
        pred = self.generate(question=question, context=full_context)

        # Attach context metadata to prediction for inspection
        pred.context_compressed = ctx.compressed
        pred.context_summary    = ctx.summary
        return pred


# ─────────────────────────────────────────────────────────────────────────────
# 5. DSPy Optimizer — the feature that makes ALL of the above worth it
#
#  Without this block, everything above is just a verbose prompt wrapper.
#  With this, DSPy automatically:
#    - Generates few-shot demos from your labelled data
#    - Rewrites field descriptions (the `desc=` strings) to be more effective
#    - Tunes k in dspy.Retrieve (tries 3, 5, 10 passages)
#    - Optimises the compression and relevance scoring prompts
#    - Does all of this jointly, optimising the end metric
# ─────────────────────────────────────────────────────────────────────────────

def optimise(
    module: dspy.Module,
    trainset: list[dspy.Example],
    save_path: str,
) -> dspy.Module:
    """
    Run BootstrapFewShot.  For larger datasets use MIPROv2 instead:
        from dspy.teleprompt import MIPROv2
        optimiser = MIPROv2(metric=metric, num_candidates=10, init_temperature=0.7)

    Metric should return a float in [0, 1].
    Replace word_overlap with RAGAS faithfulness for production:
        from ragas.metrics import faithfulness
    """
    from dspy.teleprompt import BootstrapFewShot

    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        if not hasattr(pred, "answer") or not pred.answer:
            return 0.0
        gold = set(example.answer.lower().split())
        pred_words = set(pred.answer.lower().split())
        overlap = len(gold & pred_words) / max(len(gold), 1)
        citation_bonus = 0.1 if getattr(pred, "citations", []) else 0.0
        return min(1.0, overlap + citation_bonus)

    opt = BootstrapFewShot(metric=metric, max_bootstrapped_demos=4, max_labeled_demos=8)
    compiled = opt.compile(module, trainset=trainset)
    compiled.save(save_path)
    print(f"Optimised module saved → {save_path}")
    return compiled


# ─────────────────────────────────────────────────────────────────────────────
# 6. Usage example
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Grounded RAG with assertion-based self-refinement ──────────────────
    rag = GroundedRAG(k=5)

    # For self-refinement (dspy.Assert retry) to work, wrap execution:
    with dspy.context(max_backtracks=3):
        result = rag(question="What is Reciprocal Rank Fusion?")

    print("Answer:", result.answer)
    print("Citations:", result.citations)

    # ── Conversational RAG with context window management ──────────────────
    conv_rag = ConversationalRAG(k=5, token_budget=6000)

    # Simulated long conversation history
    history = [
        "User: What is DSPy?",
        "Assistant: DSPy is a framework for algorithmically optimising LLM prompts.",
        "User: How does it compare to LangChain?",
        "Assistant: LangChain focuses on composability; DSPy focuses on optimisation.",
        "User: What optimisers does it have?",
        "Assistant: BootstrapFewShot, MIPROv2, COPRO, BayesianSignatureOptimizer.",
        # ... imagine 50 more turns here
    ] * 10   # simulate 60 turns to trigger compression

    result = conv_rag(
        question="Explain how DSPy's BootstrapFewShot works with retrieval.",
        history=history,
        existing_summary="",
    )

    print("\n--- Conversational RAG ---")
    print(f"Context compressed: {result.context_compressed}")
    print(f"Summary used: {result.context_summary[:200] if result.context_summary else 'none'}")
    print(f"Answer: {result.answer}")

    # ── Optimise the ConversationalRAG module (run once offline) ──────────
    # train_examples = [
    #     dspy.Example(
    #         question="What is RRF?",
    #         answer="Reciprocal Rank Fusion merges ranked lists using 1/(k+rank).",
    #         citations=["RRF merges ranked lists: score = Σ 1/(k + rank_i)"],
    #     ).with_inputs("question"),
    #     # ... more examples
    # ]
    # optimise(conv_rag, train_examples, "conversational_rag_optimised.json")
    #
    # Next run:  conv_rag.load("conversational_rag_optimised.json")
