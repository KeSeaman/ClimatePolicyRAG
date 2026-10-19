from functools import partial

from toolz import pipe

from climate_rag.agents.analysis import analyze_intent
from climate_rag.agents.retrieval import retrieve_documents
from climate_rag.agents.synthesis import synthesize_answer
from climate_rag.knowledge_base import KnowledgeBase, SearchOptions
from climate_rag.llm import LLM
from climate_rag.models import AgentState


def run_retrieval(
    query: str,
    kb: KnowledgeBase,
    k: int = 5,
    options: SearchOptions | None = None,
    expander: LLM | None = None,
) -> AgentState:
    """
    Intent analysis and retrieval, composed functionally; also used on its own for
    retrieval-only evaluation. Dependencies are bound with partial so every stage is
    AgentState -> AgentState.
    """
    state: AgentState = pipe(
        AgentState(query=query),
        analyze_intent,
        partial(retrieve_documents, kb=kb, k=k, options=options, expander=expander),
    )
    return state


def run_rag_pipeline(
    query: str,
    kb: KnowledgeBase,
    llm: LLM,
    k: int = 5,
    options: SearchOptions | None = None,
    expander: LLM | None = None,
) -> AgentState:
    """Full RAG flow: intent analysis and retrieval (run_retrieval), then synthesis."""
    return synthesize_answer(run_retrieval(query, kb, k, options, expander), llm)
