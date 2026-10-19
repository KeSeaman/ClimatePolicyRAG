from climate_rag.agents.expansion import expand_query
from climate_rag.knowledge_base import KnowledgeBase, SearchOptions
from climate_rag.llm import LLM
from climate_rag.models import AgentState, RetrievedContext


def retrieve_documents(
    state: AgentState,
    kb: KnowledgeBase,
    k: int = 5,
    options: SearchOptions | None = None,
    expander: LLM | None = None,
) -> AgentState:
    """
    Fetch the k best chunks for the question, optionally searching with LLM-written
    rephrasings as well (expander) and fusing all rankings.
    Errors propagate: a failed retrieval must not masquerade as "no relevant documents".
    """
    queries: tuple[str, ...] = (state.query,)
    if expander is not None:
        queries += tuple(expand_query(state.query, expander))
    options = options or SearchOptions()
    results = kb.search(queries, k, options)

    contexts = tuple(
        RetrievedContext(
            doc_id=r.doc_id,
            parent_id=r.metadata.get("parent_id", r.doc_id),
            content=r.content,
            score=r.score,
            source=r.metadata.get("source", "unknown"),
            title=r.metadata.get("title"),
            url=r.metadata.get("url"),
        )
        for r in results
    )

    return (
        state.set_search_queries(queries)
        .add_contexts(contexts)
        .add_log(f"Retrieved {len(contexts)} docs ({options.mode}, {len(queries)} queries)")
    )
