from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class Intent(StrEnum):
    SCIENTIFIC = "scientific"
    POLICY = "policy"
    IMPLEMENTATION = "implementation"
    GENERAL = "general"


class RetrievedContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    # The ingested document this chunk came from (== doc_id when it was not split)
    parent_id: str
    content: str
    score: float
    source: str
    title: str | None = None
    url: str | None = None


class AgentState(BaseModel):
    """
    Immutable state passed through the functional pipeline.
    Each agent returns a NEW instance of AgentState; collections are tuples so
    nothing can be mutated in place either.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    intent: Intent = Intent.GENERAL
    # Every query sent to the index: the question, plus any LLM expansions
    search_queries: tuple[str, ...] = ()
    # Contexts retrieved from the vector index, best match first
    contexts: tuple[RetrievedContext, ...] = ()
    # Synthesized answer
    answer: str | None = None
    # Agent chain execution log
    logs: tuple[str, ...] = ()

    def add_log(self, message: str) -> "AgentState":
        entry = f"{datetime.now(UTC).isoformat()}: {message}"
        return self.model_copy(update={"logs": (*self.logs, entry)})

    def update_intent(self, new_intent: Intent) -> "AgentState":
        return self.model_copy(update={"intent": new_intent})

    def set_search_queries(self, queries: tuple[str, ...]) -> "AgentState":
        return self.model_copy(update={"search_queries": queries})

    def add_contexts(self, new_contexts: tuple[RetrievedContext, ...]) -> "AgentState":
        return self.model_copy(update={"contexts": (*self.contexts, *new_contexts)})

    def set_answer(self, answer: str) -> "AgentState":
        return self.model_copy(update={"answer": answer})
