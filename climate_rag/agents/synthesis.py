from collections.abc import Sequence

from climate_rag.llm import LLM
from climate_rag.models import AgentState, Intent, RetrievedContext

NO_CONTEXT_ANSWER = "I could not find relevant information in the climate knowledge base."

SYSTEM_PROMPT = (
    "You are a climate policy analyst. Answer the question using only the numbered sources "
    "provided. Cite sources inline as [1], [2], etc. If the sources do not contain enough "
    "information to answer, say so plainly and do not fill the gap from general knowledge. "
    "Be concise: at most two short paragraphs."
)

_INTENT_GUIDANCE = {
    Intent.POLICY: "Focus on policy instruments, targets and commitments.",
    Intent.SCIENTIFIC: "Focus on scientific findings and quantitative evidence.",
    Intent.IMPLEMENTATION: "Focus on how measures are financed and delivered, and on barriers.",
    Intent.GENERAL: "",
}


def format_sources(contexts: Sequence[RetrievedContext]) -> str:
    """Numbered source list shared by the answer prompt and the eval judge."""
    return "\n\n".join(
        f"[{i}] {c.title or c.doc_id} (source: {c.source})\n{c.content}"
        for i, c in enumerate(contexts, 1)
    )


def build_prompt(state: AgentState) -> str:
    sources = format_sources(state.contexts)
    guidance = _INTENT_GUIDANCE[state.intent]
    return f"Sources:\n\n{sources}\n\nQuestion: {state.query}\n{guidance}".rstrip()


def synthesize_answer(state: AgentState, llm: LLM) -> AgentState:
    """
    Generate a grounded, cited answer from the retrieved contexts.
    """
    if not state.contexts:
        return state.set_answer(NO_CONTEXT_ANSWER).add_log("Synthesis: No context found.")

    answer = llm(SYSTEM_PROMPT, build_prompt(state)).strip()
    return state.set_answer(answer).add_log("Synthesis: Generated answer.")
