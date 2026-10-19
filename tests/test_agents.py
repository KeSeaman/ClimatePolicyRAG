import numpy as np
import pytest
from numpy.typing import NDArray

from climate_rag._core import Document
from climate_rag.agents.analysis import analyze_intent
from climate_rag.agents.expansion import expand_query
from climate_rag.agents.retrieval import retrieve_documents
from climate_rag.agents.synthesis import NO_CONTEXT_ANSWER, SYSTEM_PROMPT, synthesize_answer
from climate_rag.knowledge_base import KnowledgeBase, chunk_document
from climate_rag.models import AgentState, Intent, RetrievedContext
from climate_rag.pipeline import run_rag_pipeline

from .conftest import FakeLLM, HashEmbedder


@pytest.mark.parametrize(("query", "intent"), [
    ("What are the targets of the Paris Agreement?", Intent.POLICY),
    ("How does carbon pricing impact developing economies?", Intent.POLICY),
    ("Projected warming under high emissions scenarios", Intent.SCIENTIFIC),
    ("What barriers slow deployment of carbon capture projects?", Intent.IMPLEMENTATION),
    # Substring traps for the old classifier: "cop" in "scope", "action" in "fraction".
    ("What is the scope of the fraction of land used for solar?", Intent.GENERAL),
])
def test_analyze_intent(query: str, intent: Intent) -> None:
    assert analyze_intent(AgentState(query=query)).intent is intent


def test_retrieval_finds_the_relevant_document(kb: KnowledgeBase) -> None:
    state = retrieve_documents(AgentState(query="methane emissions from livestock"), kb, k=2)

    top = state.contexts[0]
    assert top.doc_id == "methane"
    assert top.source == "arxiv" and top.title == "Methane"
    assert top.content == "Methane emissions from agriculture and livestock drive warming."
    assert state.contexts[0].score > state.contexts[1].score


def test_retrieval_errors_propagate(kb: KnowledgeBase) -> None:
    class WrongDimension(HashEmbedder):
        def embed_query(self, text: str) -> NDArray[np.float32]:
            return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    kb.embedder = WrongDimension()
    with pytest.raises(ValueError, match="dimension mismatch"):
        retrieve_documents(AgentState(query="anything"), kb)


def test_synthesis_sends_numbered_full_sources(fake_llm: FakeLLM) -> None:
    long_text = "Carbon taxes are regressive without revenue recycling. " * 10
    state = AgentState(
        query="Does carbon pricing hurt the poor?",
        intent=Intent.POLICY,
        contexts=(RetrievedContext(doc_id="d1", parent_id="d1", content=long_text, score=0.9,
                                   source="arxiv", title="Carbon tax incidence"),),
    )

    result = synthesize_answer(state, fake_llm)

    assert result.answer == "Fake answer [1]."
    system, user = fake_llm.calls[0]
    assert system == SYSTEM_PROMPT
    assert "[1] Carbon tax incidence (source: arxiv)" in user
    assert long_text in user  # the whole chunk, not a 100-char preview
    assert user.rstrip().endswith("Focus on policy instruments, targets and commitments.")


def test_synthesis_without_context_skips_llm(fake_llm: FakeLLM) -> None:
    result = synthesize_answer(AgentState(query="q"), fake_llm)
    assert result.answer == NO_CONTEXT_ANSWER
    assert fake_llm.calls == []


def test_pipeline_end_to_end(kb: KnowledgeBase, fake_llm: FakeLLM) -> None:
    query = "What temperature goal does the Paris Agreement set?"
    state = run_rag_pipeline(query, kb, fake_llm, k=2)

    assert state.intent is Intent.POLICY
    assert state.contexts[0].doc_id == "paris"
    assert state.answer == "Fake answer [1]."
    assert len(state.logs) == 3
    with pytest.raises(Exception, match="frozen"):
        state.answer = "mutated"


def test_chunk_document_splits_on_sentences_and_keeps_metadata() -> None:
    text = "First sentence here. Second one is a bit longer. Third. " + "x" * 45
    chunks = chunk_document(Document("p", text, {"source": "s"}), max_chars=30)

    assert all(len(c.content) <= 30 for c in chunks)
    assert chunks[0].content == "First sentence here."
    assert [c.id for c in chunks] == [f"p#{i}" for i in range(len(chunks))]
    assert all(c.metadata == {"source": "s", "parent_id": "p"} for c in chunks)
    assert "".join(c.content.replace(" ", "") for c in chunks) == text.replace(" ", "")


def test_short_documents_are_not_chunked() -> None:
    doc = Document("p", "short", {})
    assert chunk_document(doc, 100) == [doc]


def test_re_ingesting_does_not_duplicate() -> None:
    kb = KnowledgeBase(HashEmbedder())
    docs = [Document("a", "alpha"), Document("b", "beta")]
    kb.add_documents(docs)
    kb.add_documents(docs)
    assert len(kb) == 2


def test_expand_query_parses_and_cleans_llm_output() -> None:
    llm = FakeLLM(
        "1. Monitoring, reporting and verification of emissions\n"
        '- "national greenhouse gas inventories"\n'
        "\n"
        "How are NDCs being tracked?\n"  # echo of the question: dropped
        "• Enhanced Transparency Framework reports\n"
        "• Enhanced Transparency Framework reports\n"  # duplicate: dropped
        "one too many"
    )

    expansions = expand_query("How are NDCs being tracked?", llm, n=3)

    assert expansions == [
        "Monitoring, reporting and verification of emissions",
        "national greenhouse gas inventories",
        "Enhanced Transparency Framework reports",
    ]
    assert "3 alternative search queries" in llm.calls[0][0]


def test_retrieval_with_expansion_records_queries(kb: KnowledgeBase) -> None:
    expander = FakeLLM("livestock methane\nParis temperature goal")
    state = retrieve_documents(AgentState(query="cows"), kb, k=2, expander=expander)

    assert state.search_queries == ("cows", "livestock methane", "Paris temperature goal")
    assert {c.doc_id for c in state.contexts} == {"methane", "paris"}
    assert state.logs[-1].endswith("Retrieved 2 docs (vector, 3 queries)")
