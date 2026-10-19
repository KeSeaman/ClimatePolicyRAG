import pytest

from climate_rag.evaluation import (
    cited_sources,
    judge_answer,
    score_answer,
    score_retrieval,
    summarize,
    ungrounded_numbers,
)
from climate_rag.models import AgentState, RetrievedContext

from .conftest import FakeLLM


def ctx(doc_id: str, content: str = "", parent: str | None = None) -> RetrievedContext:
    return RetrievedContext(
        doc_id=doc_id, parent_id=parent or doc_id, content=content, score=0.5, source="arxiv"
    )


@pytest.mark.parametrize(("answer", "expected"), [
    ("No citations here.", []),
    ("A [1]. B [3][2].", [1, 2, 3]),
    ("Joint claim [1, 4] and a range [2-3] and an en dash [5–6].", [1, 2, 3, 4, 5, 6]),
    ("Brackets like [a] or [] are ignored.", []),
])
def test_cited_sources(answer: str, expected: list[int]) -> None:
    assert cited_sources(answer) == expected


def test_score_retrieval_is_document_level() -> None:
    # Two chunks of the same paper count once; rank is by document.
    contexts = [ctx("x#0", parent="x"), ctx("x#1", parent="x"), ctx("rel1"), ctx("y")]

    score = score_retrieval(contexts, ["rel1", "rel2"])

    assert score is not None
    assert score.found == ["rel1"]
    assert score.missed == ["rel2"]
    assert score.reciprocal_rank == pytest.approx(1 / 2)  # documents ranked x, rel1, y
    assert score.recall == pytest.approx(1 / 2)
    assert score_retrieval(contexts, []) is None


def test_ungrounded_numbers_checks_cited_sources_only() -> None:
    contexts = [
        ctx("a", "Methane's GWP over 20 years is about 80, and 1,000 sites were surveyed."),
        ctx("b", "Emissions fell 4.6% in the long run."),
    ]
    answer = (
        "Methane's GWP is about 80 [1], across 1,000 sites [1]. "
        "Emissions fell 4.6% [1]. Sea level could rise 2.5 m by 2100 [1]. "
        "There are 3 reasons and CO2 is the baseline [1]."
    )

    # 4.6 exists only in the uncited source [2]; 2100 comes from the question;
    # "3" is a trivial integer and "CO2" is not a number.
    assert ungrounded_numbers(answer, contexts, "What happens by 2100?") == ["4.6", "2.5"]


def test_ungrounded_numbers_allows_rounding_to_the_stated_precision() -> None:
    contexts = [ctx("a", "Overreporting of 32.03% (interval 19.8 to 45.27).")]
    answer = (
        "Inflated by about 32% [1], at most 45.3% [1], at least 20 [1], not 33% [1] or 32.1 [1]."
    )
    assert ungrounded_numbers(answer, contexts) == ["33", "32.1"]


def test_ungrounded_numbers_uses_all_sources_when_nothing_is_cited() -> None:
    contexts = [ctx("a", "A 25% gain."), ctx("b", "Costs 40 dollars.")]
    assert ungrounded_numbers("A 25% gain costing 40 dollars, not 41.", contexts) == ["41"]


def test_judge_answer_parses_fenced_json_and_scores() -> None:
    judge = FakeLLM(
        'Here you go:\n```json\n{"claims": ['
        '{"claim": "A", "verdict": "Supported", "sources": [1]},'
        '{"claim": "B", "verdict": "contradicted", "sources": [2]},'
        '{"claim": "C", "verdict": "unsupported"},'
        '{"claim": "D", "verdict": "supported", "sources": [1]}]}\n```'
    )

    result = judge_answer("answer", [ctx("a", "text")], judge)

    assert result.error is None
    assert [c.verdict for c in result.claims] == [
        "supported", "contradicted", "unsupported", "supported"
    ]
    assert result.faithfulness == pytest.approx(0.5)
    system, user = judge.calls[0]
    assert "[1] a (source: arxiv)\ntext" in user and user.endswith("ANSWER:\nanswer")


@pytest.mark.parametrize("raw", [
    "I cannot do that.",
    '{"claims": "nope"}',
    '{"claims": [{"claim": "A", "verdict": "maybe"}]}',
    "{not json}",
])
def test_judge_answer_reports_bad_output(raw: str) -> None:
    result = judge_answer("answer", [ctx("a")], FakeLLM(raw))
    assert result.error is not None
    assert result.faithfulness is None


def test_score_answer_and_summary() -> None:
    good = AgentState(
        query="q1", answer="Supported claim of 80 [1].", contexts=(ctx("rel", "GWP 80"),)
    )
    bad = AgentState(query="q2", answer="Made up 99 [3].", contexts=(ctx("other", "nothing"),))
    judge = FakeLLM('{"claims": [{"claim": "x", "verdict": "supported", "sources": [1]}]}')

    results = [
        score_answer(good, 1.0, relevant=["rel"], judge=judge),
        score_answer(bad, 3.0, relevant=["rel"]),
    ]

    assert results[0].invalid_citations == [] and results[0].ungrounded_numbers == []
    assert results[1].invalid_citations == [3] and results[1].ungrounded_numbers == ["99"]
    assert results[1].judge is None
    summary = summarize(results)
    assert summary.mean_recall == 0.5
    assert summary.mrr == 0.5
    assert summary.answers_with_valid_citations == 1
    assert summary.answers_with_ungrounded_numbers == 1
    assert summary.mean_faithfulness == 1.0
    assert summary.mean_seconds == 2.0
