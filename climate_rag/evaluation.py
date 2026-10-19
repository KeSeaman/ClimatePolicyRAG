"""
Scoring for `climate-rag eval`, cheapest layer first:

1. Retrieval: are the top-k chunks from documents labelled relevant? Needs gold labels
   made for the same corpus (checked via the index's corpus_id).
2. Deterministic answer checks: are citations valid, and does every number in the answer
   appear in a source it cites? No model involved, so results are exactly reproducible.
3. LLM judge (optional): is each claim in the answer supported by the sources?
"""

import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator

from climate_rag.agents.synthesis import format_sources
from climate_rag.llm import LLM
from climate_rag.models import AgentState, RetrievedContext

_CITATION = re.compile(r"\[(\d+(?:\s*[,–-]\s*\d+)*)\]")
# A number not glued to letters, so "CO2", "CH4" and "2050s" are not claims to check.
_NUMBER = re.compile(r"(?<![\w.])\d+(?:,\d{3})*(?:\.\d+)?(?!\w)")


class GoldQuestion(BaseModel):
    question: str
    # Ids of the ingested documents (not chunks) that answer the question.
    relevant: list[str] = Field(default_factory=list)


class GoldSet(BaseModel):
    corpus_id: str | None = None  # the labels are only meaningful for this corpus
    note: str = ""
    questions: list[GoldQuestion]

    @classmethod
    def load(cls, path: Path) -> "GoldSet":
        return cls.model_validate_json(path.read_text())


class RetrievalScore(BaseModel):
    recall: float  # relevant documents retrieved / min(relevant documents, k)
    reciprocal_rank: float  # 1 / rank of the first relevant document; 0 if none retrieved
    found: list[str]
    missed: list[str]


def score_retrieval(
    contexts: Sequence[RetrievedContext], relevant: Sequence[str]
) -> RetrievalScore | None:
    if not relevant:
        return None
    relevant_set = set(relevant)
    ranked = list(dict.fromkeys(c.parent_id for c in contexts))  # documents, best first
    found = [d for d in ranked if d in relevant_set]
    rr = next((1 / rank for rank, d in enumerate(ranked, 1) if d in relevant_set), 0.0)
    return RetrievalScore(
        recall=len(found) / max(min(len(relevant_set), len(contexts)), 1),
        reciprocal_rank=rr,
        found=found,
        missed=sorted(relevant_set - set(found)),
    )


def cited_sources(answer: str) -> list[int]:
    """Source numbers cited as [1], [1, 3] or [2-4]."""
    cited: set[int] = set()
    for group in _CITATION.findall(answer):
        for part in group.split(","):
            bounds = [int(x) for x in re.split(r"\s*[–-]\s*", part.strip())]
            cited.update(range(bounds[0], bounds[-1] + 1))
    return sorted(cited)


def _numbers(text: str) -> set[float]:
    return {float(m.replace(",", "")) for m in _NUMBER.findall(_CITATION.sub(" ", text))}


def _grounded(match: str, known: set[float]) -> bool:
    """True if a source number equals this one at the precision written ("32" matches 32.03)."""
    value = float(match.replace(",", ""))
    decimals = len(match.split(".")[1]) if "." in match else 0
    return any(round(k, decimals) == value for k in known)


def ungrounded_numbers(
    answer: str, contexts: Sequence[RetrievedContext], question: str = ""
) -> list[str]:
    """
    Numbers in the answer that appear in none of the sources it cites (or in no source,
    if it cites none), allowing rounding to the precision the answer uses. Integers below
    10 are skipped as they are mostly counts and list markers, as are numbers from the
    question. Unit conversions show up as false positives.
    """
    cited = [contexts[i - 1] for i in cited_sources(answer) if 1 <= i <= len(contexts)]
    known = _numbers(question)
    for c in cited or contexts:
        known |= _numbers(f"{c.title or ''} {c.content}")

    flagged: list[str] = []
    for match in _NUMBER.findall(_CITATION.sub(" ", answer)):
        value = float(match.replace(",", ""))
        trivial = value.is_integer() and value < 10 and "." not in match
        if not trivial and not _grounded(match, known) and match not in flagged:
            flagged.append(match)
    return flagged


JUDGE_SYSTEM = (
    "You are a strict fact-checker for a retrieval-augmented question answering system. "
    "Break the ANSWER into its individual factual claims, ignoring filler and hedging. "
    "Judge each claim using ONLY the numbered SOURCES, never outside knowledge:\n"
    '- "supported": the sources state it or directly imply it\n'
    '- "contradicted": the sources state something incompatible with it\n'
    '- "unsupported": the sources do not say it\n'
    "Respond with JSON only, no prose, in exactly this shape:\n"
    '{"claims": [{"claim": "...", "verdict": "supported", "sources": [1]}]}'
)


class JudgedClaim(BaseModel):
    claim: str
    verdict: Literal["supported", "unsupported", "contradicted"]
    sources: list[int] = Field(default_factory=list)

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalise_verdict(cls, value: Any) -> Any:
        return value.strip().lower() if isinstance(value, str) else value


class JudgeResult(BaseModel):
    claims: list[JudgedClaim] = Field(default_factory=list)
    error: str | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def faithfulness(self) -> float | None:
        """Share of claims the judge found supported."""
        if not self.claims:
            return None
        return sum(c.verdict == "supported" for c in self.claims) / len(self.claims)


def judge_answer(answer: str, contexts: Sequence[RetrievedContext], judge: LLM) -> JudgeResult:
    raw = judge(JUDGE_SYSTEM, f"SOURCES:\n\n{format_sources(contexts)}\n\nANSWER:\n{answer}")
    # Models often wrap JSON in prose or code fences; take the outermost object.
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end < start:
        return JudgeResult(error=f"no JSON in judge output: {raw[:200]!r}")
    try:
        data = json.loads(raw[start : end + 1])
        claims = data.get("claims") if isinstance(data, dict) else None
        if not isinstance(claims, list):
            return JudgeResult(error="judge output has no 'claims' list")
        return JudgeResult.model_validate({"claims": claims})
    except (json.JSONDecodeError, ValidationError) as e:
        return JudgeResult(error=f"unparseable judge output: {e}")


class AnswerScores(BaseModel):
    question: str
    intent: str
    search_queries: list[str] = Field(default_factory=list)
    answer: str
    seconds: float
    contexts: list[RetrievedContext]
    retrieval: RetrievalScore | None
    cited: list[int]
    invalid_citations: list[int]
    ungrounded_numbers: list[str]
    judge: JudgeResult | None = None


def score_answer(
    state: AgentState,
    seconds: float,
    relevant: Sequence[str] = (),
    judge: LLM | None = None,
) -> AnswerScores:
    answer = state.answer or ""
    cited = cited_sources(answer)
    return AnswerScores(
        question=state.query,
        intent=state.intent.value,
        search_queries=list(state.search_queries),
        answer=answer,
        seconds=seconds,
        contexts=list(state.contexts),
        retrieval=score_retrieval(state.contexts, relevant),
        cited=cited,
        invalid_citations=[i for i in cited if not 1 <= i <= len(state.contexts)],
        ungrounded_numbers=ungrounded_numbers(answer, state.contexts, state.query),
        judge=judge_answer(answer, state.contexts, judge) if judge and state.contexts else None,
    )


class EvalSummary(BaseModel):
    questions: int
    mean_recall: float | None
    mrr: float | None
    answers_with_valid_citations: int
    answers_with_ungrounded_numbers: int
    mean_faithfulness: float | None
    judge_errors: int
    mean_seconds: float


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(results: Sequence[AnswerScores]) -> EvalSummary:
    retrieval = [r.retrieval for r in results if r.retrieval]
    faithfulness = [
        r.judge.faithfulness for r in results if r.judge and r.judge.faithfulness is not None
    ]
    return EvalSummary(
        questions=len(results),
        mean_recall=_mean([s.recall for s in retrieval]),
        mrr=_mean([s.reciprocal_rank for s in retrieval]),
        answers_with_valid_citations=sum(
            bool(r.cited) and not r.invalid_citations for r in results
        ),
        answers_with_ungrounded_numbers=sum(bool(r.ungrounded_numbers) for r in results),
        mean_faithfulness=_mean(faithfulness),
        judge_errors=sum(bool(r.judge and r.judge.error) for r in results),
        mean_seconds=_mean([r.seconds for r in results]) or 0.0,
    )


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def render_markdown(
    results: Sequence[AnswerScores], summary: EvalSummary, header: Sequence[str]
) -> str:
    n = summary.questions
    lines = [
        "# Climate RAG: Evaluation Run",
        "",
        "Generated by `climate-rag eval`; every answer below is verbatim model output.",
        "",
        *header,
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Retrieval recall@k (mean) | {_fmt(summary.mean_recall)} |",
        f"| Retrieval MRR | {_fmt(summary.mrr)} |",
        f"| Answers with valid citations | {summary.answers_with_valid_citations}/{n} |",
        f"| Answers with ungrounded numbers | {summary.answers_with_ungrounded_numbers}/{n} |",
        f"| Judge faithfulness (mean) | {_fmt(summary.mean_faithfulness)} |",
        f"| Mean time per question | {summary.mean_seconds:.1f}s |",
        "",
        "Recall and MRR are document-level against `eval/gold.json`. Ungrounded numbers are "
        "figures that appear in no cited source; check each before treating it as an error. "
        "Source scores are cosine similarity for plain vector search and fused rank scores "
        "otherwise.",
        "",
        "---",
    ]
    for i, r in enumerate(results, 1):
        checks = []
        if r.retrieval:
            checks.append(
                f"recall {r.retrieval.recall:.2f} · RR {r.retrieval.reciprocal_rank:.2f}"
            )
        if not r.cited:
            checks.append("**no citations**")
        elif r.invalid_citations:
            checks.append(f"**invalid citations** {r.invalid_citations}")
        else:
            checks.append("citations ok")
        if r.ungrounded_numbers:
            checks.append(f"**ungrounded numbers**: {', '.join(r.ungrounded_numbers)}")
        if r.judge:
            if r.judge.error:
                checks.append(f"judge error: {r.judge.error}")
            elif r.judge.faithfulness is not None:
                supported = sum(c.verdict == "supported" for c in r.judge.claims)
                checks.append(
                    f"faithfulness {r.judge.faithfulness:.2f} "
                    f"({supported}/{len(r.judge.claims)} claims)"
                )

        lines += [
            "", f"## {i}. {r.question}", "",
            f"**Intent**: `{r.intent}` · **Time**: {r.seconds:.1f}s", "",
            f"**Checks**: {' · '.join(checks)}", "",
        ]
        if len(r.search_queries) > 1:
            lines += ["**Search queries**: " + " · ".join(f"*{q}*" for q in r.search_queries), ""]
        lines += [r.answer, ""]
        flagged = [c for c in (r.judge.claims if r.judge else []) if c.verdict != "supported"]
        if flagged:
            lines += ["<details><summary>Claims the judge did not find supported</summary>", ""]
            lines += [f"- *{c.verdict}*: {c.claim}" for c in flagged]
            lines += ["", "</details>", ""]

        relevant = set(r.retrieval.found) if r.retrieval else set()
        lines += ["| # | Score | Source | Document | Relevant |", "|---|---|---|---|---|"]
        for j, c in enumerate(r.contexts, 1):
            title = (c.title or c.doc_id).replace("|", "\\|")
            link = f"[{title}]({c.url})" if c.url else title
            mark = ("✓" if c.parent_id in relevant else "") if r.retrieval else ""
            lines.append(f"| {j} | {c.score:.3f} | {c.source} | {link} | {mark} |")
    return "\n".join(lines) + "\n"
