"""`ingest` then `eval` end to end, with fake data sources, embedder and LLM."""

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from climate_rag import cli
from climate_rag._core import Document
from climate_rag.config import Settings
from climate_rag.knowledge_base import KnowledgeBase

from .conftest import FakeLLM, HashEmbedder

DOCS = [
    Document("paris", "The Paris Agreement temperature goal is well below 2 degrees.",
             {"source": "UNFCCC", "title": "Paris"}),
    Document("methane", "Methane emissions from livestock drive warming.",
             {"source": "arxiv", "title": "Methane"}),
]


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> FakeLLM:
    async def fake_ingest(topics: Sequence[str], *args: Any) -> list[Document]:
        return DOCS

    llm = FakeLLM("Well below 2 degrees [1].")
    monkeypatch.setattr(cli, "ingest_topics", fake_ingest)
    monkeypatch.setattr(cli, "make_embedder", lambda settings: HashEmbedder())
    monkeypatch.setattr(cli, "make_llm", lambda settings: llm)
    monkeypatch.setattr(cli, "make_judge", lambda settings: FakeLLM(
        '{"claims": [{"claim": "below 2 degrees", "verdict": "supported", "sources": [1]}]}'
    ))
    return llm


def test_ingest_then_eval(fakes: FakeLLM, tmp_path: Path) -> None:
    index = tmp_path / "index"
    cli.main(["ingest", "--out", str(index), "--topic", "paris"])
    manifest = json.loads((index / "manifest.json").read_text())
    assert manifest["documents"] == 2 and manifest["topics"] == ["paris"]

    corpus_id = KnowledgeBase.load(index, HashEmbedder()).corpus_id()
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"corpus_id": corpus_id, "questions": [
        {"question": "What is the Paris Agreement temperature goal?", "relevant": ["paris"]},
    ]}))
    report = tmp_path / "report.md"

    cli.main(["eval", "--index", str(index), "--gold", str(gold), "--judge",
              "--output", str(report)])

    results = json.loads(report.with_suffix(".json").read_text())
    assert results["summary"]["mean_recall"] == 1.0
    assert results["summary"]["mean_faithfulness"] == 1.0
    assert results["corpus"]["corpus_id"] == corpus_id
    text = report.read_text()
    assert "| Retrieval recall@k (mean) | 1.00 |" in text
    assert "citations ok" in text
    assert "Well below 2 degrees [1]." in text


def test_eval_skips_retrieval_metrics_for_another_corpus(fakes: FakeLLM, tmp_path: Path) -> None:
    index = tmp_path / "index"
    cli.main(["ingest", "--out", str(index)])
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"corpus_id": "something-else", "questions": [
        {"question": "Methane?", "relevant": ["methane"]},
    ]}))

    cli.main(["eval", "--index", str(index), "--gold", str(gold),
              "--output", str(tmp_path / "r.md")])

    summary = json.loads((tmp_path / "r.json").read_text())["summary"]
    assert summary["mean_recall"] is None
    assert summary["mean_faithfulness"] is None  # judge not requested


def test_missing_index_is_a_clear_error(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="No saved index"):
        cli.load_kb(Settings(), tmp_path / "nope", reembed=False)


def test_retrieval_only_eval_prints_scores(
    fakes: FakeLLM, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    index = tmp_path / "index"
    cli.main(["ingest", "--out", str(index)])
    corpus_id = KnowledgeBase.load(index, HashEmbedder()).corpus_id()
    gold = tmp_path / "gold.json"
    gold.write_text(json.dumps({"corpus_id": corpus_id, "questions": [
        {"question": "Methane emissions from livestock?", "relevant": ["methane"]},
    ]}))

    cli.main(["eval", "--index", str(index), "--gold", str(gold), "--retrieval-only",
              "--retrieval", "hybrid", "-k", "1"])

    out = capsys.readouterr().out
    assert "Retrieval: hybrid, k = 1" in out
    assert "1.000 1.000  mean recall / MRR" in out
    assert fakes.calls == []  # no answers generated
