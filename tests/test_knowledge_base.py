import json
from pathlib import Path

import pytest

from climate_rag._core import Document, SearchResult, VectorIndex
from climate_rag.knowledge_base import (
    IndexMismatchError,
    KnowledgeBase,
    read_documents,
    reciprocal_rank_fusion,
)

from .conftest import HashEmbedder


class OtherEmbedder(HashEmbedder):
    name = "test:other-model"


def test_save_and_load_preserves_search_results(kb: KnowledgeBase, tmp_path: Path) -> None:
    manifest = kb.save(tmp_path, topics=["methane"])
    loaded = KnowledgeBase.load(tmp_path, HashEmbedder())

    query = "methane emissions from livestock"
    assert [(r.doc_id, r.score) for r in loaded.search(query, 3)] == [
        (r.doc_id, r.score) for r in kb.search(query, 3)
    ]
    assert loaded.corpus_id() == kb.corpus_id() == manifest["corpus_id"]
    assert loaded.manifest == manifest
    assert manifest["embedder"] == "test:hash-bag-of-words"
    assert manifest["topics"] == ["methane"]
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_load_refuses_a_different_embedder(kb: KnowledgeBase, tmp_path: Path) -> None:
    kb.save(tmp_path)
    with pytest.raises(IndexMismatchError, match="test:hash-bag-of-words"):
        KnowledgeBase.load(tmp_path, OtherEmbedder())


def test_reembed_rebuilds_with_the_new_embedder(kb: KnowledgeBase, tmp_path: Path) -> None:
    kb.save(tmp_path)
    loaded = KnowledgeBase.load(tmp_path, OtherEmbedder(), reembed=True)

    assert loaded.manifest is not None
    assert loaded.manifest["embedder"] == "test:other-model"
    assert loaded.manifest["reembedded"] is True
    assert sorted(loaded.index.ids()) == sorted(kb.index.ids())


def test_corpus_id_depends_only_on_the_ids() -> None:
    a, b = KnowledgeBase(HashEmbedder()), KnowledgeBase(HashEmbedder())
    a.add_documents([Document("x", "one"), Document("y", "two")])
    b.add_documents([Document("y", "changed"), Document("x", "text")])
    assert a.corpus_id() == b.corpus_id()
    b.add_documents([Document("z", "three")])
    assert a.corpus_id() != b.corpus_id()


def test_read_documents_returns_stored_chunks(kb: KnowledgeBase, tmp_path: Path) -> None:
    kb.save(tmp_path)
    docs = read_documents(tmp_path)
    assert [d.id for d in docs] == kb.index.ids()
    assert docs[0].metadata["source"] == "UNFCCC"


def result(doc_id: str) -> SearchResult:
    ix = VectorIndex(1)
    ix.add(Document(doc_id, doc_id), [1.0])
    return ix.search([1.0], 1)[0]


def test_reciprocal_rank_fusion_rewards_agreement() -> None:
    a, b, c, d = (result(x) for x in "abcd")
    fused = reciprocal_rank_fusion([[a, b, c], [c, a, d]], k=3)

    assert [r.doc_id for r in fused] == ["a", "c", "b"]
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert reciprocal_rank_fusion([], k=3) == []


def test_hybrid_search_adds_exact_keyword_matches() -> None:
    # The hash embedder treats "ndc" and "ndcs" as unrelated tokens, so every vector score
    # is 0 and the NDC document ranks last (ties keep insertion order); BM25 stems the plural.
    kb = KnowledgeBase(HashEmbedder())
    kb.add_documents([
        Document("solar", "Solar panels convert sunlight into electricity."),
        Document("wind", "Wind turbines in the North Sea."),
        Document("ndc", "Countries submit NDCs to the UNFCCC every five years."),
    ])

    assert [r.doc_id for r in kb.search("ndc", 3, "vector")] == ["solar", "wind", "ndc"]
    hybrid = kb.search("ndc", 2, "hybrid")
    assert hybrid[0].doc_id == "ndc"
    assert hybrid[0].score == pytest.approx(1 / 61 + 1 / 63)  # vector rank 3, keyword rank 1


def test_multi_query_search_fuses_all_phrasings() -> None:
    kb = KnowledgeBase(HashEmbedder())
    kb.add_documents([Document("a", "alpha beta"), Document("g", "gamma delta")])
    assert {r.doc_id for r in kb.search(["alpha", "gamma"], 2)} == {"a", "g"}
