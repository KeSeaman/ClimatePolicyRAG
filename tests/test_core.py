"""Behaviour of the Rust VectorIndex as seen from Python."""

import math
import random
from pathlib import Path

import numpy as np
import pytest

from climate_rag._core import Document, VectorIndex


def doc(i: int, content: str = "text") -> Document:
    return Document(f"d{i}", content, {"n": str(i)})


def test_search_ranks_by_cosine_similarity_and_returns_full_documents() -> None:
    ix = VectorIndex(3)
    ix.add_batch([doc(0, "x" * 500), doc(1), doc(2)], [[1, 0, 0], [0, 1, 0], [1, 1, 0]])

    results = ix.search([2, 0, 0], 3)

    assert [r.doc_id for r in results] == ["d0", "d2", "d1"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(2**-0.5)
    assert results[0].content == "x" * 500  # full text, not a preview
    assert results[0].metadata == {"n": "0"}


def test_matches_brute_force_on_random_data() -> None:
    rng = random.Random(0)
    dim, n = 32, 500
    rows = [[rng.gauss(0, 1) for _ in range(dim)] for _ in range(n)]
    query = [rng.gauss(0, 1) for _ in range(dim)]
    ix = VectorIndex(dim)
    ix.add_batch([doc(i) for i in range(n)], rows)

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))

    expected = sorted(range(n), key=lambda i: -cosine(query, rows[i]))[:10]
    assert [r.doc_id for r in ix.search(query, 10)] == [f"d{i}" for i in expected]


def test_re_adding_an_id_replaces_it() -> None:
    ix = VectorIndex(2)
    ix.add(Document("a", "old"), [1, 0])
    ix.add_batch([Document("a", "new"), Document("b", "other")], [[0, 1], [1, 0]])

    assert len(ix) == 2
    assert "a" in ix and "zzz" not in ix
    top = ix.search([0, 1], 1)[0]
    assert (top.doc_id, top.content) == ("a", "new")


def test_edge_cases() -> None:
    ix = VectorIndex(2)
    assert ix.search([1, 0], 5) == []
    ix.add_batch([doc(0), doc(1), doc(2)], [[1, 0], [1, 0], [0, 0]])
    assert len(ix.search([1, 0], 99)) == 3
    assert ix.search([1, 0], 0) == []
    # Equal scores keep insertion order; zero vectors score 0.
    assert [r.doc_id for r in ix.search([1, 0], 3)] == ["d0", "d1", "d2"]
    assert ix.search([1, 0], 3)[2].score == 0.0


@pytest.mark.parametrize("call", [
    lambda ix: ix.add(doc(0), [1.0]),
    lambda ix: ix.search([1.0, 2.0, 3.0], 1),
    lambda ix: ix.add_batch([doc(0)], []),
    lambda ix: ix.add_batch([doc(0), doc(1)], [[1.0, 0.0], [1.0]]),
])
def test_rejects_bad_input(call) -> None:  # type: ignore[no-untyped-def]
    ix = VectorIndex(2)
    with pytest.raises(ValueError):
        call(ix)
    assert len(ix) == 0  # failed batches insert nothing


def test_zero_dimension_rejected() -> None:
    with pytest.raises(ValueError):
        VectorIndex(0)


def test_numpy_float32_input_matches_list_input() -> None:
    rng = np.random.default_rng(0)
    rows = rng.standard_normal((50, 8)).astype(np.float32)
    query = rng.standard_normal(8).astype(np.float32)
    from_lists, from_numpy = VectorIndex(8), VectorIndex(8)
    from_lists.add_batch([doc(i) for i in range(50)], rows.tolist())
    from_numpy.add_batch([doc(i) for i in range(50)], rows)

    expected = [(r.doc_id, r.score) for r in from_lists.search(query.tolist(), 5)]
    assert [(r.doc_id, r.score) for r in from_numpy.search(query, 5)] == expected
    # Non-contiguous views and float64 arrays take the slower conversion paths.
    strided = VectorIndex(8)
    strided.add_batch([doc(i) for i in range(50)], np.asfortranarray(rows))
    assert [(r.doc_id, r.score) for r in strided.search(query.astype(np.float64), 5)] == expected


def test_numpy_input_is_validated() -> None:
    ix = VectorIndex(4)
    with pytest.raises(ValueError, match="dimension mismatch"):
        ix.add_batch([doc(0)], np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="1 documents but 2 embeddings"):
        ix.add_batch([doc(0)], np.zeros((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="dimension mismatch"):
        ix.search(np.zeros(5, dtype=np.float32), 1)


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    ix = VectorIndex(3)
    ix.add_batch([Document("a", "alpha", {"k": "v"}), Document("b", "beta")],
                 [[1, 0, 0], [0, 1, 1]])
    ix.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, 3)

    assert loaded.ids() == ["a", "b"]
    top = loaded.search([0, 1, 1], 1)[0]
    assert (top.doc_id, top.content) == ("b", "beta")
    assert top.score == pytest.approx(1.0)
    assert loaded.search([1, 0, 0], 1)[0].metadata == {"k": "v"}
    # The id map is rebuilt, so re-adding still replaces instead of duplicating.
    loaded.add(Document("a", "alpha v2"), [1, 0, 0])
    assert len(loaded) == 2


def test_load_rejects_inconsistent_files(tmp_path: Path) -> None:
    ix = VectorIndex(3)
    ix.add(Document("a", "alpha"), [1, 0, 0])
    ix.save(tmp_path)
    with pytest.raises(ValueError, match="bytes"):
        VectorIndex.load(tmp_path, 4)  # wrong dimension for the stored floats
    bad_line = "not json"
    (tmp_path / "documents.jsonl").write_text(
        '{"id": "a", "content": "x", "metadata": {}}\n' + bad_line + "\n"
    )
    with pytest.raises(ValueError, match="line 2"):
        VectorIndex.load(tmp_path, 3)
    with pytest.raises(OSError):
        VectorIndex.load(tmp_path / "missing", 3)


def test_keyword_search_ranks_by_bm25_and_survives_reload(tmp_path: Path) -> None:
    ix = VectorIndex(2)
    ix.add_batch([
        Document("mrv", "Monitoring, reporting and verification of methane emissions"),
        Document("tax", "Carbon tax and carbon pricing effects on emissions"),
        Document("sea", "Sea level rise projections"),
    ], [[1, 0], [0, 1], [1, 1]])

    # Only inflections are stemmed: "reported" and "reporting" both become "report".
    assert [r.doc_id for r in ix.keyword_search("How is methane reported?", 5)] == ["mrv"]
    assert ix.keyword_search("emissions", 5)[0].score > 0
    assert ix.keyword_search("the of and", 5) == []  # stopwords only

    ix.save(tmp_path)
    loaded = VectorIndex.load(tmp_path, 2)
    assert [r.doc_id for r in loaded.keyword_search("sea level", 5)] == ["sea"]
    loaded.add(Document("sea", "Glacier melt"), [1, 1])  # replacement drops old terms
    assert loaded.keyword_search("sea level", 5) == []
    assert [r.doc_id for r in loaded.keyword_search("glacier", 5)] == ["sea"]
