import hashlib
import json
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from climate_rag._core import Document, SearchResult, VectorIndex
from climate_rag.embeddings import Embedder

SearchMode = Literal["vector", "hybrid"]

# Standard RRF constant (Cormack et al., 2009): damps the weight of top ranks so no single
# ranking dominates the fused order.
RRF_K = 60

MANIFEST_FILE = "manifest.json"
DOCUMENTS_FILE = "documents.jsonl"  # written by the Rust index alongside embeddings.f32
FORMAT_VERSION = 1

# Split after sentence-ending punctuation or at blank lines.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class IndexMismatchError(ValueError):
    """A saved index was built with a different embedder than the one configured."""


def chunk_document(doc: Document, max_chars: int) -> list[Document]:
    """Split a long document into sentence-aligned chunks of at most max_chars."""
    if len(doc.content) <= max_chars:
        return [doc]

    pieces: list[str] = []
    current = ""
    for sentence in _BOUNDARY.split(doc.content):
        # A single over-long sentence is hard-split rather than dropped.
        while len(sentence) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + 1 + len(sentence) > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)

    return [
        Document(f"{doc.id}#{i}", text, {**doc.metadata, "parent_id": doc.id})
        for i, text in enumerate(pieces)
    ]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[SearchResult]], k: int
) -> list[SearchResult]:
    """
    Merge ranked lists by summing 1 / (RRF_K + rank) per document. Rank-based, so it can
    combine rankings whose scores are not comparable (cosine similarity vs BM25). Each
    returned result's score is replaced by its fused score.
    """
    fused: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, 1):
            fused[result.doc_id] = fused.get(result.doc_id, 0.0) + 1 / (RRF_K + rank)
            first_seen.setdefault(result.doc_id, result)
    # sorted() is stable, so ties keep the order in which documents were first ranked.
    best = sorted(fused, key=lambda doc_id: -fused[doc_id])[:k]
    for doc_id in best:
        first_seen[doc_id].score = fused[doc_id]
    return [first_seen[doc_id] for doc_id in best]


def read_documents(path: Path) -> list[Document]:
    """The chunks stored in a saved index, without their embeddings."""
    docs = []
    for line in (path / DOCUMENTS_FILE).read_text().splitlines():
        if line.strip():
            raw = json.loads(line)
            docs.append(Document(raw["id"], raw["content"], raw["metadata"]))
    return docs


class KnowledgeBase:
    """The Rust vector index together with the embedder that fills and queries it."""

    def __init__(
        self, embedder: Embedder, chunk_chars: int = 2000, index: VectorIndex | None = None
    ) -> None:
        self.embedder = embedder
        self.index = index if index is not None else VectorIndex(embedder.dimension)
        self.chunk_chars = chunk_chars
        # Set by save()/load(): where the corpus came from and how it was embedded.
        self.manifest: dict[str, Any] | None = None

    def add_documents(self, docs: Sequence[Document]) -> int:
        """Chunk, embed (as one batch) and index docs. Returns the number of chunks indexed."""
        chunks = [c for doc in docs for c in chunk_document(doc, self.chunk_chars)]
        if chunks:
            embeddings = self.embedder.embed_documents([c.content for c in chunks])
            self.index.add_batch(chunks, embeddings)
        return len(chunks)

    def search(
        self, queries: str | Sequence[str], k: int, mode: SearchMode = "vector"
    ) -> list[SearchResult]:
        """
        Top-k chunks for one query or several phrasings of it. A single vector query
        returns cosine scores; anything fused (hybrid mode, several queries) returns RRF
        scores, which only order results and are not comparable to cosine similarity.
        """
        queries = [queries] if isinstance(queries, str) else list(queries)
        if mode == "vector" and len(queries) == 1:
            return self.index.search(self.embedder.embed_query(queries[0]), k)

        # Each ranking contributes a deeper candidate pool than k so fusion can promote
        # documents that rank moderately in several lists.
        pool = max(4 * k, 20)
        rankings: list[list[SearchResult]] = []
        for query in queries:
            rankings.append(self.index.search(self.embedder.embed_query(query), pool))
            if mode == "hybrid":
                rankings.append(self.index.keyword_search(query, pool))
        return reciprocal_rank_fusion(rankings, k)

    def corpus_id(self) -> str:
        """Fingerprint of the indexed chunk ids; ties relevance labels to one corpus."""
        ids = "\n".join(sorted(self.index.ids()))
        return hashlib.sha256(ids.encode()).hexdigest()[:12]

    def save(self, path: Path, **info: Any) -> dict[str, Any]:
        """Write the index plus a manifest; `info` adds provenance such as the topics."""
        self.index.save(path)
        manifest = {
            "format_version": FORMAT_VERSION,
            "created": datetime.now(UTC).isoformat(timespec="seconds"),
            "embedder": self.embedder.name,
            "dimension": self.embedder.dimension,
            "chunk_chars": self.chunk_chars,
            "chunks": len(self),
            "corpus_id": self.corpus_id(),
            **info,
        }
        # Written last: a manifest only exists once the data files are complete.
        (path / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2) + "\n")
        self.manifest = manifest
        return manifest

    @classmethod
    def load(cls, path: Path, embedder: Embedder, reembed: bool = False) -> "KnowledgeBase":
        """
        Load a saved index. If it was built with a different embedder, raise
        IndexMismatchError, or with reembed=True re-embed the stored chunks instead.
        """
        manifest: dict[str, Any] = json.loads((path / MANIFEST_FILE).read_text())
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported index format {manifest.get('format_version')}")

        same_embedder = (manifest["embedder"], manifest["dimension"]) == (
            embedder.name,
            embedder.dimension,
        )
        if same_embedder:
            index = VectorIndex.load(path, manifest["dimension"])
            kb = cls(embedder, manifest["chunk_chars"], index)
        elif reembed:
            kb = cls(embedder, manifest["chunk_chars"])
            kb.add_documents(read_documents(path))
            manifest = {**manifest, "embedder": embedder.name, "dimension": embedder.dimension,
                        "reembedded": True}
        else:
            raise IndexMismatchError(
                f"{path} was embedded with {manifest['embedder']} ({manifest['dimension']} dims)"
                f" but the configured embedder is {embedder.name} ({embedder.dimension} dims)."
                " Use --reembed, or rebuild it with `climate-rag ingest`."
            )
        kb.manifest = manifest
        return kb

    def __len__(self) -> int:
        return len(self.index)
