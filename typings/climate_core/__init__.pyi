"""Type stubs for climate_core – a Rust/PyO3 native extension module."""

from typing import Dict, List, Optional

class Document:
    """A document with id, content, and metadata."""
    id: str
    content: str
    metadata: Dict[str, str]

    def __init__(
        self,
        id: str,
        content: str,
        metadata: Optional[Dict[str, str]] = None,
    ) -> None: ...

class SearchResult:
    """A search result from VectorIndex.search()."""
    doc_id: str
    score: float
    content_preview: str

class VectorIndex:
    """An in-memory vector index backed by cosine similarity search."""

    def __init__(self, dimension: int) -> None: ...
    def add(self, doc: Document, embedding: List[float]) -> None: ...
    def search(self, query_embedding: List[float], k: int) -> List[SearchResult]: ...
