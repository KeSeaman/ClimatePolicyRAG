import os
from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from climate_rag.config import Settings


class Embedder(Protocol):
    # Backend + model, recorded in saved indexes so a mismatched index is refused on load.
    name: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Shape (len(texts), dimension); float32 so the Rust index reads it in place."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Shape (dimension,)."""
        ...


class FastEmbedEmbedder:
    """Local ONNX embedding model (default: BAAI/bge-small-en-v1.5, 384 dims)."""

    def __init__(self, model_name: str, cache_dir: str | None = None) -> None:
        # Imported lazily: onnxruntime takes a second to load and tests never need it.
        from fastembed import TextEmbedding

        # An explicit thread count stops onnxruntime pinning threads to CPUs this process
        # may not own (cgroup/cpuset limits, containers), which otherwise logs errors.
        threads = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        self._model = TextEmbedding(model_name, cache_dir=cache_dir, threads=threads)
        self.name = f"fastembed:{model_name}"
        self.dimension: int = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        # passage_embed/query_embed add the instruction prefixes some models (e.g. BGE) expect.
        vectors = list(self._model.passage_embed(list(texts)))
        if not vectors:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.stack(vectors).astype(np.float32, copy=False)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector = next(iter(self._model.query_embed(text)))
        return np.asarray(vector, dtype=np.float32)


class OpenAIEmbedder:
    """Any OpenAI-compatible /embeddings endpoint."""

    def __init__(self, model_name: str, base_url: str, api_key: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model_name
        self.name = f"openai:{model_name}"
        self.dimension: int = len(self.embed_query("dimension probe"))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        response = self._client.embeddings.create(model=self._model, input=list(texts))
        rows = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        return np.asarray(rows, dtype=np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        vector: NDArray[np.float32] = self.embed_documents([text])[0]
        return vector


def make_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "openai":
        return OpenAIEmbedder(settings.embedding_model, settings.llm_base_url, settings.llm_api_key)
    return FastEmbedEmbedder(settings.embedding_model, str(settings.embedding_cache_dir))
