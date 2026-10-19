import hashlib
import math
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from climate_rag._core import Document
from climate_rag.knowledge_base import KnowledgeBase

FIXTURES = Path(__file__).parent / "fixtures"


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    name = "test:hash-bag-of-words"
    dimension = 256

    def _embed(self, text: str) -> NDArray[np.float32]:
        vec = np.zeros(self.dimension, dtype=np.float32)
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dimension] += 1.0
        return vec / (math.sqrt(float(vec @ vec)) or 1.0)

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.stack([self._embed(t) for t in texts]).astype(np.float32)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        return self._embed(text)


class FakeLLM:
    """Records prompts and returns a canned answer."""

    def __init__(self, answer: str = "Fake answer [1].") -> None:
        self.answer = answer
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.answer


@pytest.fixture
def kb() -> KnowledgeBase:
    kb = KnowledgeBase(HashEmbedder(), chunk_chars=2000)
    kb.add_documents([
        Document("paris", "The Paris Agreement temperature goal is well below 2 degrees.",
                 {"source": "UNFCCC", "title": "Paris Agreement", "url": "https://unfccc.int"}),
        Document("methane", "Methane emissions from agriculture and livestock drive warming.",
                 {"source": "arxiv", "title": "Methane"}),
        Document("solar", "Solar tracking systems increase panel efficiency.",
                 {"source": "TechReport"}),
    ])
    return kb


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()
