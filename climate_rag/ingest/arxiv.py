import asyncio
import re
import time
import xml.etree.ElementTree as ET

import httpx

from climate_rag._core import Document
from climate_rag.ingest.base import get_with_retry

_NS = {"atom": "http://www.w3.org/2005/Atom"}
_STOPWORDS = frozenset({"a", "an", "and", "the", "of", "in", "on", "for", "to", "by", "with"})


def build_search_query(query: str) -> str:
    """
    AND together every term, e.g. "carbon pricing" -> "all:carbon AND all:pricing".
    A bare "all:carbon pricing" only applies the field to the first word and returns
    near-random results once sorted by date.
    """
    terms = [t for t in re.findall(r"[\w-]+", query.lower()) if t not in _STOPWORDS]
    return " AND ".join(f"all:{t}" for t in terms)


def _text(node: ET.Element, path: str) -> str:
    found = node.find(path, _NS)
    return " ".join((found.text or "").split()) if found is not None else ""


def parse_atom(xml_content: str) -> list[Document]:
    docs = []
    for entry in ET.fromstring(xml_content).findall("atom:entry", _NS):
        doc_id, title, summary = (_text(entry, f"atom:{f}") for f in ("id", "title", "summary"))
        # Error responses come back as a single entry without an abstract.
        if not (doc_id and title and summary):
            continue
        authors = [_text(a, "atom:name") for a in entry.findall("atom:author", _NS)]

        docs.append(Document(
            id=doc_id,
            content=f"Title: {title}\nAbstract: {summary}",
            metadata={
                "source": "arxiv",
                "title": title,
                "authors": ", ".join(a for a in authors if a),
                "published": _text(entry, "atom:published")[:10],
                "url": doc_id,
            },
        ))
    return docs


class ArxivIngestor:
    name = "arxiv"
    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, min_interval: float = 3.0) -> None:
        # arXiv's API terms ask for at most one request every 3 seconds.
        self._min_interval = min_interval
        self._lock = asyncio.Lock()
        self._last_request = float("-inf")

    async def fetch(self, client: httpx.AsyncClient, query: str, limit: int = 5) -> list[Document]:
        params = {
            "search_query": build_search_query(query),
            "start": 0,
            "max_results": limit,
            "sortBy": "relevance",
        }
        async with self._lock:
            wait = self._last_request + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            try:
                response = await get_with_retry(client, self.BASE_URL, params)
            finally:
                self._last_request = time.monotonic()
        return parse_atom(response.text)
