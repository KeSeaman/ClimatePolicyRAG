import asyncio
import logging
from collections.abc import Sequence

import httpx

from climate_rag._core import Document
from climate_rag.ingest.arxiv import ArxivIngestor
from climate_rag.ingest.base import USER_AGENT, Ingestor
from climate_rag.ingest.worldbank import WorldBankIngestor

logger = logging.getLogger(__name__)

# Broad enough to cover the eval questions in climate_rag.cli without naming them.
DEFAULT_TOPICS = (
    "Paris Agreement temperature goal",
    "carbon pricing developing countries",
    "renewable energy Europe",
    "forest carbon sequestration",
    "methane emissions",
    "nationally determined contributions",
    "sea level rise projections",
    "climate adaptation finance",
    "carbon capture storage",
    "agriculture greenhouse gas emissions",
)


async def ingest_topics(
    topics: Sequence[str],
    limit_per_topic: int = 5,
    timeout: float = 30.0,
    ingestors: Sequence[Ingestor] | None = None,
    transport: httpx.AsyncBaseTransport | None = None,
) -> list[Document]:
    """
    Fetch every (source, topic) pair concurrently over one shared client.
    A failing source is logged and skipped; results are de-duplicated by id.
    """
    sources: list[Ingestor] = (
        list(ingestors) if ingestors is not None else [ArxivIngestor(), WorldBankIngestor()]
    )
    jobs = [(src, topic) for src in sources for topic in topics]

    async with httpx.AsyncClient(
        timeout=timeout,
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        transport=transport,
    ) as client:
        results = await asyncio.gather(
            *(src.fetch(client, topic, limit_per_topic) for src, topic in jobs),
            return_exceptions=True,
        )

    docs: dict[str, Document] = {}
    for (src, topic), result in zip(jobs, results, strict=True):
        if isinstance(result, BaseException):
            logger.warning("%s failed for %r: %s", src.name, topic, result)
            continue
        for doc in result:
            docs.setdefault(doc.id, doc)
    return list(docs.values())
