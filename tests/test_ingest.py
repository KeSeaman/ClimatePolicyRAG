import asyncio
import json

import httpx
import pytest

from climate_rag._core import Document
from climate_rag.ingest import ingest_topics
from climate_rag.ingest.arxiv import ArxivIngestor, build_search_query, parse_atom
from climate_rag.ingest.base import get_with_retry
from climate_rag.ingest.worldbank import WorldBankIngestor, parse_projects

from .conftest import FIXTURES


def test_build_search_query_ands_terms_and_drops_stopwords() -> None:
    assert build_search_query("Carbon pricing in the EU") == "all:carbon AND all:pricing AND all:eu"


def test_parse_atom() -> None:
    docs = parse_atom((FIXTURES / "arxiv.xml").read_text())

    # The entry without an abstract is skipped; a missing author is tolerated.
    assert [d.id for d in docs] == [
        "http://arxiv.org/abs/2401.00001v1",
        "http://arxiv.org/abs/2401.00003v1",
    ]
    first = docs[0]
    assert first.content == (
        "Title: Carbon Pricing and Distributional Outcomes\n"
        "Abstract: We study how carbon taxes affect households in developing economies."
    )
    assert first.metadata["authors"] == "Ada Lovelace, Alan Turing"
    assert first.metadata["published"] == "2024-01-02"
    assert first.metadata["source"] == "arxiv"
    assert docs[1].metadata["authors"] == ""


def test_parse_projects() -> None:
    docs = parse_projects(json.loads((FIXTURES / "worldbank_projects.json").read_text()))

    assert [d.id for d in docs] == ["wb_P100001", "wb_P100002"]
    assert docs[0].content.startswith(
        "World Bank project: Coastal Resilience Program "
        "(Bangladesh, approved 2025-03-01, Active, commitment US$250.0M)"
    )
    assert docs[0].metadata["url"].endswith("/P100001")
    assert "climate-smart agriculture" in docs[1].content
    assert "commitment" not in docs[1].content  # zero amounts are omitted


def mock_transport(responses: dict[str, list[httpx.Response]]) -> httpx.MockTransport:
    """Serve queued responses per host; record nothing else."""

    def handler(request: httpx.Request) -> httpx.Response:
        return responses[request.url.host].pop(0)

    return httpx.MockTransport(handler)


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make retry backoff instant."""

    async def instant(_: float) -> None:
        return None

    monkeypatch.setattr("climate_rag.ingest.base.asyncio.sleep", instant)


@pytest.mark.usefixtures("no_sleep")
def test_fetch_retries_transient_errors() -> None:
    xml = (FIXTURES / "arxiv.xml").read_text()
    transport = mock_transport({
        "export.arxiv.org": [httpx.Response(503), httpx.Response(200, text=xml)],
    })

    async def run() -> list[Document]:
        async with httpx.AsyncClient(transport=transport) as client:
            return await ArxivIngestor(min_interval=0).fetch(client, "carbon", 5)

    assert len(asyncio.run(run())) == 2


def test_retry_honours_retry_after(monkeypatch: pytest.MonkeyPatch) -> None:
    delays: list[float] = []

    async def record(seconds: float) -> None:
        delays.append(seconds)

    monkeypatch.setattr("climate_rag.ingest.base.asyncio.sleep", record)
    transport = mock_transport({"example.org": [
        httpx.Response(429, headers={"Retry-After": "7"}),
        httpx.Response(503, headers={"Retry-After": "9999"}),
        httpx.Response(200),
    ]})

    async def run() -> httpx.Response:
        async with httpx.AsyncClient(transport=transport) as client:
            return await get_with_retry(client, "https://example.org/x")

    assert asyncio.run(run()).status_code == 200
    assert delays == [7.0, 60.0]  # server hint, then capped


@pytest.mark.usefixtures("no_sleep")
def test_ingest_topics_dedupes_and_survives_a_failing_source() -> None:
    xml = (FIXTURES / "arxiv.xml").read_text()
    transport = mock_transport({
        # Both topics return the same two papers.
        "export.arxiv.org": [httpx.Response(200, text=xml), httpx.Response(200, text=xml)],
        # The World Bank is down for every attempt.
        "search.worldbank.org": [httpx.Response(500)] * 6,
    })

    docs = asyncio.run(ingest_topics(
        ["carbon", "methane"],
        ingestors=[ArxivIngestor(min_interval=0), WorldBankIngestor()],
        transport=transport,
    ))

    assert sorted(d.id for d in docs) == [
        "http://arxiv.org/abs/2401.00001v1",
        "http://arxiv.org/abs/2401.00003v1",
    ]


@pytest.mark.network
def test_live_sources_return_on_topic_documents() -> None:
    docs = asyncio.run(ingest_topics(["carbon pricing"], limit_per_topic=3))
    sources = {d.metadata["source"] for d in docs}
    assert sources == {"arxiv", "world_bank_projects"}
    arxiv_text = " ".join(d.content.lower() for d in docs if d.metadata["source"] == "arxiv")
    assert "carbon" in arxiv_text
