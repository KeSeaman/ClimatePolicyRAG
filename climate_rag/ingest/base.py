import asyncio
import logging
from typing import Any, Protocol

import httpx

from climate_rag._core import Document

logger = logging.getLogger(__name__)

USER_AGENT = "climate-rag/0.2 (+https://github.com/captc/ClimatePolicyRAG)"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class Ingestor(Protocol):
    name: str

    async def fetch(self, client: httpx.AsyncClient, query: str, limit: int) -> list[Document]:
        """
        Fetch documents from the external source and normalize them into
        the core Document definition.
        """
        ...


async def get_with_retry(
    client: httpx.AsyncClient,
    url: str,
    params: dict[str, Any] | None = None,
    attempts: int = 3,
    backoff: float = 2.0,
    max_retry_after: float = 60.0,
) -> httpx.Response:
    """
    GET with exponential backoff on transport errors and 429/5xx responses,
    honouring a numeric Retry-After header (capped at max_retry_after seconds).
    """
    for attempt in range(1, attempts + 1):
        delay = backoff * 2 ** (attempt - 1)
        try:
            response = await client.get(url, params=params)
            if response.status_code not in RETRY_STATUSES or attempt == attempts:
                response.raise_for_status()
                return response
            reason = f"HTTP {response.status_code}"
            retry_after = response.headers.get("Retry-After", "")
            if retry_after.isdigit():
                delay = max(delay, min(float(retry_after), max_retry_after))
        except httpx.TransportError as e:
            if attempt == attempts:
                raise
            reason = repr(e)
        logger.warning("GET %s failed (%s); retrying in %.0fs", url, reason, delay)
        await asyncio.sleep(delay)
    raise AssertionError("unreachable")
