from typing import Any

import httpx

from climate_rag._core import Document
from climate_rag.ingest.base import get_with_retry

PROJECT_URL = "https://projects.worldbank.org/en/projects-operations/project-detail/{id}"


def _as_text(value: Any) -> str:
    # Older API versions wrap long text fields as {"cdata": "..."}.
    if isinstance(value, dict):
        value = value.get("cdata", "")
    return " ".join(str(value or "").split())


def parse_projects(payload: dict[str, Any]) -> list[Document]:
    docs = []
    for project in (payload.get("projects") or {}).values():
        project_id = _as_text(project.get("id"))
        name = _as_text(project.get("project_name"))
        abstract = _as_text(project.get("project_abstract"))
        if not (project_id and name and abstract):
            continue
        country = _as_text(project.get("countryshortname")) or "unknown country"
        approved = _as_text(project.get("boardapprovaldate"))[:10]
        status = _as_text(project.get("status"))
        amount = _as_text(project.get("totalamt")).replace(",", "")

        facts = [country, f"approved {approved}" if approved else "", status]
        if amount.isdigit() and int(amount) > 0:
            facts.append(f"commitment US${int(amount) / 1e6:,.1f}M")
        header = f"World Bank project: {name} ({', '.join(f for f in facts if f)})"

        docs.append(Document(
            id=f"wb_{project_id}",
            content=f"{header}\n{abstract}",
            metadata={
                "source": "world_bank_projects",
                "title": name,
                "country": country,
                "approved": approved,
                "url": PROJECT_URL.format(id=project_id),
            },
        ))
    return docs


class WorldBankIngestor:
    """World Bank project abstracts: what is actually being financed and delivered."""

    name = "world_bank"
    BASE_URL = "https://search.worldbank.org/api/v3/projects"
    FIELDS = "id,project_name,project_abstract,countryshortname,boardapprovaldate,status,totalamt"

    async def fetch(self, client: httpx.AsyncClient, query: str, limit: int = 5) -> list[Document]:
        params = {"format": "json", "qterm": query, "rows": limit, "fl": self.FIELDS}
        response = await get_with_retry(client, self.BASE_URL, params)
        return parse_projects(response.json())
