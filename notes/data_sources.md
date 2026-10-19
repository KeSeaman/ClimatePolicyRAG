# Climate Data Sources

Survey of candidate sources. **Integrated** ones have an adapter in `climate_rag/ingest/`.

## 1. Scientific & Climate Data
- **NASA EarthData**: [https://www.earthdata.nasa.gov/](https://www.earthdata.nasa.gov/)
    - API: `https://cmr.earthdata.nasa.gov/search/`
- **ArXiv** (**integrated**): [https://arxiv.org/help/api](https://arxiv.org/help/api)
    - API: `https://export.arxiv.org/api/query` (Atom). Max 1 request / 3 s.

## 2. Policy Data
- **UNFCCC Documents**: [https://unfccc.int/documents](https://unfccc.int/documents)
    - API: `https://unfccc.int/api/documents`
- **Climate Laws**: [https://www.climate-laws.org/](https://www.climate-laws.org/)
    - API: `https://www.climate-laws.org/api`

## 3. Implementation Data
- **Climate Action Tracker**: [https://climateactiontracker.org/data-portal/](https://climateactiontracker.org/data-portal/)
    - JSON: `/api/v1/countries`, `/api/v1/sectors`
- **CDP (Carbon Disclosure Project)**: [https://data.cdp.net/](https://data.cdp.net/)
    - Socrata API: `https://dev.socrata.com/foundry/data.cdp.net`
- **World Bank Projects** (**integrated**): [https://projects.worldbank.org/](https://projects.worldbank.org/)
    - API: `https://search.worldbank.org/api/v3/projects?format=json&qterm=...&fl=...`
    - `project_abstract` is only returned when requested via `fl`.
- **World Bank Climate Knowledge Portal**: [https://climateknowledgeportal.worldbank.org/](https://climateknowledgeportal.worldbank.org/)
    - API: `https://climateknowledgeportal.worldbank.org/api/data`
    - Returned HTTP 403 to programmatic requests as of 2026-10; previously used, now replaced by the Projects API.

## 4. Structured Datasets
- **OWID CO2 Data**: [https://github.com/owid/co2-data](https://github.com/owid/co2-data)
- **Global Carbon Project**: [https://www.globalcarbonproject.org/carbonbudget/](https://www.globalcarbonproject.org/carbonbudget/)
- **Climate Watch**: [https://www.climatewatchdata.org/data-explorer](https://www.climatewatchdata.org/data-explorer)
    - API: `https://www.climatewatchdata.org/api/v1`
