import httpx
from typing import List
from climate_rag.ingest.base import Ingestor
from climate_core import Document

class WorldBankIngestor:
    # World Bank Climate Knowledge Portal API
    # Fetching historical annual mean temperature (tas) for USA
    BASE_URL = "https://climateknowledgeportal.worldbank.org/api/data/get-CCKP-historical-data/USA/year/tas/2010/2022"

    async def fetch(self, query: str = "", limit: int = 10) -> List[Document]:
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(self.BASE_URL)
                response.raise_for_status()
            
            data = response.json()
        except Exception as e:
            print(f"[WB Error] Failed to fetch World Bank data: {e}")
            return []
        docs = []
        
        # CCKP API returns a list directly
        for record in data[:limit]:
            year = record.get('year')
            tas = record.get('tas')
            iso = record.get('iso3')
            
            content = f"In {year}, {iso} had an average temperature of {tas} degrees Celsius."
            doc_id = f"wb_cckp_{iso}_{year}"
            
            docs.append(Document(
                id=doc_id,
                content=content,
                metadata={
                    "source": "world_bank_cckp",
                    "country": iso,
                    "year": str(year),
                    "indicator": "Average Temperature (tas)"
                }
            ))
            
        return docs
