import httpx
import xml.etree.ElementTree as ET
from typing import List
from climate_rag.ingest.base import Ingestor
from climate_core import Document

class ArxivIngestor:
    BASE_URL = "https://export.arxiv.org/api/query"

    async def fetch(self, query: str = "climate change", limit: int = 5) -> List[Document]:
        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": limit,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        async with httpx.AsyncClient(follow_redirects=True) as client:
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            
        return self._parse_atom(response.text)

    def _parse_atom(self, xml_content: str) -> List[Document]:
        root = ET.fromstring(xml_content)
        # Atom namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        docs = []
        for entry in root.findall('atom:entry', ns):
            doc_id = entry.find('atom:id', ns).text
            title = entry.find('atom:title', ns).text.strip().replace('\n', ' ')
            summary = entry.find('atom:summary', ns).text.strip().replace('\n', ' ')
            author = entry.find('atom:author/atom:name', ns).text
            
            # Combine title and summary for the content
            content = f"Title: {title}\nAbstract: {summary}"
            
            docs.append(Document(
                id=doc_id,
                content=content,
                metadata={
                    "source": "arxiv",
                    "author": author,
                    "title": title
                }
            ))
            
        return docs
