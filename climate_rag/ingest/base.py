from typing import List, Protocol
from climate_core import Document

class Ingestor(Protocol):
    async def fetch(self, query: str = "", limit: int = 5) -> List[Document]:
        """
        Fetch documents from the external source and normalize them into
        the core Document definition.
        """
        ...
