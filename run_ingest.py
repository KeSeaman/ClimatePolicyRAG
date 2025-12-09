import asyncio
import climate_core
from climate_rag.ingest.arxiv import ArxivIngestor
from climate_rag.ingest.worldbank import WorldBankIngestor
from climate_rag.pipeline import run_rag_pipeline

async def main():
    # Initialize Core
    print("Initializing Core...")
    index = climate_core.VectorIndex(768)
    
    # 1. Ingest Arxiv
    print("\n=== Ingesting Scientific Data (ArXiv) ===")
    arxiv = ArxivIngestor()
    papers = await arxiv.fetch(query="climate policy", limit=3)
    for doc in papers:
        print(f" -> [Arxiv] {doc.metadata['title'][:50]}...")
        # Mock embedding
        index.add(doc, [0.1] * 768)
        
    # 2. Ingest World Bank
    print("\n=== Ingesting Implementation Data (World Bank) ===")
    wb = WorldBankIngestor()
    wb_data = await wb.fetch(limit=5)
    for doc in wb_data:
        print(f" -> [WB] {doc.content}")
        # Mock embedding
        index.add(doc, [0.2] * 768) # different vector
        
    # 3. Query
    print("\n=== Querying System ===")
    # Query relevant to WB Data
    result = run_rag_pipeline("What are the CO2 emissions stats?", index)
    print(f"Index size: {len(wb_data) + len(papers)} documents")
    print(f"Retrieved Contexts: {len(result.contexts)}")
    for c in result.contexts:
        print(f" - [{c.source}] {c.content[:60]}...")

if __name__ == "__main__":
    asyncio.run(main())
