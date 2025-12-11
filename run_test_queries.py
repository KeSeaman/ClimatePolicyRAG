import asyncio
import climate_core
from climate_rag.ingest.arxiv import ArxivIngestor
from climate_rag.ingest.worldbank import WorldBankIngestor
from climate_rag.pipeline import run_rag_pipeline
import os

QUERIES = [
    "What are the key goals of the Paris Agreement regarding temperature limits?",
    "How does carbon pricing impact developing economies?",
    "What are the latest trends in renewable energy adoption in Europe?",
    "Explain the role of forestry in carbon sequestration strategies.",
    "What is the impact of methane emissions on global warming compared to CO2?",
    "How are NDCs (Nationally Determined Contributions) being tracked?",
    "What are the projected sea-level rises by 2100 under different scenarios?",
    "Describe the financial mechanisms available for climate adaptation.",
    "What are the major barriers to implementing carbon capture and storage technologies?",
    "How does agriculture contribute to greenhouse gas emissions?"
]

async def main():
    print("Initializing Core...")
    index = climate_core.VectorIndex(768)
    
    # 1. Ingest Data to have something to search
    print("\n=== Ingesting Data for Context ===")
    
    # ArXiv
    print(" -> Fetching ArXiv papers...")
    arxiv = ArxivIngestor()
    papers = await arxiv.fetch(query="climate change policy", limit=5)
    for doc in papers:
        # Mock embedding
        index.add(doc, [0.1] * 768)
        
    # World Bank
    print(" -> Fetching World Bank data...")
    wb = WorldBankIngestor()
    wb_data = await wb.fetch(limit=5)
    for doc in wb_data:
        # Mock embedding (distinct from arxiv for variety if real embeddings used)
        index.add(doc, [0.15] * 768)

    print(f"\nTotal documents indexed: {len(papers) + len(wb_data)}")
    
    # 2. Run Queries
    output_lines = []
    output_lines.append("# Climate RAG - Test Query Results")
    output_lines.append(f"**Date**: {os.popen('date').read().strip()}")
    output_lines.append(f"**Indexed Documents**: {len(papers) + len(wb_data)}")
    output_lines.append("\n---\n")

    print("\n=== Running 10 Sample Queries ===")
    
    for i, q in enumerate(QUERIES, 1):
        print(f"Running Query {i}/{len(QUERIES)}: {q[:30]}...")
        result = run_rag_pipeline(q, index)
        
        output_lines.append(f"## Query {i}")
        output_lines.append(f"**Q**: {result.query}")
        output_lines.append(f"**Intent**: `{result.intent.value}`")
        output_lines.append(f"\n**Synthesized Answer**:\n{result.answer}\n")
        output_lines.append("**Retrieved Contexts**:")
        if result.contexts:
            for ctx in result.contexts:
                output_lines.append(f"- *[{ctx.source}]*: {ctx.content[:150]}...")
        else:
            output_lines.append("- *No contexts retrieved*")
        
        output_lines.append("\n---\n")

    # 3. Save to File
    output_path = "notes/Test_Queries.md"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("\n".join(output_lines))
        
    print(f"\nSuccessfully wrote results to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
