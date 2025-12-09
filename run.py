import climate_core
from climate_core import Document
from climate_rag.pipeline import run_rag_pipeline

def main():
    print("Initializing Core...")
    # 768 dim embeddings
    index = climate_core.VectorIndex(768)
    
    print("Ingesting Data...")
    docs = [
        Document("doc1", "The Paris Agreement sets a target of keeping warming well below 2 degrees Celsius.", {"source": "UNFCCC"}),
        Document("doc2", "Implementation of NDCs requires significant climate finance flows to developing nations.", {"source": "IPCC"}),
        Document("doc3", "Solar tracking systems increase efficiency by 25% but require higher maintenance.", {"source": "TechReport"}),
    ]
    
    # Mock embeddings (random for demo)
    # Real app would use an encoder model here.
    import random
    
    for doc in docs:
        # Generate pseudo-random embedding normalized
        emb = [0.1] * 768
        index.add(doc, emb)
        
    print("Running Pipeline...")
    query = "What are the targets of the Paris Agreement?"
    result = run_rag_pipeline(query, index)
    
    print("\n=== Final Result ===")
    print(f"Query: {result.query}")
    print(f"Intent: {result.intent.value}")
    print(f"Answer: {result.answer}")
    print("Logs:")
    for log in result.logs:
        print(f" - {log}")

if __name__ == "__main__":
    main()
