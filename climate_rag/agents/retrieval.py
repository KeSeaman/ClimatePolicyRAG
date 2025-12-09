from climate_rag.models import AgentState, RetrievedContext
import climate_core
from typing import Optional

def retrieve_documents(state: AgentState, index: climate_core.VectorIndex, k: int = 3) -> AgentState:
    """
    Pure function to retrieve documents using the Rust core.
    Side-effect: Reads from Index (Query).
    """
    try:
        # Vector search (Rust)
        # Assuming we have an embedding function helper. 
        # For now, we mock embedding generation or assume input is already processed?
        # Real system needs an Embedder. Let's assume a function `embed_query`.
        
        # Simplified: We generate a dummy embedding for demo if no real embedder connected
        # In prod: embedding = embedder.encode(state.query)
        embedding = [0.1] * 768 # Placeholder 768-dim vector
        
        results = index.search(embedding, k)
        
        contexts = [
            RetrievedContext(
                doc_id=r.doc_id,
                content=r.content_preview,
                score=r.score,
                source=r.doc_id # Simplification
            ) for r in results
        ]
        
        return state.add_contexts(contexts).add_log(f"Retrieved {len(contexts)} docs")
        
    except Exception as e:
        return state.add_log(f"Retrieval Error: {str(e)}")

# Placeholder for real embedding logic
def embed_query(query: str) -> list[float]:
    # TODO: Connect to OpenAI/SentenceTransformer
    return [0.0] * 768
