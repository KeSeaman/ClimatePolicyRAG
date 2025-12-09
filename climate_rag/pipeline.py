from toolz import pipe
from functools import partial
from climate_rag.models import AgentState
from climate_rag.agents.analysis import analyze_intent
from climate_rag.agents.retrieval import retrieve_documents
from climate_rag.agents.synthesis import synthesize_answer
import climate_core

def run_rag_pipeline(query: str, index: climate_core.VectorIndex) -> AgentState:
    """
    Orchestrates the RAG flow using functional composition.
    """
    initial_state = AgentState(query=query)
    
    # We bind the 'index' to the retrieval function using partial
    # to maintain the single-argument signature expected by the pipe,
    # or we can wrap it.
    retrieve_with_index = partial(retrieve_documents, index=index, k=3)
    
    final_state = pipe(
        initial_state,
        analyze_intent,
        retrieve_with_index,
        synthesize_answer
    )
    
    return final_state
