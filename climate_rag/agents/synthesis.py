from climate_rag.models import AgentState

def synthesize_answer(state: AgentState) -> AgentState:
    """
    Pure function to generate answer from context.
    """
    if not state.contexts:
        return state.set_answer("I could not find relevant information in the climate knowledge base.") \
                    .add_log("Synthesis: No context found.")
    
    # Construct prompt (Conceptually)
    context_str = "\n\n".join([f"[{c.doc_id}]: {c.content}" for c in state.contexts])
    prompt = f"Query: {state.query}\nContext:\n{context_str}\nAnswer:"
    
    # Simulate LLM Generation
    # In prod: response = openai.chat(models="gpt-4", messages=[...])
    
    fake_response = f"Based on {len(state.contexts)} sources (e.g. {state.contexts[0].doc_id}), the answer regarding '{state.intent.value}' is [Simulated Answer generated from context]."
    
    return state.set_answer(fake_response).add_log("Synthesis: Generated answer.")
