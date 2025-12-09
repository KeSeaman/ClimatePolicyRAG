from climate_rag.models import AgentState, Intent

def analyze_intent(state: AgentState) -> AgentState:
    """
    Pure function to determine query intent.
    """
    q = state.query.lower()
    
    intent = Intent.GENERAL
    if any(w in q for w in ["target", "policy", "agreement", "paris", "cop"]):
        intent = Intent.POLICY
    elif any(w in q for w in ["temperature", "degree", "warming", "ppm", "science"]):
        intent = Intent.SCIENTIFIC
    elif any(w in q for w in ["implement", "finance", "action", "project", "deploy"]):
        intent = Intent.IMPLEMENTATION
        
    return state.update_intent(intent).add_log(f"Intent classified as: {intent.value}")
