import re

from climate_rag.models import AgentState, Intent

# Whole-word keywords per intent. Matching tokens (not substrings) avoids false
# hits such as "cop" in "scope". Dict order breaks ties: policy > scientific > implementation.
_KEYWORDS: dict[Intent, frozenset[str]] = {
    Intent.POLICY: frozenset({
        "policy", "policies", "agreement", "paris", "cop", "ndc", "ndcs", "target", "targets",
        "pledge", "pledges", "regulation", "law", "laws", "treaty", "unfccc", "pricing", "tax",
        "contributions", "governance",
    }),
    Intent.SCIENTIFIC: frozenset({
        "temperature", "warming", "degree", "degrees", "ppm", "science", "scientific",
        "emissions", "methane", "co2", "sea", "sequestration", "projected", "projections",
        "scenarios", "greenhouse",
    }),
    Intent.IMPLEMENTATION: frozenset({
        "implement", "implementing", "implementation", "finance", "financial", "funding",
        "investment", "project", "projects", "deploy", "deployment", "adoption", "barriers",
        "mechanisms", "tracked", "tracking", "action",
    }),
}
_TOKEN = re.compile(r"[a-z0-9-]+")


def analyze_intent(state: AgentState) -> AgentState:
    """
    Pure function to determine query intent: the intent with the most keyword hits wins.
    """
    words = set(_TOKEN.findall(state.query.lower()))
    hits = {intent: len(words & keywords) for intent, keywords in _KEYWORDS.items()}
    best = max(hits, key=lambda intent: hits[intent])
    intent = best if hits[best] else Intent.GENERAL

    return state.update_intent(intent).add_log(f"Intent classified as: {intent.value}")
