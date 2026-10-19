import re

from climate_rag.llm import LLM

EXPANSION_SYSTEM = (
    "You rewrite questions into search queries for a knowledge base of climate research "
    "abstracts and World Bank project descriptions. Write {n} alternative search queries for "
    "the user's question. Each must use different vocabulary from the question: synonyms, "
    "technical terms, and the names of mechanisms, institutions, programmes or metrics that a "
    "relevant document would use. Spell out acronyms. Output one query per line, with no "
    "numbering, quotes or commentary."
)

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


def expand_query(query: str, llm: LLM, n: int = 3) -> list[str]:
    """Up to n alternative phrasings of query, for searching alongside the original."""
    raw = llm(EXPANSION_SYSTEM.format(n=n), query)
    expansions: list[str] = []
    for line in raw.splitlines():
        text = _LIST_MARKER.sub("", line).strip().strip('"').strip()
        if text and text.lower() != query.lower() and text not in expansions:
            expansions.append(text)
    return expansions[:n]
