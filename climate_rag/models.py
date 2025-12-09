from pydantic import BaseModel, Field, ConfigDict
from  enum import Enum
from typing import List, Optional, Dict
from datetime import datetime

class Intent(str, Enum):
    SCIENTIFIC = "scientific"
    POLICY = "policy"
    IMPLEMENTATION = "implementation"
    GENERAL = "general"

class RetrievedContext(BaseModel):
    model_config = ConfigDict(frozen=True)
    
    doc_id: str
    content: str
    score: float
    source: str

class AgentState(BaseModel):
    """
    Immutable state passed through the functional pipeline.
    Each agent returns a NEW instance of AgentState.
    """
    model_config = ConfigDict(frozen=True)
    
    query: str
    intent: Intent = Intent.GENERAL
    # Contexts retrieved from vector store
    contexts: List[RetrievedContext] = Field(default_factory=list)
    # Synthesized answer
    answer: Optional[str] = None
    # Agent chain execution log
    logs: List[str] = Field(default_factory=list)
    # Global metadata (e.g. user id, timestamp)
    metadata: Dict[str, str] = Field(default_factory=dict)

    def add_log(self, message: str) -> "AgentState":
        new_logs = self.logs + [f"{datetime.utcnow().isoformat()}: {message}"]
        return self.model_copy(update={"logs": new_logs})

    def update_intent(self, new_intent: Intent) -> "AgentState":
        return self.model_copy(update={"intent": new_intent})
        
    def add_contexts(self, new_contexts: List[RetrievedContext]) -> "AgentState":
        return self.model_copy(update={"contexts": self.contexts + new_contexts})

    def set_answer(self, answer: str) -> "AgentState":
        return self.model_copy(update={"answer": answer})
