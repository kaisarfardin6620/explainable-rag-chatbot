from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal

class HealthResponse(BaseModel):
    status: str

class UploadResponse(BaseModel):
    status: str
    document: str
    chunks_processed: int
    message: str

class DocumentInfo(BaseModel):
    id: int
    filename: str
    status: str
    upload_time: str

class DocumentListResponse(BaseModel):
    count: int
    documents: List[DocumentInfo]

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str

class ChatHistoryResponse(BaseModel):
    session_id: int
    messages: List[ChatMessage]

class Entity(BaseModel):
    name: str
    type: str = Field(..., description="Type of entity, e.g., PERSON, DRUG, PROCEDURE, CONCEPT")
    description: Optional[str] = Field(None, description="Brief context about the entity")

class Relation(BaseModel):
    source: str
    target: str
    relation: str = Field(..., description="The relationship predicate in UPPER_CASE, e.g., CAUSES, TREATS")
    description: Optional[str] = Field(None, description="Context explaining the relationship")

class KnowledgeGraphSchema(BaseModel):
    """Strict schema for LLM extraction to ensure graph quality."""
    entities: List[Entity]
    relationships: List[Relation]

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[int] = None
    mode: Optional[str] = "hybrid"

class Citation(BaseModel):
    document: str
    page: int
    text_snippet: str
    source_type: Literal["text", "graph"]

class Explanation(BaseModel):
    summary: str
    reasoning_chain: List[str]
    confidence_score: float
    metrics: Dict[str, float] 

class ChatResponse(BaseModel):
    answer: str
    citations: List[Citation]
    explanation: Optional[Explanation] = None
    refusal: bool
    refusal_reason: Optional[str] = None