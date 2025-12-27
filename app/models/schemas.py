from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class HealthResponse(BaseModel):
    status: str

class UploadResponse(BaseModel):
    status: str
    document: str
    chunks_processed: int
    message: str

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[int] = None
    mode: Optional[str] = "hybrid" 

class Citation(BaseModel):
    document: str
    page: int
    source: str

class Explanation(BaseModel):
    summary: str
    supported_claims: List[Dict[str, str]]
    unsupported_claims: List[str]
    confidence_signals: Dict[str, float]

class ChatResponse(BaseModel):
    answer: str
    confidence: float
    confidence_level: str
    citations: List[Citation]
    explanation: Optional[Explanation] = None
    refusal: bool

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