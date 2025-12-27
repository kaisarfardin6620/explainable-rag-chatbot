from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import shutil
import os
from app.core.config import settings
from app.services.document_processor import process_uploaded_file
from app.services.rag_pipeline import run_rag_pipeline
from app.database.repository import (
    get_all_documents, 
    get_session_history, 
    save_chat_message
)
from app.models.schemas import (
    ChatRequest, ChatResponse, 
    UploadResponse, HealthResponse,
    DocumentListResponse, ChatHistoryResponse
)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health():
    return {"status": "healthy"}

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(('.pdf', '.txt')):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files supported")
    
    file_path = f"{settings.upload_folder}/{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        result = process_uploaded_file(file_path, file.filename)
        return result
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chat/ask", response_model=ChatResponse)
async def chat_ask(request: ChatRequest):
    try:
        session_id = request.session_id or 1 
        response = run_rag_pipeline(
            question=request.question, 
            session_id=session_id,
            mode=request.mode
        )
        
        save_chat_message(session_id, "user", request.question)
        save_chat_message(session_id, "assistant", response["answer"])
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    try:
        docs = get_all_documents()
        return {"count": len(docs), "documents": docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history/{session_id}", response_model=ChatHistoryResponse)
async def get_history(session_id: int):
    try:
        messages = get_session_history(session_id)
        return {"session_id": session_id, "messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))