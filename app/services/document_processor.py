import uuid
from typing import List, Dict, Any
from pathlib import Path
import pdfplumber
from app.core.config import settings
from app.services.embedding_service import get_embeddings
from app.services.vector_store import upsert_chunks
from app.services.kg_builder import build_kg_from_chunks
from app.database.repository import save_chunks, get_or_create_document_id
from app.utils.helpers import semantic_chunk_text

def process_uploaded_file(file_path: Path, filename: str) -> Dict[str, Any]:
    """
    Full document processing pipeline.
    Called from /upload endpoint.
    """
    raw_pages = extract_text_with_pages(file_path)

    chunks = []
    for page_num, text in raw_pages:
        page_chunks = semantic_chunk_text(text, max_tokens=800) 
        for chunk_text in page_chunks:
            chunk_id = str(uuid.uuid4())
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "document": filename,
                    "page": page_num
                }
            })

    if not chunks:
        return {"status": "error", "message": "No text extracted from document"}

    document_id = get_or_create_document_id(filename)

    save_chunks(chunks, document_id)

    pinecone_chunks = [
        {
            "text": chunk["text"],
            "metadata": {
                "document": chunk["metadata"]["document"],
                "page": chunk["metadata"]["page"],
                "chunk_id": chunk["chunk_id"]
            }
        }
        for chunk in chunks
    ]

    upsert_chunks(pinecone_chunks)

    build_kg_from_chunks(chunks, document_name=filename, document_id=document_id)

    return {
        "status": "success",
        "document": filename,
        "chunks_processed": len(chunks),
        "message": "Document processed, indexed in Pinecone, and added to Knowledge Graph"
    }


def extract_text_with_pages(file_path: Path) -> List[tuple[int, str]]:
    """
    Extract text from PDF with page numbers using pdfplumber.
    Supports .pdf and .txt
    """
    text_pages = []

    if file_path.suffix.lower() == ".txt":
        with open(file_path, "r", encoding="utf-8") as f:
            full_text = f.read()
        text_pages.append((1, full_text))
        return text_pages

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                if text and text.strip():
                    text_pages.append((i, text.strip()))
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {e}")

    return text_pages


def semantic_chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    """
    Simple paragraph-based chunking (good baseline).
    Replace with LangChain RecursiveCharacterTextSplitter for better semantics.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 2 < max_tokens * 4:
            current_chunk += ("\n\n" + para if current_chunk else para)
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para

    if current_chunk:
        chunks.append(current_chunk)

    return chunks if chunks else [text[:max_tokens*4]]