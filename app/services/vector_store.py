import uuid
import time
import logging
from typing import List, Dict, Any, Optional
from pinecone import Pinecone, ServerlessSpec
from app.core.config import settings
from app.services.embedding_service import get_embeddings

logger = logging.getLogger("rag_chatbot")

pc = Pinecone(api_key=settings.pinecone_api_key)

INDEX_NAME = settings.pinecone_index_name
EMBEDDING_DIMENSION = 3072 if "large" in settings.openai_embedding_model.lower() else 1536

if INDEX_NAME not in pc.list_indexes().names():
    logger.info(f"Creating Pinecone index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region=settings.pinecone_environment
        )
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(1)

index = pc.Index(INDEX_NAME)

def upsert_chunks(chunks: List[Dict[str, Any]]) -> None:
    """
    Embeds and upserts chunks. Uses the 'chunk_id' from metadata 
    to ensure consistency with SQLite and Neo4j.
    """
    if not chunks:
        return

    texts = [chunk["text"] for chunk in chunks]
    embeddings = get_embeddings(texts)

    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        chunk_id = chunk["metadata"].get("chunk_id") or str(uuid.uuid4())
        
        metadata = {
            "document": chunk["metadata"].get("document", "unknown.pdf"),
            "page": int(chunk["metadata"].get("page", 0)),
            "chunk_id": chunk_id,
            "text": chunk["text"][:1000] 
        }

        vectors_to_upsert.append({
            "id": chunk_id,
            "values": list(embeddings[i]),
            "metadata": metadata
        })

    batch_size = 100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i + batch_size]
        index.upsert(vectors=batch)

def query(
    question: str,
    top_k: int = 5,
    min_similarity: float = 0.65
) -> List[Dict[str, Any]]:
    """
    Semantic search returning consistent chunk_ids.
    """
    query_embedding = list(get_embeddings([question])[0])

    results = index.query(
        vector=query_embedding,
        top_k=top_k * 2,
        include_metadata=True
    )

    matches = []
    for match in results.matches:
        if match.score < min_similarity:
            continue

        matches.append({
            "chunk_id": match.id,
            "document": match.metadata.get("document", "unknown"),
            "page": int(match.metadata.get("page", 0)),
            "similarity": round(match.score, 4)
        })

        if len(matches) >= top_k:
            break

    return matches

def delete_all_vectors() -> None:
    index.delete(delete_all=True)

def get_index_stats() -> Dict[str, Any]:
    return index.describe_index_stats()