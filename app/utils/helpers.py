import hashlib
from typing import List

def clean_text(text: str) -> str:
    """Removes excessive whitespace."""
    return " ".join(text.split())

def generate_hash(text: str) -> str:
    """Generate MD5 hash for deduping."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def semantic_chunk_text(text: str, max_tokens: int = 800) -> List[str]:
    """
    Splits text into chunks roughly based on paragraphs.
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