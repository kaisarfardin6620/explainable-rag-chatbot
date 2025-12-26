from openai import OpenAI
from app.core.config import settings
import numpy as np
from typing import List

client = OpenAI(api_key=settings.openai_api_key)

def get_embeddings(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.array([])

    cleaned_texts = [text.replace("\n", " ") for text in texts]

    response = client.embeddings.create(
        model=settings.openai_embedding_model,
        input=cleaned_texts
    )

    embeddings = [item.embedding for item in response.data]
    return np.array(embeddings)