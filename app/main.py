from fastapi import FastAPI
from app.api.routes import router

app = FastAPI(
    title="Explainable RAG Chatbot",
    description="Compliance-grade single-user RAG with citations & confidence",
    version="1.0.0"
)

app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "healthy"}