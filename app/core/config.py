from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str
    openai_embedding_model: str

    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str

    upload_folder: str
    sqlite_db_path: str

    top_k: int
    min_similarity_threshold: float

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "forbid"

settings = Settings()