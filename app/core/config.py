from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    pinecone_api_key: str
    pinecone_environment: str
    pinecone_index_name: str

    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    neo4j_database: str = "neo4j"

    upload_folder: str = "./data/uploads"
    sqlite_db_path: str = "./data/sqlite.db"

    top_k: int = 5 
    min_similarity_threshold: float = 0.5 

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()