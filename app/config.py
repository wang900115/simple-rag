from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    collection_name: str = "simple_rag"

    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "bge-m3"

    llm_model: str = "qwen3:4b"
    llm_max_tokens: int = 16000

    chunk_size: int = 800
    chunk_overlap: int = 100
    retriever_k: int = 5
    retriever_fetch_k: int = 20

    # LangGraph
    checkpointer: str = "memory"  # "memory" (lost on restart) or "postgres"
    max_rewrites: int = 2  # how many times a query may be rewritten when retrieval misses
    history_turns: int = 6  # prior Q&A pairs sent to the model

    @property
    def psycopg_url(self) -> str:
        """Plain libpq URL for psycopg (the checkpointer doesn't use SQLAlchemy)."""
        return self.database_url.replace("postgresql+psycopg://", "postgresql://", 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
