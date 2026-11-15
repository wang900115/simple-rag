from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    anthropic_api_key: str = ""
    database_url: str = "postgresql+psycopg://rag:rag@localhost:5432/rag"
    collection_name: str = "simple_rag"

    embedding_model: str = "BAAI/bge-m3"
    embedding_device: str = "cpu"

    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
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
