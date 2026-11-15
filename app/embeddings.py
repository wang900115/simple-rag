from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    return OllamaEmbeddings(
        model=s.embedding_model,
        base_url=s.ollama_base_url,
    )
