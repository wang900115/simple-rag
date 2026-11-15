from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import get_settings


@lru_cache
def get_embeddings() -> Embeddings:
    s = get_settings()
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        model_kwargs={"device": s.embedding_device},
        encode_kwargs={"normalize_embeddings": True},
    )
