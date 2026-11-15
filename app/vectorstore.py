from functools import lru_cache

from langchain_postgres import PGVector
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from app.config import get_settings
from app.embeddings import get_embeddings


@lru_cache
def get_engine() -> Engine:
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def get_vectorstore() -> PGVector:
    s = get_settings()
    return PGVector(
        embeddings=get_embeddings(),
        collection_name=s.collection_name,
        connection=get_engine(),
        use_jsonb=True,
    )


def get_retriever():
    s = get_settings()
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={"k": s.retriever_k, "fetch_k": s.retriever_fetch_k},
    )


# PGVector has no delete-by-metadata, so these helpers query its tables directly.
_COLLECTION_FILTER = """
    collection_id = (SELECT uuid FROM langchain_pg_collection WHERE name = :collection)
"""


def list_sources() -> list[dict]:
    sql = text(
        f"""
        SELECT cmetadata->>'source' AS source, COUNT(*) AS chunks
        FROM langchain_pg_embedding
        WHERE {_COLLECTION_FILTER}
        GROUP BY 1 ORDER BY 1
        """
    )
    with get_engine().connect() as conn:
        rows = conn.execute(sql, {"collection": get_settings().collection_name})
        return [{"source": r.source, "chunks": r.chunks} for r in rows]


def delete_source(source: str) -> int:
    sql = text(
        f"""
        DELETE FROM langchain_pg_embedding
        WHERE {_COLLECTION_FILTER} AND cmetadata->>'source' = :source
        """
    )
    get_vectorstore()  # make sure tables exist
    with get_engine().begin() as conn:
        result = conn.execute(
            sql, {"collection": get_settings().collection_name, "source": source}
        )
        return result.rowcount


def ping() -> bool:
    with get_engine().connect() as conn:
        conn.execute(text("SELECT 1"))
    return True
