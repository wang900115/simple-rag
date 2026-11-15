"""Building blocks for the RAG graph: prompts, LLM, retrieval and grading."""

from functools import lru_cache

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from app.config import get_settings

# Fixed strings returned without calling the model, so they can't follow the user's language.
NO_ANSWER = "The documents don't contain information about this."
REFUSAL_MESSAGE = "Sorry, I can't answer this question. Please try rephrasing it."

SYSTEM_PROMPT = """You are an internal knowledge-base assistant for the company.

Rules:
- Answer in the same language the user asked the question in.
- Answer only from the documents in the <context> of the latest message. Do not use outside knowledge or speculate.
- Earlier conversation turns are only for understanding what the user is asking; they are not a source of facts.
- If the documents don't contain enough information to answer, say so plainly instead of making something up.
- Cite each key point with [Source: filename], taking the filename from the <doc source="..."> attribute.
- Keep answers concise and well organized."""

# Turns a follow-up like "What about sick leave?" into a standalone search query.
CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Using the conversation history, rewrite the user's latest question as a "
            "standalone question that can be understood without the history, for searching "
            "internal documents. Keep the language of the latest question. Output only the "
            "rewritten question; do not answer it. If it is already standalone, output it unchanged.",
        ),
        MessagesPlaceholder("history"),
        ("human", "Latest question: {question}"),
    ]
)

# Used when retrieval came back with irrelevant documents.
REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "The search query below found nothing relevant in the internal document library. "
            "Rewrite it with different wording: synonyms, or broader or more specific keywords. "
            "Keep the same language as the query. Output only the rewritten query.",
        ),
        ("human", "Original query: {query}"),
    ]
)

GRADE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Decide whether the retrieved documents contain the information needed to answer "
            "the question. They count as relevant if any one document directly addresses it.",
        ),
        ("human", "Question: {query}\n\n<context>\n{context}\n</context>"),
    ]
)


class Grade(BaseModel):
    relevant: bool = Field(description="Whether the documents can answer the question")


@lru_cache
def get_llm() -> BaseChatModel:
    """Local Ollama chat model used for answering, rewriting, and grading."""
    s = get_settings()
    return ChatOllama(
        model=s.llm_model,
        base_url=s.ollama_base_url,
        temperature=0,
        num_predict=s.llm_max_tokens,
        reasoning=False,
    )


def format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        headers = d.metadata.get("headers", "")
        parts.append(
            f'<doc source="{src}" section="{headers}">\n{d.page_content}\n</doc>'
        )
    return "\n\n".join(parts)


def to_sources(docs: list[Document]) -> list[dict]:
    return [
        {
            "source": d.metadata.get("source"),
            "headers": d.metadata.get("headers", ""),
            "snippet": d.page_content[:200],
        }
        for d in docs
    ]


def retrieve(query: str) -> list[Document]:
    from app.vectorstore import get_retriever

    return get_retriever().invoke(query)


def grade_documents(query: str, docs: list[Document]) -> bool:
    grader = get_llm().with_structured_output(Grade, method="json_schema")
    result = grader.invoke(GRADE_PROMPT.invoke({"query": query, "context": format_docs(docs)}))
    return result.relevant
