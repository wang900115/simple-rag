"""LangGraph RAG flow with multi-turn memory and self-correcting retrieval.

    START → contextualize → retrieve → grade ─┬─ relevant / out of retries → generate → END
                               ↑              └─ not relevant → rewrite ─┐
                               └─────────────────────────────────────────┘

- contextualize: turns a follow-up question into a standalone search query using history.
- grade / rewrite: if the retrieved chunks don't answer the query, rewrite it and retry
  (at most `max_rewrites` times).
- Conversation history lives in `messages` and is persisted per `thread_id` by the
  checkpointer, so each request only sends the new question.
"""

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from app import rag
from app.config import get_settings


class RAGState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]  # full conversation (persisted)
    query: str  # current search query (may be rewritten)
    docs: list[dict]  # retrieved chunks, stored as plain dicts so they serialize cleanly
    relevant: bool
    retries: int


def _to_documents(docs: list[dict]) -> list[Document]:
    return [Document(page_content=d["page_content"], metadata=d["metadata"]) for d in docs]


def _history(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Prior turns (excluding the latest question), capped to bound token usage."""
    limit = get_settings().history_turns * 2
    return messages[:-1][-limit:]


def contextualize(state: RAGState) -> dict:
    question = state["messages"][-1].text
    history = _history(state["messages"])
    query = question
    if history:
        msgs = rag.CONTEXTUALIZE_PROMPT.invoke({"history": history, "question": question})
        query = rag.get_llm().invoke(msgs).text.strip() or question
    # Reset per-turn fields; they would otherwise carry over from the previous turn.
    return {"query": query, "retries": 0, "relevant": False, "docs": []}


def retrieve(state: RAGState) -> dict:
    docs = rag.retrieve(state["query"])
    return {"docs": [{"page_content": d.page_content, "metadata": d.metadata} for d in docs]}


def grade(state: RAGState) -> dict:
    if not state["docs"]:
        return {"relevant": False}
    return {"relevant": rag.grade_documents(state["query"], _to_documents(state["docs"]))}


def route_after_grade(state: RAGState) -> str:
    if state["relevant"] or state["retries"] >= get_settings().max_rewrites:
        return "generate"
    return "rewrite"


def rewrite(state: RAGState) -> dict:
    msgs = rag.REWRITE_PROMPT.invoke({"query": state["query"]})
    new_query = rag.get_llm().invoke(msgs).text.strip() or state["query"]
    return {"query": new_query, "retries": state["retries"] + 1}


def generate(state: RAGState) -> dict:
    docs = _to_documents(state["docs"])
    if not docs:
        return {"messages": [AIMessage(content=rag.NO_ANSWER)]}

    question = state["messages"][-1].text
    msgs = [
        SystemMessage(content=rag.SYSTEM_PROMPT),
        *_history(state["messages"]),
        HumanMessage(content=f"<context>\n{rag.format_docs(docs)}\n</context>\n\nQuestion: {question}"),
    ]
    result = rag.get_llm().invoke(msgs)
    if result.additional_kwargs.get("refusal"):
        return {"messages": [AIMessage(content=rag.REFUSAL_MESSAGE)]}
    # Store plain text only; the retrieved context is not kept in history.
    return {"messages": [AIMessage(content=result.text)]}


def build_graph(checkpointer: BaseCheckpointSaver):
    g = StateGraph(RAGState)
    g.add_node("contextualize", contextualize)
    g.add_node("retrieve", retrieve)
    g.add_node("grade", grade)
    g.add_node("rewrite", rewrite)
    g.add_node("generate", generate)

    g.add_edge(START, "contextualize")
    g.add_edge("contextualize", "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, ["generate", "rewrite"])
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)
    return g.compile(checkpointer=checkpointer)


def result_payload(state: dict) -> dict:
    """Final answer + sources from a graph state."""
    return {
        "answer": state["messages"][-1].text,
        "query": state.get("query", ""),
        "sources": rag.to_sources(_to_documents(state.get("docs", []))),
    }
