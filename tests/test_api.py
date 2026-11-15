import json

import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver

from app import graph as graph_mod
from app import main, rag

DOCS = [
    Document(
        page_content="Annual leave must be requested 3 working days in advance.",
        metadata={"source": "leave-policy.md", "headers": "Leave Policy > Annual Leave"},
    )
]


class Fakes:
    """Records retrieval queries and feeds scripted LLM replies / grades."""

    def __init__(self, monkeypatch):
        self.queries: list[str] = []
        self.grades: list[bool] = []
        self.docs = DOCS
        self.llm = GenericFakeChatModel(messages=iter([]))
        monkeypatch.setattr(rag, "retrieve", self._retrieve)
        monkeypatch.setattr(rag, "grade_documents", lambda q, d: self.grades.pop(0))
        monkeypatch.setattr(rag, "get_llm", lambda effort=None: self.llm)

    def _retrieve(self, query):
        self.queries.append(query)
        return self.docs

    def script(self, replies: list[str], grades: list[bool]):
        self.llm.messages = iter(AIMessage(content=r) for r in replies)
        self.grades = list(grades)


@pytest.fixture
def fakes(monkeypatch):
    return Fakes(monkeypatch)


@pytest.fixture
def client():
    # No `with` block: lifespan (embedding model, Postgres) is skipped.
    main.app.state.graph = graph_mod.build_graph(InMemorySaver())
    return TestClient(main.app)


def ask(client, question, session_id=None):
    r = client.post("/ask", json={"question": question, "session_id": session_id})
    assert r.status_code == 200, r.text
    return r.json()


def test_first_turn_skips_contextualize(client, fakes):
    fakes.script(replies=["3 working days in advance [Source: leave-policy.md]"], grades=[True])
    body = ask(client, "How do I request annual leave?")
    assert fakes.queries == ["How do I request annual leave?"]
    assert "leave-policy.md" in body["answer"]
    assert body["sources"][0]["headers"] == "Leave Policy > Annual Leave"
    assert body["session_id"]


def test_follow_up_is_contextualized_with_history(client, fakes):
    fakes.script(replies=["answer 1"], grades=[True])
    sid = ask(client, "How do I request annual leave?")["session_id"]

    fakes.script(replies=["What is the sick leave policy?", "answer 2"], grades=[True])
    body = ask(client, "What about sick leave?", sid)
    assert fakes.queries[-1] == "What is the sick leave policy?"
    assert body["query"] == "What is the sick leave policy?"
    assert body["answer"] == "answer 2"

    history = client.get(f"/sessions/{sid}").json()
    assert [m["role"] for m in history] == ["human", "ai", "human", "ai"]
    assert history[2]["content"] == "What about sick leave?"


def test_sessions_are_isolated(client, fakes):
    fakes.script(replies=["answer 1"], grades=[True])
    ask(client, "How do I request annual leave?", "a")
    fakes.script(replies=["answer 2"], grades=[True])
    ask(client, "How do I install the VPN?", "b")  # new thread: no contextualize call
    assert fakes.queries == ["How do I request annual leave?", "How do I install the VPN?"]
    assert len(client.get("/sessions/b").json()) == 2


def test_irrelevant_docs_trigger_rewrite_and_retry(client, fakes):
    fakes.script(replies=["annual leave request process", "answer"], grades=[False, True])
    body = ask(client, "how do vacation days work")
    assert fakes.queries == ["how do vacation days work", "annual leave request process"]
    assert body["query"] == "annual leave request process"


def test_rewrites_stop_at_max_then_generate(client, fakes):
    fakes.script(replies=["rewrite 1", "rewrite 2", "final answer"], grades=[False, False, False])
    body = ask(client, "what's for lunch")
    assert len(fakes.queries) == 3  # original + max_rewrites (2)
    assert body["answer"] == "final answer"


def test_no_docs_answers_without_calling_llm_for_generation(client, fakes):
    fakes.docs = []
    fakes.script(replies=["rewrite 1", "rewrite 2"], grades=[])
    body = ask(client, "what's for lunch")
    assert body == {**body, "answer": rag.NO_ANSWER, "sources": []}


def test_ask_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_ask_stream_events(client, fakes):
    fakes.script(replies=["3 working days in advance"], grades=[True])
    r = client.post("/ask/stream", json={"question": "How do I request annual leave?"})
    events = [json.loads(line) for line in r.text.splitlines() if line]
    assert "session_id" in events[0]
    steps = [e["step"] for e in events if "step" in e]
    assert steps == ["contextualize", "retrieve", "grade", "generate"]
    streamed = "".join(e["delta"] for e in events if "delta" in e)
    assert streamed == "3 working days in advance"
    final = events[-1]
    assert final["answer"] == "3 working days in advance"
    assert final["sources"][0]["source"] == "leave-policy.md"


def test_ingest_rejects_unsupported_file(client):
    r = client.post("/ingest", files={"files": ("a.pdf", b"%PDF", "application/pdf")})
    assert r.status_code == 400


def test_format_docs_includes_source_attribute():
    assert 'source="leave-policy.md"' in rag.format_docs(DOCS)
