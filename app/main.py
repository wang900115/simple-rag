import json
import uuid
from contextlib import asynccontextmanager
from pathlib import PurePath

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field

from app.config import get_settings
from app.graph import build_graph, result_payload
from app.ingest import SUPPORTED_SUFFIXES, ingest_text


@asynccontextmanager
async def lifespan(app: FastAPI):
    from app.vectorstore import get_vectorstore

    # Load the embedding model and create tables up front so the first request is fast.
    await run_in_threadpool(get_vectorstore)

    if get_settings().checkpointer == "postgres":
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

        async with AsyncPostgresSaver.from_conn_string(get_settings().psycopg_url) as saver:
            await saver.setup()  # creates checkpoint tables on first run
            app.state.graph = build_graph(saver)
            yield
    else:
        app.state.graph = build_graph(InMemorySaver())
        yield


app = FastAPI(title="simple-rag", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str | None = Field(default=None, description="Omit to start a new conversation")


class Source(BaseModel):
    source: str | None
    headers: str
    snippet: str


class AskResponse(BaseModel):
    session_id: str
    answer: str
    query: str  # the search query actually used (after contextualize / rewrite)
    sources: list[Source]


def _graph_args(req: AskRequest) -> tuple[dict, dict, str]:
    session_id = req.session_id or uuid.uuid4().hex
    inputs = {"messages": [HumanMessage(content=req.question)]}
    config = {"configurable": {"thread_id": session_id}}
    return inputs, config, session_id


@app.get("/health")
def health():
    from app.vectorstore import ping

    try:
        ping()
    except Exception as e:  # noqa: BLE001
        raise HTTPException(503, f"database unavailable: {e}") from e
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile]):
    results = {}
    for f in files:
        name = PurePath(f.filename or "").name
        if PurePath(name).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, f"Unsupported file type: {name} (only .md / .txt are accepted)")
        try:
            text = (await f.read()).decode("utf-8")
        except UnicodeDecodeError as e:
            raise HTTPException(400, f"{name} is not UTF-8 encoded") from e
        results[name] = await run_in_threadpool(ingest_text, text, name)
    return {"files": results, "total_chunks": sum(results.values())}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    inputs, config, session_id = _graph_args(req)
    state = await request.app.state.graph.ainvoke(inputs, config)
    return {"session_id": session_id, **result_payload(state)}


@app.post("/ask/stream")
async def ask_stream(req: AskRequest, request: Request):
    """NDJSON stream. Event lines:
    {"session_id"} → {"step", "query"?}* → {"delta"}* → {"answer", "query", "sources"}
    """
    graph = request.app.state.graph
    inputs, config, session_id = _graph_args(req)

    async def gen():
        yield _line({"session_id": session_id})
        try:
            async for mode, chunk in graph.astream(
                inputs, config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node, update in chunk.items():
                        event = {"step": node}
                        if node in ("contextualize", "rewrite"):
                            event["query"] = update["query"]
                        if node == "grade":
                            event["relevant"] = update["relevant"]
                        yield _line(event)
                else:
                    message, metadata = chunk
                    # Only stream token chunks of the final answer: skip helper LLM calls,
                    # and skip the complete AIMessage the node writes to state afterwards.
                    if (
                        metadata.get("langgraph_node") == "generate"
                        and isinstance(message, AIMessageChunk)
                        and message.text
                    ):
                        yield _line({"delta": message.text})
            state = (await graph.aget_state(config)).values
            yield _line(result_payload(state))
        except Exception as e:  # noqa: BLE001
            yield _line({"error": str(e)})

    return StreamingResponse(gen(), media_type="application/x-ndjson")


@app.get("/sessions/{session_id}")
async def session_history(session_id: str, request: Request):
    config = {"configurable": {"thread_id": session_id}}
    state = await request.app.state.graph.aget_state(config)
    messages = state.values.get("messages", [])
    return [{"role": m.type, "content": m.text} for m in messages]


@app.get("/documents")
def documents():
    from app.vectorstore import list_sources

    return list_sources()


@app.delete("/documents/{source:path}")
def delete_document(source: str):
    from app.vectorstore import delete_source

    deleted = delete_source(source)
    if not deleted:
        raise HTTPException(404, f"Document not found: {source}")
    return {"deleted_chunks": deleted}


def _line(obj: dict) -> str:
    return json.dumps(obj, ensure_ascii=False) + "\n"
