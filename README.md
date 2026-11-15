# simple-rag

An internal knowledge-base Q&A system (PoC). Markdown and plain-text documents are imported into PostgreSQL + pgvector. Users ask questions in natural language; a local Qwen model writes an answer grounded in relevant passages and cites its sources. It supports multi-turn conversations and automatically rewrites the query and retries when retrieval comes back irrelevant.

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Project structure](#project-structure)
- [Quick start (Docker)](#quick-start-docker)
- [Development workflow](#development-workflow)
- [API](#api)
- [Configuration](#configuration)
- [Known limitations](#known-limitations)


## Features

- **Document import**: upload `.md` / `.markdown` / `.txt`. Markdown is split by headings (h1–h3), then by length. Re-importing a file with the same name replaces its old content.
- **RAG Q&A**: MMR retrieval of 5 chunks. The local Qwen model answers only from the documents and cites each point as `[Source: filename]`. When the documents have nothing relevant, it says so.
- **Multilingual**: Qwen answers in the language of the question. The bge-m3 embedding model supports more than 100 languages, so English and Chinese documents can be mixed, and chunking also splits on CJK punctuation.
- **Multi-turn conversations**: conversations are identified by `session_id`. A follow-up such as "What about sick leave?" is first rewritten into a standalone query using the history, then retrieved.
- **Self-correcting retrieval**: Qwen grades whether the retrieved chunks are relevant. If not, the query is rewritten and retrieval runs again, up to 2 times.
- **Local models**: Qwen and bge-m3 run locally through Ollama; no paid LLM or embedding API is required.
- **Streaming answers**: an NDJSON stream with processing steps, the answer token by token, and sources.
- **Web UI**: a Streamlit interface for uploading and deleting documents and chatting.

## Architecture

```
┌──────────────┐  HTTP   ┌──────────────────────────────────────────┐
│ Streamlit UI │ ──────> │ FastAPI                                  │
└──────────────┘         │                                          │
                         │  /ingest ─> ingest.py                    │
                         │             chunk → Ollama/bge-m3 ─┐      │
                         │                                   ▼      │
                         │  /ask ────> graph.py (LangGraph) ─> PGVector ──┐
                         │             └─> Ollama/Qwen3 (ChatOllama) │     │
                         └──────────────────────────────────────────┘     │
                                                                          ▼
                                          ┌─────────────────────────────────────┐
                                          │ PostgreSQL + pgvector               │
                                          │  · document vectors (langchain_pg_*)│
                                          │  · conversations (LangGraph         │
                                          │    checkpoints)                     │
                                          └─────────────────────────────────────┘
```

### Q&A flow (LangGraph)

```
START → contextualize → retrieve → grade ─┬─ relevant, or out of retries → generate → END
                           ↑              └─ not relevant → rewrite ─┐
                           └─────────────────────────────────────────┘
```

| Node | Responsibility |
|---|---|
| `contextualize` | Rewrites a follow-up into a standalone query using the history (skipped on the first turn, no LLM call) |
| `retrieve` | MMR retrieval from pgvector with the query |
| `grade` | Qwen decides, via structured output, whether the documents can answer the question |
| `rewrite` | Rewords the query, `retries + 1`, back to `retrieve` |
| `generate` | Writes the answer from the history and retrieved context; only the plain-text answer is written back to history |

Conversation history lives in the State's `messages` (with the `add_messages` reducer) and is persisted by the checkpointer per `thread_id` (the `session_id`), so each request only sends the new question.

## Tech stack

Versions are those actually installed on 2026-11-15 (see `uv.lock`).

| Category | Technology | Version | Purpose |
|---|---|---|---|
| Language | Python | 3.12 (Docker) | |
| Package management | [uv](https://docs.astral.sh/uv/) | 0.12.11 | Dependencies, virtual environment |
| Orchestration | LangGraph | 1.2.22 | RAG state graph, multi-turn memory |
| Conversation memory | langgraph-checkpoint-postgres | 3.2.1 | Persists conversation state in PostgreSQL |
| LLM framework | langchain-core | 1.8.0 | Base types: Document, Prompt, Message |
| LLM runtime | Ollama | | Runs open-weight models locally |
| LLM | langchain-ollama (Qwen3 `qwen3:4b`) | 1.1.0 | Answering, query rewriting, relevance grading |
| Vector database | PostgreSQL 17 + pgvector (`pgvector/pgvector:pg17`) | | Stores document vectors |
| Vector access | langchain-postgres (`PGVector`) | 0.0.19 | Writes, MMR retrieval |
| Embeddings | Ollama `bge-m3` via langchain-ollama | | Multilingual vectors (1024-dim), run locally |
| Chunking | langchain-text-splitters | 1.1.3 | Markdown heading split, recursive length split |
| DB drivers | psycopg 3 / SQLAlchemy | 3.3.9 / 2.1.8 | |
| API | FastAPI / Uvicorn | 0.163.0 / 0.54.0 | REST + NDJSON streaming |
| UI | Streamlit | 1.67.0 | Chat interface |
| Configuration | pydantic-settings | 2.16.3 | Reads `.env` |
| Testing | pytest, FastAPI TestClient | 9.2.0 | Tests with a fake LLM, no API cost |
| Deployment | Docker Compose | | Four services: db / ollama / api / ui |

## Project structure

```
simple-rag/
├── app/
│   ├── config.py        # All settings (pydantic-settings, reads .env)
│   ├── embeddings.py    # bge-m3 embeddings (singleton)
│   ├── vectorstore.py   # PGVector, retriever, document list/delete (hand-written SQL)
│   ├── ingest.py        # Chunking and import; also a CLI: python -m app.ingest <path>
│   ├── rag.py           # Building blocks: prompts, get_llm, retrieve, grade_documents
│   ├── graph.py         # LangGraph flow: State, nodes, wiring
│   └── main.py          # FastAPI endpoints, checkpointer setup
├── ui/streamlit_app.py  # Web UI
├── tests/
│   ├── test_ingest.py   # Chunking, metadata, deterministic IDs
│   └── test_api.py      # Graph paths and API (fake LLM, fake retrieval)
├── data/docs/           # Documents to import (currently two fictional samples)
├── docker/init.sql      # Creates the vector extension
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml / uv.lock
└── .env.example
```

Suggested reading order: `config.py` → `embeddings.py` → `ingest.py` → `vectorstore.py` → `rag.py` → `graph.py` → `main.py` → `ui/`.

## Quick start (Docker)

Requires Docker. The local models are free to run; downloading them requires several GB of disk space. CPU inference works but can be slow.

```bash
cp .env.example .env
docker compose up -d --build
docker compose exec ollama ollama pull qwen3:4b
docker compose exec ollama ollama pull bge-m3
```

| Service | URL |
|---|---|
| Web UI | http://localhost:8501 |
| API docs (Swagger) | http://localhost:8000/docs |
| PostgreSQL | `localhost:5432` (user, password and database are all `rag`) |

Ollama stores downloaded models in the persistent `ollama` volume, so rebuilding containers doesn't download them again. The Qwen3 4B model is about 2.5 GB and bge-m3 is about 1.2 GB. Both run locally; CPU-only inference may take time.

Import the sample documents:

```bash
docker compose exec api python -m app.ingest data/docs
```

If you already imported documents with the previous embedding integration, re-import each original source so the stored vectors are refreshed. Ingestion replaces chunks for a matching source name.

You can also upload files from the UI sidebar.

## Development workflow

### 1. Setup

```bash
uv sync                         # install dependencies from uv.lock (including dev: pytest, httpx)
cp .env.example .env
ollama pull qwen3:4b
ollama pull bge-m3
docker compose up -d db         # start only the database
```

Locally, run Ollama on the host at `http://localhost:11434`; `.env` uses this URL by default. The api service in Docker connects to the `ollama` Compose service. Locally the default is `CHECKPOINTER=memory`: conversations are kept in memory and lost when the API restarts. The api service in Docker uses `postgres`.

### 2. Run locally

```bash
uv run python -m app.ingest data/docs          # import documents (first pull both models in Ollama)
uv run uvicorn app.main:app --reload           # API: http://localhost:8000
uv run streamlit run ui/streamlit_app.py       # UI: http://localhost:8501 (in another terminal)
```

### 3. Tests

```bash
uv run pytest
```

Tests need no database, embedding model or API key:

- `retrieve`, `grade_documents` and `get_llm` are monkeypatched with fakes.
- The LLM is a `GenericFakeChatModel` that returns scripted replies in order.
- The FastAPI lifespan is skipped; the checkpointer is an `InMemorySaver`.

When you add a graph branch or node, add a test for that path in `tests/test_api.py` with `fakes.script(replies=[...], grades=[...])`.

### 4. Common changes

| Goal | Where |
|---|---|
| Tune chunk size, retrieval count, retries, history length | `.env` (fields in `app/config.py`) |
| Change answer rules or tone | `SYSTEM_PROMPT` in `app/rag.py` |
| Add a flow step (e.g. permission filtering) | `app/graph.py`: write the node function → `add_node` → wire the edges; add a test |
| Support a new file format (e.g. PDF) | `app/ingest.py`: `SUPPORTED_SUFFIXES` and file reading |
| Switch the embedding model | `EMBEDDING_MODEL` in `.env`. **If the vector dimension changes, clear the old collection and re-import** |

### 5. Dependencies

```bash
uv add <package>          # runtime dependency
uv add --dev <package>    # dev dependency
```

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Checks the database connection |
| `POST` | `/ingest` | Multipart upload of `.md`/`.txt` (field name `files`, multiple allowed) |
| `POST` | `/ask` | `{"question": "...", "session_id": "optional"}` → `{session_id, answer, query, sources}` |
| `POST` | `/ask/stream` | Same, as an NDJSON stream |
| `GET` | `/sessions/{session_id}` | Conversation history |
| `GET` | `/documents` | Imported documents and chunk counts |
| `DELETE` | `/documents/{source}` | Deletes all chunks of a document |

Omitting `session_id` starts a new conversation; pass the returned `session_id` in the next request to continue it.

Each line of `/ask/stream` is one JSON event, in this order:

```jsonc
{"session_id": "..."}
{"step": "contextualize", "query": "..."}
{"step": "retrieve"}
{"step": "grade", "relevant": true}
{"delta": "3 working"}                                    // answer fragment, repeated
{"answer": "...", "query": "...", "sources": [...]}      // final result, authoritative
```

## Configuration

Every setting can be overridden with an environment variable or `.env` (the upper-cased field name from `app/config.py`).

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+psycopg://rag:rag@localhost:5432/rag` | Overridden by compose inside Docker |
| `COLLECTION_NAME` | `simple_rag` | PGVector collection name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Overridden by compose inside Docker |
| `EMBEDDING_MODEL` | `bge-m3` | Ollama embedding model |
| `LLM_MODEL` | `qwen3:4b` | Ollama chat model |
| `LLM_MAX_TOKENS` | `16000` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `100` | In characters |
| `RETRIEVER_K` / `RETRIEVER_FETCH_K` | `5` / `20` | MMR returns k chunks chosen from fetch_k candidates |
| `CHECKPOINTER` | `memory` | `memory` or `postgres` |
| `MAX_REWRITES` | `2` | Maximum query rewrites when retrieval is irrelevant |
| `HISTORY_TURNS` | `6` | Prior Q&A turns sent to the model |

## Known limitations

**Not yet verified**
- End-to-end Docker Compose startup, real pgvector reads and writes, real Postgres checkpointer writes, and the quality/performance of local Qwen answers.
- `list_sources` / `delete_source` in `vectorstore.py` query the `langchain_pg_embedding` / `langchain_pg_collection` tables directly. The table and column names are inferred from langchain-postgres conventions and haven't been checked against a real database.

**By design**
- No login or access control: everyone can see and delete every document.
- Only UTF-8 Markdown and plain text are supported.
- The fixed fallback messages (no documents found, refusal) are in English, since they are returned without calling the model and can't follow the question's language.
- Each question makes 2 to 7 local-model calls (grading, rewriting, generation), so CPU inference can be slow.
- `AsyncPostgresSaver.from_conn_string` uses a single connection and isn't suited to high concurrency.
- On Windows, psycopg's async mode may not work with the default event loop, so local development defaults to the `memory` checkpointer (an inference, not tested).
- Embedding many documents with bge-m3 on CPU is slow.
