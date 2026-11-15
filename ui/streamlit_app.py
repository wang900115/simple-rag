import json
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="simple-rag Knowledge Base", page_icon="📚")
st.title("📚 simple-rag Knowledge Base Q&A")

with st.sidebar:
    st.header("Documents")
    uploads = st.file_uploader(
        "Upload Markdown / plain text", type=["md", "markdown", "txt"], accept_multiple_files=True
    )
    if uploads and st.button("Import", type="primary"):
        files = [("files", (f.name, f.getvalue(), "text/plain")) for f in uploads]
        with st.spinner("Chunking and embedding…"):
            r = requests.post(f"{API_URL}/ingest", files=files, timeout=600)
        if r.ok:
            st.success(f"Imported {r.json()['total_chunks']} chunks")
        else:
            st.error(r.text)

    st.subheader("Imported")
    try:
        docs = requests.get(f"{API_URL}/documents", timeout=10).json()
        for d in docs:
            col1, col2 = st.columns([4, 1])
            col1.write(f"{d['source']} ({d['chunks']})")
            if col2.button("🗑", key=f"del-{d['source']}"):
                requests.delete(f"{API_URL}/documents/{d['source']}", timeout=30)
                st.rerun()
        if not docs:
            st.caption("No documents yet")
    except requests.RequestException as e:
        st.error(f"Cannot reach the API: {e}")

if "history" not in st.session_state:
    st.session_state.history = []
    st.session_state.session_id = None  # assigned by the API on the first question

with st.sidebar:
    st.divider()
    if st.button("🆕 New conversation"):
        st.session_state.history = []
        st.session_state.session_id = None
        st.rerun()
    if st.session_state.session_id:
        st.caption(f"session: {st.session_state.session_id[:8]}…")

STEP_LABELS = {
    "contextualize": "Understanding the question",
    "retrieve": "Retrieving documents",
    "grade": "Grading relevance",
    "rewrite": "Rewriting the query",
    "generate": "Generating the answer",
}


def describe_step(event: dict) -> str:
    label = STEP_LABELS.get(event["step"], event["step"])
    if "query" in event:
        return f"{label}: “{event['query']}”"
    if "relevant" in event:
        return f"{label}: {'relevant ✅' if event['relevant'] else 'not relevant ❌'}"
    return label


def render_sources(sources):
    if sources:
        with st.expander(f"Sources ({len(sources)})"):
            for s in sources:
                st.markdown(f"**{s['source']}** {('— ' + s['headers']) if s['headers'] else ''}")
                st.caption(s["snippet"])


for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("sources"))

if question := st.chat_input("Ask a question…"):
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        status = st.status("Working…", expanded=False)
        placeholder = st.empty()
        answer, sources = "", []
        try:
            with requests.post(
                f"{API_URL}/ask/stream",
                json={"question": question, "session_id": st.session_state.session_id},
                stream=True,
                timeout=600,
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    event = json.loads(line)
                    if "session_id" in event:
                        st.session_state.session_id = event["session_id"]
                    elif "step" in event:
                        status.write(describe_step(event))
                    elif "delta" in event:
                        answer += event["delta"]
                        placeholder.markdown(answer + "▌")
                    elif "answer" in event:
                        # Final state is authoritative (covers no-answer / refusal paths).
                        answer, sources = event["answer"], event["sources"]
                    elif "error" in event:
                        answer += f"\n\n⚠️ {event['error']}"
            status.update(label="Done", state="complete")
            placeholder.markdown(answer)
        except requests.RequestException as e:
            answer = f"⚠️ Request failed: {e}"
            placeholder.markdown(answer)
        render_sources(sources)

    st.session_state.history.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
