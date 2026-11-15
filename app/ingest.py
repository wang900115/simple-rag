"""Load Markdown / text files, split into chunks and write them to pgvector.

CLI:  python -m app.ingest data/docs
"""

import hashlib
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.config import get_settings

SUPPORTED_SUFFIXES = {".md", ".markdown", ".txt"}

_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]
# Includes CJK punctuation so Chinese documents also split on sentence boundaries.
_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", "；", "，", " ", ""]


def split_text(text: str, source: str) -> list[Document]:
    """Split one file's text into chunks with deterministic ids in metadata."""
    s = get_settings()
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=s.chunk_size,
        chunk_overlap=s.chunk_overlap,
        separators=_SEPARATORS,
        keep_separator="end",  # keep 「。」/ ". " at the end of a sentence, not the next chunk's start
    )

    if Path(source).suffix.lower() in {".md", ".markdown"}:
        sections = MarkdownHeaderTextSplitter(
            headers_to_split_on=_HEADERS, strip_headers=False
        ).split_text(text)
    else:
        sections = [Document(page_content=text)]

    chunks = char_splitter.split_documents(sections)

    docs: list[Document] = []
    for i, chunk in enumerate(c for c in chunks if c.page_content.strip()):
        headers = " > ".join(
            chunk.metadata[k] for _, k in _HEADERS if k in chunk.metadata
        )
        chunk_id = hashlib.sha256(
            f"{source}\x00{i}\x00{chunk.page_content}".encode("utf-8")
        ).hexdigest()
        docs.append(
            Document(
                id=chunk_id,
                page_content=chunk.page_content,
                metadata={"source": source, "headers": headers, "chunk_index": i},
            )
        )
    return docs


def ingest_text(text: str, source: str) -> int:
    """Replace all chunks of `source` with freshly split ones. Returns chunk count."""
    from app.vectorstore import delete_source, get_vectorstore

    docs = split_text(text, source)
    delete_source(source)
    if docs:
        get_vectorstore().add_documents(docs, ids=[d.id for d in docs])
    return len(docs)


def ingest_path(root: Path) -> dict[str, int]:
    files = [root] if root.is_file() else sorted(
        p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES
    )
    results: dict[str, int] = {}
    for path in files:
        source = path.name if root.is_file() else path.relative_to(root).as_posix()
        results[source] = ingest_text(path.read_text(encoding="utf-8"), source)
    return results


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/docs")
    if not root.exists():
        sys.exit(f"Path not found: {root}")
    results = ingest_path(root)
    for source, n in results.items():
        print(f"{source}: {n} chunks")
    print(f"Total: {len(results)} files, {sum(results.values())} chunks")


if __name__ == "__main__":
    main()
