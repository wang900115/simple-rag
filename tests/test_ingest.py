from app.ingest import split_text

MD = """# Title One

First paragraph.

## Section A

Content of section A.

## Section B

Content of section B.
"""


def test_markdown_split_keeps_headers_metadata():
    docs = split_text(MD, "guide.md")
    assert len(docs) == 3
    assert docs[1].metadata["headers"] == "Title One > Section A"
    assert all(d.metadata["source"] == "guide.md" for d in docs)
    assert [d.metadata["chunk_index"] for d in docs] == [0, 1, 2]


def test_ids_are_deterministic_and_unique():
    a = split_text(MD, "guide.md")
    b = split_text(MD, "guide.md")
    assert [d.id for d in a] == [d.id for d in b]
    assert len({d.id for d in a}) == len(a)
    # Same content under another source name yields different ids.
    c = split_text(MD, "other.md")
    assert not {d.id for d in a} & {d.id for d in c}


def test_long_text_is_chunked_within_size():
    text = "This is a test sentence. " * 120  # 3000 chars, no newlines
    docs = split_text(text, "long.txt")
    assert len(docs) > 1
    assert all(len(d.page_content) <= 800 for d in docs)
    assert docs[0].metadata["headers"] == ""


def test_cjk_text_splits_on_cjk_punctuation():
    # Chinese documents are still supported: no spaces, so splitting relies on 「。」.
    text = "這是一個測試句子。" * 300  # 2700 chars
    docs = split_text(text, "long-zh.txt")
    assert len(docs) > 1
    assert all(len(d.page_content) <= 800 for d in docs)
    assert all(d.page_content.endswith("。") for d in docs[:-1])


def test_blank_text_produces_no_chunks():
    assert split_text("   \n\n  ", "empty.txt") == []
