import pytest

from rag_assistant.chunking import chunk_documents, split_text
from rag_assistant.loaders import Document


def test_short_text_single_chunk():
    assert split_text("Hello world.", 100, 10) == ["Hello world."]


def test_chunks_respect_size_bound():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 60 for i in range(20))
    chunks = split_text(text, 300, 50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 + 50 for c in chunks)


def test_overlap_shares_text():
    text = "\n\n".join(f"Sentence number {i} has unique token zz{i}." for i in range(30))
    chunks = split_text(text, 200, 60)
    assert len(chunks) > 2
    # tail of chunk n appears at start of chunk n+1
    shared = chunks[0].split()[-3:]
    assert all(w in chunks[1] for w in shared)


def test_very_long_sentence_is_hard_split():
    chunks = split_text("x" * 5 + " " + "word " * 500, 100, 10)
    assert all(len(c) <= 110 for c in chunks)


def test_overlap_must_be_smaller():
    with pytest.raises(ValueError):
        split_text("abc", 10, 10)


def test_chunk_ids_stable_and_unique():
    docs = [Document("Alpha beta gamma.\n\nDelta epsilon.", "a.md")]
    a = chunk_documents(docs, 20, 5)
    b = chunk_documents(docs, 20, 5)
    assert [c.id for c in a] == [c.id for c in b]
    assert len({c.id for c in a}) == len(a)
