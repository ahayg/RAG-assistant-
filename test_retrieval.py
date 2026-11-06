import numpy as np

from rag_assistant.bm25 import BM25
from rag_assistant.chunking import Chunk
from rag_assistant.vectorstore import VectorStore


def test_bm25_ranks_exact_term_first():
    bm = BM25(["the cat sat on the mat", "error code E1234 occurred", "dogs bark loudly"])
    assert bm.search("E1234", 3)[0][0] == 1
    assert bm.search("zzzz", 3) == []


def test_vectorstore_search_and_roundtrip(tmp_path):
    chunks = [Chunk(f"id{i}", f"t{i}", "s.md", None, i) for i in range(3)]
    vecs = np.eye(3, dtype=np.float32)
    store = VectorStore(3)
    assert store.add(chunks, vecs) == 3
    assert store.add(chunks, vecs) == 0  # dedupe
    assert store.search(np.array([0, 1, 0], dtype=np.float32), 1)[0][0] == 1

    store.save(tmp_path, {"embedder": "x", "dim": 3})
    loaded = VectorStore.load(tmp_path)
    assert len(loaded) == 3
    assert loaded.chunks[2].id == "id2"
    assert VectorStore.read_meta(tmp_path)["dim"] == 3


def test_hybrid_retrieval_finds_right_doc(pipeline):
    hits = pipeline.retrieve("What does the nprobe parameter control in IVF?", 3)
    assert hits[0].chunk.source == "vector_databases.md"
    assert hits[0].bm25_rank is not None and hits[0].dense_rank is not None


def test_hybrid_beats_nothing_on_keyword_only_query(pipeline):
    hits = pipeline.retrieve("pgvector", 3)
    assert any(h.chunk.source == "vector_databases.md" for h in hits)
