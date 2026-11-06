from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25
from .chunking import Chunk
from .embeddings import Embedder
from .vectorstore import VectorStore


@dataclass
class Hit:
    chunk: Chunk
    score: float  # fused RRF score
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None


class HybridRetriever:
    """Dense + BM25 retrieval fused with weighted Reciprocal Rank Fusion."""

    def __init__(
        self,
        store: VectorStore,
        embedder: Embedder,
        rrf_k: int = 60,
        dense_weight: float = 1.0,
        bm25_weight: float = 1.0,
    ):
        self.store, self.embedder = store, embedder
        self.rrf_k, self.dense_weight, self.bm25_weight = rrf_k, dense_weight, bm25_weight
        self.bm25 = BM25([c.text for c in store.chunks])

    def refresh(self) -> None:
        """Rebuild the lexical index after the store changed."""
        self.bm25 = BM25([c.text for c in self.store.chunks])

    def retrieve(self, query: str, k: int = 5, candidates: int = 30) -> list[Hit]:
        dense = self.store.search(self.embedder.embed_query(query), candidates)
        sparse = self.bm25.search(query, candidates)

        hits: dict[int, Hit] = {}
        for rank, (idx, s) in enumerate(dense, start=1):
            h = hits.setdefault(idx, Hit(self.store.chunks[idx], 0.0))
            h.dense_rank, h.dense_score = rank, s
            h.score += self.dense_weight / (self.rrf_k + rank)
        for rank, (idx, s) in enumerate(sparse, start=1):
            h = hits.setdefault(idx, Hit(self.store.chunks[idx], 0.0))
            h.bm25_rank, h.bm25_score = rank, s
            h.score += self.bm25_weight / (self.rrf_k + rank)
        return sorted(hits.values(), key=lambda h: -h.score)[:k]
