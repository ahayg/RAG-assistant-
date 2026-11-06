from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from .chunking import chunk_documents
from .config import Settings
from .embeddings import Embedder, get_embedder
from .generator import Generator, get_generator
from .loaders import Document, load_path
from .retriever import Hit, HybridRetriever
from .vectorstore import VectorStore

BATCH = 64


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Hit]


@dataclass
class IngestStats:
    documents: int
    chunks_added: int
    chunks_skipped: int
    total_chunks: int


class RAGPipeline:
    """Ingest -> chunk -> embed -> index (dense + BM25) -> retrieve (RRF) -> generate."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedder: Embedder | None = None,
        generator: Generator | None = None,
    ):
        self.settings = settings or Settings.from_env()
        s = self.settings
        meta = VectorStore.read_meta(s.index_dir)

        if embedder is None:
            # An existing index pins its embedder so vectors stay comparable.
            name = meta["embedder"] if meta and s.embedder == "auto" else s.embedder
            embedder = get_embedder(name)
        self.embedder = embedder
        self.generator = generator or get_generator(s.llm_model, s.max_tokens)

        if meta:
            if meta["embedder"] != embedder.name or meta["dim"] != embedder.dim:
                raise ValueError(
                    f"Index in {s.index_dir} was built with embedder '{meta['embedder']}' "
                    f"(dim {meta['dim']}), but '{embedder.name}' (dim {embedder.dim}) is active. "
                    "Reset the index or set RAG_EMBEDDER to match."
                )
            self.store = VectorStore.load(s.index_dir)
        else:
            self.store = VectorStore(embedder.dim)
        self.retriever = HybridRetriever(
            self.store, embedder, dense_weight=s.dense_weight, bm25_weight=s.bm25_weight
        )

    # ---------- ingestion ----------
    def ingest(self, path: str | Path) -> IngestStats:
        return self.ingest_documents(load_path(path))

    def ingest_documents(self, docs: list[Document]) -> IngestStats:
        s = self.settings
        chunks = chunk_documents(docs, s.chunk_size, s.chunk_overlap)
        fresh = [c for c in chunks if c.id not in self.store._ids]
        added = 0
        for i in range(0, len(fresh), BATCH):
            batch = fresh[i : i + BATCH]
            vecs = self.embedder.embed_documents([c.text for c in batch])
            added += self.store.add(batch, vecs)
        if added:
            self.retriever.refresh()
            self.save()
        return IngestStats(len(docs), added, len(chunks) - added, len(self.store))

    def save(self) -> None:
        self.store.save(
            self.settings.index_dir, {"embedder": self.embedder.name, "dim": self.embedder.dim}
        )

    def reset(self) -> None:
        shutil.rmtree(self.settings.index_dir, ignore_errors=True)
        self.store = VectorStore(self.embedder.dim)
        self.retriever = HybridRetriever(
            self.store,
            self.embedder,
            dense_weight=self.settings.dense_weight,
            bm25_weight=self.settings.bm25_weight,
        )

    # ---------- querying ----------
    def retrieve(self, question: str, k: int | None = None) -> list[Hit]:
        s = self.settings
        return self.retriever.retrieve(question, k or s.top_k, s.candidates)

    def ask(self, question: str, k: int | None = None) -> Answer:
        hits = self.retrieve(question, k)
        return Answer(question, self.generator.generate(question, hits), hits)

    def stats(self) -> dict:
        sources = sorted({c.source for c in self.store.chunks})
        return {
            "chunks": len(self.store),
            "sources": sources,
            "embedder": self.embedder.name,
            "llm": type(self.generator).__name__,
        }
