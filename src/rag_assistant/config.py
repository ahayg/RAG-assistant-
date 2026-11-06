from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    index_dir: Path = Path(".rag_index")
    embedder: str = "auto"  # auto | hashing | <sentence-transformers model id>
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120  # characters
    top_k: int = 5
    candidates: int = 30  # per retriever, before rank fusion
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    llm_model: str = "claude-sonnet-5-5"
    max_tokens: int = 1024

    @classmethod
    def from_env(cls) -> "Settings":
        d = cls()
        return cls(
            index_dir=Path(os.getenv("RAG_INDEX_DIR", str(d.index_dir))),
            embedder=os.getenv("RAG_EMBEDDER", d.embedder),
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", d.chunk_size)),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", d.chunk_overlap)),
            top_k=int(os.getenv("RAG_TOP_K", d.top_k)),
            candidates=int(os.getenv("RAG_CANDIDATES", d.candidates)),
            dense_weight=float(os.getenv("RAG_DENSE_WEIGHT", d.dense_weight)),
            bm25_weight=float(os.getenv("RAG_BM25_WEIGHT", d.bm25_weight)),
            llm_model=os.getenv("RAG_LLM_MODEL", d.llm_model),
            max_tokens=int(os.getenv("RAG_MAX_TOKENS", d.max_tokens)),
        )
