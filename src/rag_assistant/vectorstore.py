from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .chunking import Chunk


class VectorStore:
    """Exact cosine-similarity store (vectors are pre-normalised) persisted as npy + jsonl."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self.chunks: list[Chunk] = []
        self._ids: set[str] = set()

    def __len__(self) -> int:
        return len(self.chunks)

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> int:
        """Add chunks, skipping ids already present. Returns number added."""
        keep = [i for i, c in enumerate(chunks) if c.id not in self._ids]
        if not keep:
            return 0
        self.vectors = np.vstack([self.vectors, vectors[keep].astype(np.float32)])
        for i in keep:
            self.chunks.append(chunks[i])
            self._ids.add(chunks[i].id)
        return len(keep)

    def search(self, query: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.chunks) or k <= 0:
            return []
        scores = self.vectors @ query.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [(int(i), float(scores[i])) for i in top]

    def save(self, directory: str | Path, meta: dict) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        with open(d / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps(c.to_dict(), ensure_ascii=False) + "\n")
        (d / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def read_meta(cls, directory: str | Path) -> dict | None:
        p = Path(directory) / "meta.json"
        return json.loads(p.read_text()) if p.exists() else None

    @classmethod
    def load(cls, directory: str | Path) -> "VectorStore":
        d = Path(directory)
        vectors = np.load(d / "vectors.npy")
        store = cls(vectors.shape[1])
        store.vectors = vectors
        with open(d / "chunks.jsonl", encoding="utf-8") as f:
            store.chunks = [Chunk.from_dict(json.loads(line)) for line in f if line.strip()]
        store._ids = {c.id for c in store.chunks}
        return store
