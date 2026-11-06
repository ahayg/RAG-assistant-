from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashingEmbedder:
    """Dependency-light, offline lexical embedder (hashed uni+bigrams, L2-normalised).

    Not semantic, but deterministic and instant: used as a fallback and in tests.
    """

    name = "hashing"

    def __init__(self, dim: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.dim = dim
        self._v = HashingVectorizer(
            n_features=dim,
            ngram_range=(1, 2),
            alternate_sign=False,
            norm="l2",
            stop_words="english",
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._v.transform(texts).toarray().astype(np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    """Semantic embeddings via sentence-transformers (pip install '.[embeddings]')."""

    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self._prefix = self.QUERY_PREFIX if "bge" in model_name.lower() else ""

    def _encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(
                texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False
            ),
            dtype=np.float32,
        )

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([self._prefix + text])[0]


def get_embedder(name: str = "auto") -> Embedder:
    """'auto' -> sentence-transformers if installed, else hashing. Anything else is a model id."""
    if name == "hashing":
        return HashingEmbedder()
    if name == "auto":
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            return HashingEmbedder()
        try:
            return SentenceTransformerEmbedder()
        except Exception:  # model download blocked / offline
            return HashingEmbedder()
    return SentenceTransformerEmbedder(name)
