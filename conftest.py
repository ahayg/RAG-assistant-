from pathlib import Path

import pytest

from rag_assistant.config import Settings
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.generator import ExtractiveGenerator
from rag_assistant.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DOCS = ROOT / "data" / "sample_docs"
EVAL_SET = ROOT / "data" / "eval" / "qa.jsonl"


@pytest.fixture
def make_pipeline(tmp_path):
    def _make(generator=None, **kw):
        settings = Settings(index_dir=tmp_path / "idx", embedder="hashing", **kw)
        return RAGPipeline(settings, HashingEmbedder(), generator or ExtractiveGenerator())

    return _make


@pytest.fixture
def pipeline(make_pipeline):
    p = make_pipeline()
    p.ingest(SAMPLE_DOCS)
    return p
