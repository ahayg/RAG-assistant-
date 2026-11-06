import pytest
from conftest import EVAL_SET, SAMPLE_DOCS

from rag_assistant.config import Settings
from rag_assistant.embeddings import HashingEmbedder
from rag_assistant.evaluation import evaluate_retrieval, load_eval_set
from rag_assistant.generator import AnthropicGenerator, build_user_message
from rag_assistant.pipeline import RAGPipeline


def test_ingest_is_idempotent(make_pipeline):
    p = make_pipeline()
    first = p.ingest(SAMPLE_DOCS)
    second = p.ingest(SAMPLE_DOCS)
    assert first.chunks_added > 0
    assert second.chunks_added == 0 and second.chunks_skipped == first.chunks_added


def test_index_persists_across_instances(make_pipeline, tmp_path):
    p = make_pipeline()
    p.ingest(SAMPLE_DOCS)
    again = RAGPipeline(
        Settings(index_dir=tmp_path / "idx", embedder="hashing"), HashingEmbedder()
    )
    assert len(again.store) == len(p.store)
    assert again.retrieve("reciprocal rank fusion", 1)[0].chunk.source == "rag_overview.md"


def test_embedder_mismatch_raises(make_pipeline, tmp_path):
    make_pipeline().ingest(SAMPLE_DOCS)

    class Other(HashingEmbedder):
        name = "other"

    with pytest.raises(ValueError, match="built with embedder"):
        RAGPipeline(Settings(index_dir=tmp_path / "idx"), Other())


def test_ask_returns_sources(pipeline):
    ans = pipeline.ask("How is MRR computed?", 3)
    assert len(ans.sources) == 3 and ans.answer


def test_reset(pipeline):
    pipeline.reset()
    assert len(pipeline.store) == 0 and pipeline.ask("anything").sources == []


def test_retrieval_eval_meets_baseline(pipeline):
    r = evaluate_retrieval(pipeline, load_eval_set(EVAL_SET), k=5)
    assert r["hit@5"] >= 0.8, r["failures"]


class _FakeClient:
    class messages:
        captured = {}

        @classmethod
        def create(cls, **kw):
            cls.captured = kw

            class B:
                type = "text"
                text = " Answer [1]. "

            class R:
                content = [B()]

            return R()


def test_anthropic_generator_prompt(pipeline):
    hits = pipeline.retrieve("What is HNSW?", 2)
    gen = AnthropicGenerator("test-model", 100, client=_FakeClient())
    assert gen.generate("What is HNSW?", hits) == "Answer [1]."
    sent = _FakeClient.messages.captured
    assert sent["model"] == "test-model" and "ONLY from the numbered context" in sent["system"]
    assert "<context>" in sent["messages"][0]["content"]
    assert build_user_message("q", hits).endswith("Question: q")
