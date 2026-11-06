from __future__ import annotations

import json
from pathlib import Path

from .pipeline import RAGPipeline


def load_eval_set(path: str | Path) -> list[dict]:
    """JSONL rows: {"question": str, "expected_sources": [str], "expected_contains": str (optional)}."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_relevant(hit, item: dict) -> bool:
    ok = True
    if item.get("expected_sources"):
        ok = hit.chunk.source in item["expected_sources"]
    if ok and item.get("expected_contains"):
        ok = item["expected_contains"].lower() in hit.chunk.text.lower()
    return ok


def evaluate_retrieval(pipeline: RAGPipeline, items: list[dict], k: int = 5) -> dict:
    """Hit@1, Hit@k and MRR@k over a labelled question set."""
    hit1 = hitk = rr = 0.0
    failures = []
    for item in items:
        hits = pipeline.retrieve(item["question"], k)
        rank = next((i for i, h in enumerate(hits, 1) if _is_relevant(h, item)), None)
        if rank:
            hitk += 1
            hit1 += rank == 1
            rr += 1 / rank
        else:
            failures.append(item["question"])
    n = max(len(items), 1)
    return {
        "n": len(items),
        "k": k,
        "hit@1": round(hit1 / n, 3),
        f"hit@{k}": round(hitk / n, 3),
        f"mrr@{k}": round(rr / n, 3),
        "failures": failures,
    }
