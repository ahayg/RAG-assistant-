from fastapi.testclient import TestClient

from rag_assistant.api import create_app


def test_api_flow(make_pipeline):
    client = TestClient(create_app(make_pipeline()))
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/").status_code == 200

    # empty index -> 409
    assert client.post("/query", json={"question": "hi"}).status_code == 409

    files = [("files", ("notes.md", b"# Cats\n\nCats purr when content. Dogs bark loudly.", "text/markdown"))]
    r = client.post("/ingest", files=files)
    assert r.status_code == 200 and r.json()["chunks_added"] >= 1

    r = client.post("/query", json={"question": "When do cats purr?", "top_k": 2})
    body = r.json()
    assert r.status_code == 200
    assert body["sources"][0]["source"] == "notes.md"
    assert client.get("/stats").json()["chunks"] >= 1


def test_api_rejects_bad_type_and_empty_question(make_pipeline):
    client = TestClient(create_app(make_pipeline()))
    r = client.post("/ingest", files=[("files", ("x.exe", b"zzz", "application/octet-stream"))])
    assert r.status_code == 400
    assert client.post("/query", json={"question": ""}).status_code == 422
