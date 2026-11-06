from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .loaders import SUPPORTED_EXTENSIONS, load_file
from .pipeline import RAGPipeline

MAX_UPLOAD_BYTES = 25 * 1024 * 1024


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(5, ge=1, le=20)


class SourceOut(BaseModel):
    source: str
    page: int | None
    score: float
    dense_rank: int | None
    bm25_rank: int | None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceOut]


def create_app(pipeline: RAGPipeline | None = None) -> FastAPI:
    app = FastAPI(title="RAG Assistant", version="0.1.0")
    state: dict = {"pipeline": pipeline}

    def get_pipeline() -> RAGPipeline:
        if state["pipeline"] is None:
            state["pipeline"] = RAGPipeline()
        return state["pipeline"]

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/stats")
    def stats():
        return get_pipeline().stats()

    @app.post("/ingest")
    async def ingest(files: list[UploadFile] = File(...)):
        pipe = get_pipeline()
        docs = []
        with tempfile.TemporaryDirectory() as tmp:
            for up in files:
                name = Path(up.filename or "upload").name
                if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                    raise HTTPException(400, f"Unsupported file type: {name}")
                data = await up.read()
                if len(data) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, f"{name} exceeds {MAX_UPLOAD_BYTES // 2**20} MB")
                p = Path(tmp) / name
                p.write_bytes(data)
                docs.extend(load_file(p, name))
        st = pipe.ingest_documents(docs)
        return st.__dict__

    @app.post("/query", response_model=QueryResponse)
    def query(req: QueryRequest):
        pipe = get_pipeline()
        if not len(pipe.store):
            raise HTTPException(409, "Index is empty. Ingest documents first.")
        ans = pipe.ask(req.question, req.top_k)
        return QueryResponse(
            question=ans.question,
            answer=ans.answer,
            sources=[
                SourceOut(
                    source=h.chunk.source,
                    page=h.chunk.page,
                    score=round(h.score, 5),
                    dense_rank=h.dense_rank,
                    bm25_rank=h.bm25_rank,
                    text=h.chunk.text,
                )
                for h in ans.sources
            ],
        )

    @app.get("/", response_class=HTMLResponse)
    def ui():
        return UI_HTML

    return app


UI_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>RAG Assistant</title>
<style>
:root{--bg:#fff;--fg:#1a1a1a;--mut:#666;--card:#f5f5f4;--bd:#ddd;--ac:#2f5fd0}
@media(prefers-color-scheme:dark){:root{--bg:#161616;--fg:#eee;--mut:#9a9a9a;--card:#222;--bd:#333;--ac:#7aa2ff}}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
main{max-width:780px;margin:0 auto;padding:24px 16px}
h1{font-size:1.4rem;margin:0 0 16px}
.row{display:flex;gap:8px;margin-bottom:16px;flex-wrap:wrap}
input[type=text]{flex:1;min-width:200px;padding:10px;border:1px solid var(--bd);border-radius:8px;background:var(--card);color:var(--fg)}
button{padding:10px 16px;border:0;border-radius:8px;background:var(--ac);color:#fff;cursor:pointer}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px;margin-bottom:12px;white-space:pre-wrap}
.src{font-size:.85rem;color:var(--mut)} small{color:var(--mut)}
</style></head><body><main>
<h1>RAG Assistant</h1>
<div class="row"><input type="file" id="f" multiple accept=".txt,.md,.pdf"><button onclick="up()">Ingest</button><small id="st"></small></div>
<div class="row"><input type="text" id="q" placeholder="Ask a question about your documents" onkeydown="if(event.key==='Enter')ask()"><button onclick="ask()">Ask</button></div>
<div id="out"></div>
<script>
const $=id=>document.getElementById(id);
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function up(){const fs=$('f').files;if(!fs.length)return;const fd=new FormData();for(const f of fs)fd.append('files',f);
 $('st').textContent='Ingesting...';const r=await fetch('/ingest',{method:'POST',body:fd});const j=await r.json();
 $('st').textContent=r.ok?`+${j.chunks_added} chunks (total ${j.total_chunks})`:(j.detail||'error');}
async function ask(){const q=$('q').value.trim();if(!q)return;$('out').innerHTML='<div class="card">Thinking...</div>';
 const r=await fetch('/query',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
 const j=await r.json();if(!r.ok){$('out').innerHTML=`<div class="card">${esc(j.detail||'error')}</div>`;return;}
 $('out').innerHTML=`<div class="card">${esc(j.answer)}</div>`+j.sources.map((s,i)=>
 `<div class="card"><div class="src">[${i+1}] ${esc(s.source)}${s.page?', p.'+s.page:''}</div>${esc(s.text)}</div>`).join('');}
</script></main></body></html>"""
