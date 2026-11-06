from __future__ import annotations

import argparse
import json
import sys

from .config import Settings
from .evaluation import evaluate_retrieval, load_eval_set
from .pipeline import RAGPipeline


def _print_answer(ans) -> None:
    print(f"\n{ans.answer}\n")
    print("Sources:")
    for i, h in enumerate(ans.sources, 1):
        page = f", p.{h.chunk.page}" if h.chunk.page else ""
        print(f"  [{i}] {h.chunk.source}{page}  (rrf={h.score:.4f})")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="rag", description="RAG assistant")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("ingest", help="Index files or directories (.txt .md .pdf)")
    s.add_argument("paths", nargs="+")

    s = sub.add_parser("ask", help="Ask one question")
    s.add_argument("question")
    s.add_argument("-k", type=int, default=None)

    sub.add_parser("chat", help="Interactive Q&A loop")

    s = sub.add_parser("serve", help="Run the FastAPI server + web UI")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)

    s = sub.add_parser("eval", help="Retrieval eval (hit@k, MRR) on a JSONL file")
    s.add_argument("file")
    s.add_argument("-k", type=int, default=5)

    sub.add_parser("stats", help="Show index stats")
    sub.add_parser("reset", help="Delete the index")

    args = p.parse_args(argv)

    if args.cmd == "serve":
        import uvicorn

        from .api import create_app

        uvicorn.run(create_app(), host=args.host, port=args.port)
        return 0

    pipe = RAGPipeline(Settings.from_env())

    if args.cmd == "ingest":
        for path in args.paths:
            st = pipe.ingest(path)
            print(
                f"{path}: {st.documents} docs, +{st.chunks_added} chunks "
                f"({st.chunks_skipped} duplicates), total {st.total_chunks}"
            )
    elif args.cmd == "ask":
        _print_answer(pipe.ask(args.question, args.k))
    elif args.cmd == "chat":
        print("Ask a question (Ctrl-D / 'exit' to quit).")
        while True:
            try:
                q = input("\n> ").strip()
            except EOFError:
                break
            if q.lower() in {"exit", "quit"}:
                break
            if q:
                _print_answer(pipe.ask(q))
    elif args.cmd == "eval":
        print(json.dumps(evaluate_retrieval(pipe, load_eval_set(args.file), args.k), indent=2))
    elif args.cmd == "stats":
        print(json.dumps(pipe.stats(), indent=2))
    elif args.cmd == "reset":
        pipe.reset()
        print("Index deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
