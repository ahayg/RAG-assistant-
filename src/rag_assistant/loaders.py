from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Document:
    text: str
    source: str
    page: int | None = None


def load_file(path: str | Path, source: str | None = None) -> list[Document]:
    """Load one file into Documents (one per page for PDFs)."""
    path = Path(path)
    ext = path.suffix.lower()
    name = source or path.name
    if ext == ".pdf":
        from pypdf import PdfReader

        docs = []
        for i, page in enumerate(PdfReader(str(path)).pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(text, name, i))
        return docs
    if ext in {".txt", ".md", ".markdown"}:
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        return [Document(text, name)] if text else []
    raise ValueError(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")


def iter_files(path: str | Path) -> Iterator[Path]:
    p = Path(path)
    if p.is_file():
        yield p
    elif p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield f
    else:
        raise FileNotFoundError(f"No such file or directory: {path}")


def load_path(path: str | Path) -> list[Document]:
    """Load a file, or every supported file under a directory."""
    base = Path(path)
    docs: list[Document] = []
    for f in iter_files(base):
        name = str(f.relative_to(base)) if base.is_dir() else f.name
        docs.extend(load_file(f, name))
    return docs
