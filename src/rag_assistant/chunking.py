from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass

from .loaders import Document


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    page: int | None
    index: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Chunk":
        return cls(**d)


_SENT = re.compile(r"(?<=[.!?])\s+")


def _hard_split(text: str, size: int) -> list[str]:
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + len(word) + 1 > size:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def _split_long_paragraph(para: str, size: int) -> list[str]:
    units, cur = [], ""
    for sent in _SENT.split(para):
        if len(sent) > size:
            if cur:
                units.append(cur)
                cur = ""
            units.extend(_hard_split(sent, size))
        elif cur and len(cur) + len(sent) + 1 > size:
            units.append(cur)
            cur = sent
        else:
            cur = f"{cur} {sent}".strip()
    if cur:
        units.append(cur)
    return units


def split_text(text: str, chunk_size: int = 800, overlap: int = 120) -> list[str]:
    """Paragraph/sentence-aware splitter.

    Packs paragraphs up to ``chunk_size`` chars; each new chunk starts with the last
    ``overlap`` chars (word-aligned) of the previous one. Chunks are therefore at most
    ``chunk_size + overlap`` chars.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")
    units: list[str] = []
    for para in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        para = re.sub(r"[ \t]+", " ", para).strip()
        if not para:
            continue
        units.extend([para] if len(para) <= chunk_size else _split_long_paragraph(para, chunk_size))

    chunks: list[str] = []
    cur = ""
    for u in units:
        if cur and len(cur) + len(u) + 2 > chunk_size:
            chunks.append(cur)
            tail = cur[-overlap:] if overlap else ""
            if tail and " " in tail:
                tail = tail[tail.index(" ") + 1 :]
            cur = f"{tail} {u}".strip() if tail else u
        else:
            cur = f"{cur}\n\n{u}" if cur else u
    if cur:
        chunks.append(cur)
    return chunks


def chunk_documents(docs: list[Document], chunk_size: int = 800, overlap: int = 120) -> list[Chunk]:
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}
    for doc in docs:
        for text in split_text(doc.text, chunk_size, overlap):
            idx = counters.get(doc.source, 0)
            counters[doc.source] = idx + 1
            cid = hashlib.sha1(f"{doc.source}|{doc.page}|{text}".encode()).hexdigest()[:16]
            chunks.append(Chunk(cid, text, doc.source, doc.page, idx))
    return chunks
