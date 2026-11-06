from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the to was were will with "
    "this these those which who what when where how why do does did can could should would".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class BM25:
    """Okapi BM25 over an in-memory corpus."""

    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(t)) for t in corpus]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        terms = [t for t in set(tokenize(query)) if t in self.idf]
        if not terms or not self.docs:
            return []
        scores: list[tuple[int, float]] = []
        for i, (doc, dl) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            for t in terms:
                tf = doc.get(t, 0)
                if tf:
                    s += self.idf[t] * tf * (self.k1 + 1) / (tf + norm)
            if s > 0:
                scores.append((i, s))
        scores.sort(key=lambda x: -x[1])
        return scores[:k]
