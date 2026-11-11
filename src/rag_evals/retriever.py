"""A tiny, dependency-free BM25 retriever.

It exists only so we have a *real* system to evaluate in Branch 1.
Later branches swap in dense / hybrid retrievers and compare them
using exactly the same metrics.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Protocol

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "of", "to", "in", "on", "and", "or",
    "for", "with", "by", "be", "it", "as", "at", "that", "this", "what", "which",
    "how", "do", "does", "can", "we", "i", "my", "should", "when", "into", "from",
    "there", "their", "so", "if", "not", "its", "such", "like", "often",
}


def tokenize(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    return [t for t in tokens if t not in STOPWORDS]


class Retriever(Protocol):
    """Anything with .retrieve(query, k) -> ranked doc ids can be evaluated."""

    def retrieve(self, query: str, k: int) -> list[str]: ...


class BM25Retriever:
    def __init__(self, docs: list[dict], k1: float = 1.5, b: float = 0.75):
        self.ids = [d["id"] for d in docs]
        self.tokens = [tokenize(f"{d.get('title', '')} {d['text']}") for d in docs]
        self.tf = [Counter(t) for t in self.tokens]
        self.lengths = [len(t) for t in self.tokens]
        self.avgdl = sum(self.lengths) / len(self.lengths)
        self.k1, self.b = k1, b
        n = len(docs)
        df = Counter(term for toks in self.tokens for term in set(toks))
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, query_tokens: list[str], i: int) -> float:
        score, tf, dl = 0.0, self.tf[i], self.lengths[i]
        for term in query_tokens:
            if term not in tf:
                continue
            f = tf[term]
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            score += self.idf[term] * f * (self.k1 + 1) / denom
        return score

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        q = tokenize(query)
        scored = [(self._score(q, i), doc_id) for i, doc_id in enumerate(self.ids)]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: (-s[0], s[1]))
        return [doc_id for _, doc_id in scored[:k]]

    @classmethod
    def from_json(cls, path: str | Path) -> "BM25Retriever":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
