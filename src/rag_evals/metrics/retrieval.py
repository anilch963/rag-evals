"""Retrieval metrics for RAG evaluation, implemented from scratch.

Every function scores ONE query. Inputs:
    retrieved: ranked list of doc ids returned by the retriever (best first)
    relevant:  ground truth for the query, either
               - a set/list of relevant doc ids (binary relevance), or
               - a dict {doc_id: grade} (graded relevance, e.g. 3=perfect, 1=partial)

Binary metrics treat any doc with grade > 0 as relevant.
Average them over all queries to get the system-level score
(e.g. the mean of reciprocal_rank over queries is MRR, mean of
average_precision is MAP).
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence, Union

Relevance = Union[Mapping[str, float], Iterable[str]]


def _relevant_set(relevant: Relevance) -> set[str]:
    if isinstance(relevant, Mapping):
        return {doc for doc, grade in relevant.items() if grade > 0}
    return set(relevant)


def _grades(relevant: Relevance) -> dict[str, float]:
    if isinstance(relevant, Mapping):
        return {doc: float(g) for doc, g in relevant.items()}
    return {doc: 1.0 for doc in relevant}


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be a positive integer, got {k}")


def hit_rate_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    """1.0 if at least one relevant doc is in the top k, else 0.0.

    Answers: "Did the retriever find *anything* useful?"
    """
    _check_k(k)
    rel = _relevant_set(relevant)
    return 1.0 if any(doc in rel for doc in retrieved[:k]) else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    """Fraction of the top k slots that hold relevant docs.

    Denominator is k (not len(retrieved)), so returning fewer than k docs
    is not rewarded. Answers: "How much noise goes into the LLM prompt?"
    """
    _check_k(k)
    rel = _relevant_set(relevant)
    hits = sum(1 for doc in retrieved[:k] if doc in rel)
    return hits / k


def recall_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    """Fraction of ALL relevant docs that appear in the top k.

    Answers: "Did the LLM get all the evidence it needs?"
    Returns 0.0 for a query with no relevant docs (nothing can be recalled).
    """
    _check_k(k)
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    hits = sum(1 for doc in retrieved[:k] if doc in rel)
    return hits / len(rel)


def reciprocal_rank(retrieved: Sequence[str], relevant: Relevance, k: int | None = None) -> float:
    """1 / rank of the first relevant doc (0.0 if none found within k).

    Mean over queries = MRR. Answers: "How high is the first good result?"
    """
    rel = _relevant_set(relevant)
    window = retrieved if k is None else retrieved[:k]
    for rank, doc in enumerate(window, start=1):
        if doc in rel:
            return 1.0 / rank
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Relevance, k: int | None = None) -> float:
    """Average of precision@i at every rank i where a relevant doc appears,
    divided by the total number of relevant docs.

    Mean over queries = MAP. Rewards ranking ALL relevant docs early,
    not only the first one (that is what makes it different from MRR).
    """
    rel = _relevant_set(relevant)
    if not rel:
        return 0.0
    window = retrieved if k is None else retrieved[:k]
    hits, score = 0, 0.0
    for rank, doc in enumerate(window, start=1):
        if doc in rel:
            hits += 1
            score += hits / rank
    return score / len(rel)


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain: sum of (2^g - 1) / log2(rank + 1)."""
    _check_k(k)
    return sum((2**g - 1) / math.log2(rank + 1) for rank, g in enumerate(gains[:k], start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Relevance, k: int) -> float:
    """DCG of the actual ranking divided by DCG of the ideal ranking.

    Uses graded relevance: a grade-3 doc at rank 1 is worth more than a
    grade-1 doc at rank 1, and any doc is worth less the lower it sits.
    Answers: "How close is the ranking to the perfect ordering?"
    """
    _check_k(k)
    grades = _grades(relevant)
    actual = [grades.get(doc, 0.0) for doc in retrieved[:k]]
    ideal = sorted(grades.values(), reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(actual, k) / idcg
