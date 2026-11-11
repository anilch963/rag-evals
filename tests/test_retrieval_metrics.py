"""Hand-computed checks for every retrieval metric.

Each expected value below can be worked out with pen and paper —
that is the point: you should trust a metric only after verifying it by hand.
"""

import math

import pytest

from rag_evals.evaluate import evaluate_retriever
from rag_evals.metrics import (
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from rag_evals.retriever import BM25Retriever

# Ranked output: relevant docs (A, C) sit at ranks 1 and 3.
RETRIEVED = ["A", "B", "C", "D", "E"]
RELEVANT = {"A", "C", "F"}  # F was never retrieved


def test_hit_rate():
    assert hit_rate_at_k(RETRIEVED, RELEVANT, 1) == 1.0
    assert hit_rate_at_k(["B", "D"], RELEVANT, 2) == 0.0


def test_precision():
    assert precision_at_k(RETRIEVED, RELEVANT, 1) == 1.0       # A
    assert precision_at_k(RETRIEVED, RELEVANT, 3) == pytest.approx(2 / 3)  # A, C
    assert precision_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 5)


def test_precision_penalises_short_lists():
    # Only 1 doc returned but we asked for 5 -> 1/5, not 1/1
    assert precision_at_k(["A"], RELEVANT, 5) == pytest.approx(1 / 5)


def test_recall():
    assert recall_at_k(RETRIEVED, RELEVANT, 1) == pytest.approx(1 / 3)
    assert recall_at_k(RETRIEVED, RELEVANT, 5) == pytest.approx(2 / 3)  # F never found
    assert recall_at_k(RETRIEVED, set(), 5) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(RETRIEVED, RELEVANT) == 1.0
    assert reciprocal_rank(["B", "C"], RELEVANT) == 0.5
    assert reciprocal_rank(["B", "D", "E", "C"], RELEVANT, k=3) == 0.0


def test_average_precision():
    # relevant at rank 1 (P=1/1) and rank 3 (P=2/3), 3 relevant in total
    expected = (1 / 1 + 2 / 3) / 3
    assert average_precision(RETRIEVED, RELEVANT) == pytest.approx(expected)


def test_ndcg_perfect_and_worst():
    graded = {"A": 3, "B": 1}
    assert ndcg_at_k(["A", "B"], graded, 2) == pytest.approx(1.0)
    assert ndcg_at_k(["X", "Y"], graded, 2) == 0.0


def test_ndcg_swapped_order():
    graded = {"A": 3, "B": 1}
    dcg = (2**1 - 1) / math.log2(2) + (2**3 - 1) / math.log2(3)
    idcg = (2**3 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(["B", "A"], graded, 2) == pytest.approx(dcg / idcg)


def test_invalid_k():
    with pytest.raises(ValueError):
        precision_at_k(RETRIEVED, RELEVANT, 0)


def test_end_to_end_bm25():
    docs = [
        {"id": "d1", "text": "cats purr and sleep"},
        {"id": "d2", "text": "dogs bark loudly"},
        {"id": "d3", "text": "python is a programming language"},
    ]
    dataset = [
        {"qid": "q1", "question": "why do dogs bark", "relevant": {"d2": 3}},
        {"qid": "q2", "question": "programming in python", "relevant": {"d3": 3}},
    ]
    result = evaluate_retriever(BM25Retriever(docs), dataset, ks=[1, 3])
    assert result["summary"]["hit@1"] == 1.0
    assert result["summary"]["MRR"] == 1.0
