"""Run retrieval evaluation over a golden dataset.

Usage:
    python -m rag_evals.evaluate
    python -m rag_evals.evaluate --k 1 3 5 --out results/bm25.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from .metrics import (
    average_precision,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from .retriever import BM25Retriever, Retriever

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"


def load_dataset(path: str | Path) -> list[dict]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def evaluate_retriever(retriever: Retriever, dataset: list[dict], ks: list[int]) -> dict:
    """Score every query, then average. Returns {'summary': ..., 'per_query': ...}."""
    max_k = max(ks)
    per_query = []
    for item in dataset:
        retrieved = retriever.retrieve(item["question"], k=max_k)
        rel = item["relevant"]
        row = {
            "qid": item["qid"],
            "question": item["question"],
            "retrieved": retrieved,
            "relevant": rel,
            "rr": reciprocal_rank(retrieved, rel, max_k),
            "ap": average_precision(retrieved, rel, max_k),
        }
        for k in ks:
            row[f"hit@{k}"] = hit_rate_at_k(retrieved, rel, k)
            row[f"precision@{k}"] = precision_at_k(retrieved, rel, k)
            row[f"recall@{k}"] = recall_at_k(retrieved, rel, k)
            row[f"ndcg@{k}"] = ndcg_at_k(retrieved, rel, k)
        per_query.append(row)

    metric_keys = [key for key in per_query[0] if key not in {"qid", "question", "retrieved", "relevant"}]
    summary = {key: round(mean(r[key] for r in per_query), 4) for key in metric_keys}
    summary["MRR"] = summary.pop("rr")
    summary["MAP"] = summary.pop("ap")
    return {"num_queries": len(per_query), "summary": summary, "per_query": per_query}


def print_report(result: dict, ks: list[int]) -> None:
    s = result["summary"]
    print(f"\nRetrieval evaluation on {result['num_queries']} queries\n")
    print(f"{'k':>4} | {'Hit':>6} | {'Prec':>6} | {'Recall':>6} | {'NDCG':>6}")
    print("-" * 42)
    for k in ks:
        print(f"{k:>4} | {s[f'hit@{k}']:>6.3f} | {s[f'precision@{k}']:>6.3f} | "
              f"{s[f'recall@{k}']:>6.3f} | {s[f'ndcg@{k}']:>6.3f}")
    print(f"\nMRR@{max(ks)}: {s['MRR']:.3f}    MAP@{max(ks)}: {s['MAP']:.3f}")

    # Failure analysis: the most useful part of any eval is *which* queries fail.
    k = max(ks)
    misses = [r for r in result["per_query"] if r[f"hit@{k}"] == 0]
    weak = [r for r in result["per_query"] if r[f"hit@{k}"] == 1 and r["rr"] < 1.0]
    if misses:
        print(f"\nMissed entirely (no relevant doc in top {k}):")
        for r in misses:
            print(f"  {r['qid']}: {r['question']}\n      got {r['retrieved']}  expected {list(r['relevant'])}")
    if weak:
        print("\nFound, but not at rank 1:")
        for r in weak:
            print(f"  {r['qid']}: first relevant at rank {round(1 / r['rr'])} -> {r['retrieved']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a retriever with classic IR metrics.")
    parser.add_argument("--corpus", default=DATA / "corpus.json")
    parser.add_argument("--dataset", default=DATA / "eval_dataset.json")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--out", help="optional path to save JSON results")
    args = parser.parse_args()

    ks = sorted(set(args.k))
    retriever = BM25Retriever.from_json(args.corpus)
    result = evaluate_retriever(retriever, load_dataset(args.dataset), ks)
    print_report(result, ks)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"\nSaved results to {out}")


if __name__ == "__main__":
    main()
