# rag-evals

Learning **RAG evaluation** one concept at a time. Each concept lives on its own branch and builds on the one before it.

## Roadmap (one branch per concept)

| # | Branch | Concept | Needs an LLM? |
|---|--------|---------|---------------|
| 01 | `01-retrieval-metrics` | Golden dataset + Hit Rate, Precision@k, Recall@k, MRR, MAP, NDCG | No |
| 02 | `02-dense-vs-sparse` | Compare BM25 vs embeddings vs hybrid with the same metrics | Embeddings only |
| 03 | `03-generation-metrics` | Faithfulness, answer relevancy, correctness (LLM-as-judge from scratch) | Yes |
| 04 | `04-context-metrics` | Context precision / context recall (claim-level) | Yes |
| 05 | `05-ragas` | Same metrics with the RAGAS library, compare to our versions | Yes |
| 06 | `06-deepeval` | DeepEval + pytest-style assertions for RAG | Yes |
| 07 | `07-synthetic-testset` | Generating a golden dataset with an LLM | Yes |
| 08 | `08-judge-calibration` | Checking the judge itself: agreement with human labels | Yes |
| 09 | `09-ci-regression` | Eval gates in GitHub Actions (fail the build if scores drop) | Optional |
| 10 | `10-tracing-observability` | Langfuse / LangSmith traces + online evals | Yes |

---

## Branch 01: Retrieval metrics

### Why start here?
A RAG answer can only be as good as the context it was given. If the retriever misses the right document, no LLM can fix it. So **retrieval is measured first**, separately from generation. These metrics come from classic information retrieval, need no API keys, and are cheap enough to run on every commit.

### Golden dataset
`data/eval_dataset.json` holds the ground truth for each question:

```json
{"qid": "q04", "question": "How does hybrid search merge dense and keyword results?",
 "relevant": {"doc_06": 3, "doc_05": 2, "doc_03": 1}}
```

`relevant` maps doc id to a **relevance grade** (3 = perfect, 2 = useful, 1 = partial). Binary metrics treat any grade > 0 as relevant; NDCG uses the grades.

### The metrics

Example used below: retrieved = `[A, B, C, D, E]`, relevant = `{A, C, F}`

| Metric | Formula | Example | What it tells you |
|---|---|---|---|
| **Hit Rate@k** | 1 if any relevant doc in top k | @1 = 1 | Did we find *anything* useful? |
| **Precision@k** | relevant in top k / k | @3 = 2/3 | How much noise reaches the prompt |
| **Recall@k** | relevant in top k / all relevant | @5 = 2/3 (F missed) | Did the LLM get *all* the evidence? |
| **MRR** | mean of 1 / rank of first relevant | 1/1 = 1.0 | Is the best doc at the top? |
| **MAP** | mean of average precision | (1/1 + 2/3) / 3 = 0.556 | Are *all* relevant docs ranked early? |
| **NDCG@k** | DCG / ideal DCG, gain = 2^grade − 1, discount = log2(rank+1) | see tests | How close to a perfect ordering, with grades |

**How to choose k:** use the same k your RAG pipeline actually passes to the LLM. If you put the top 5 chunks into the prompt, Recall@5 is the number that matters.

**Precision vs recall trade-off:** raising k raises recall but lowers precision. More context means a better chance of including the answer, but more noise and higher cost for the LLM.

**MRR vs MAP:** MRR looks only at the *first* relevant doc. MAP rewards ranking *every* relevant doc high. Use MAP or NDCG when an answer needs several documents.

### Project layout

```
data/
  corpus.json              15 short docs about RAG
  eval_dataset.json        12 questions with graded relevant doc ids
src/rag_evals/
  metrics/retrieval.py     all metrics, written from scratch
  retriever.py             tiny BM25 retriever (no dependencies)
  evaluate.py              runs the eval, prints a report + failure analysis
tests/
  test_retrieval_metrics.py  hand-computed checks for each metric
```

### Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

pytest -q                                   # 10 tests
python -m rag_evals.evaluate                # report for k = 1, 3, 5
python -m rag_evals.evaluate --k 1 3 5 10 --out results/bm25.json
```

### Results (BM25 baseline)

```
   k |    Hit |   Prec | Recall |   NDCG
------------------------------------------
   1 |  0.833 |  0.833 |  0.569 |  0.833
   3 |  0.833 |  0.361 |  0.681 |  0.797
   5 |  0.833 |  0.233 |  0.708 |  0.808

MRR@5: 0.833    MAP@5: 0.701

Missed entirely (no relevant doc in top 5):
  q06: How can we tell if an answer is invented instead of grounded?
      got ['doc_01', 'doc_09']  expected ['doc_08']
  q11: What should my test set with expected outputs contain?
      got []  expected ['doc_14']
```

### What the results teach
- **Keyword retrieval fails on paraphrases.** q06 means "hallucination" and q11 means "golden dataset", but neither shares words with the right document. This is the *vocabulary mismatch* problem, and it is why Branch 02 adds dense embeddings.
- **The averages hide the story; the failure list tells it.** Always read which queries fail, not only the averages.
- **Recall@1 is low even when Hit@1 is high.** Questions with several relevant docs can't be fully covered by one result.

### Key takeaways
1. Evaluate retrieval separately from generation.
2. A golden dataset (question → relevant doc ids) is the foundation of every later branch.
3. Choose k to match what your pipeline sends to the LLM.
4. Verify every metric by hand once (see the tests) before trusting it.

**Next: `02-dense-vs-sparse`** reuses these exact metrics to compare BM25, embeddings and hybrid search.
