---
name: run-evals
description: Run and interpret RAGAS evaluations, set baselines, and gate changes. Use when the user mentions evaluation, RAGAS, quality check, regression, or asks "did my change hurt quality?"
---

# Running Evaluations

## Quick smoke (after any prompt/retrieval change)
```bash
make eval                      # = python -m evaluation.ragas_eval --quick
```
Reads SMOKE_CASES, prints the 5 metrics vs targets.

## Setting a baseline
```bash
python -m evaluation.ragas_eval --quick --output baseline.json
```
Commit baseline.json after a known-good run.

## Regression gate (what CI does)
```bash
python -m evaluation.ragas_eval --quick --baseline baseline.json
```
Exit code 1 if ANY metric dropped > 0.03 → CI blocks deploy.

## Interpreting failures
| Metric drop | Usual cause | Look at |
|---|---|---|
| faithfulness | synthesizer prompt change, weaker critic | agents/synthesizer.py, agents/critic.py |
| context_recall | chunking/HyDE/embedding change | rag/chunker.py, rag/hyde.py |
| context_precision | retrieval too broad, reranker off | rag/retriever.py top_k/final_k |
| answer_relevancy | CoT step 1 weakened | COT_SYSTEM step 1 |

## Golden dataset growth
High-quality production answers (faithfulness > 0.92) auto-append via
evaluation/golden_dataset.maybe_collect_from_production. Periodically
human-review entries where human_reviewed=False.
