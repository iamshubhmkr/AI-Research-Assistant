"""
RAGAS evaluation + regression gate + CLI — judged by BEDROCK, not OpenAI.

Design decision (v3.1): RAGAS needs a judge LLM; the default is OpenAI, which
contradicts this project's "all spend on the AWS bill" rule. We inject
Claude Sonnet (judge) and Titan (embeddings) via langchain-aws, so offline
evaluation bills to the same account as the pipeline.

Usage:
  python -m evaluation.ragas_eval --quick                 # built-in smoke cases
  python -m evaluation.ragas_eval --cases cases.json      # real pipeline output
  python -m evaluation.ragas_eval --quick --baseline scores.json   # CI gate
Cases without ground_truth are scored on faithfulness + relevancy only.
"""
import json
import sys
import time
import argparse
from ragas import evaluate
from ragas.metrics import (faithfulness, answer_relevancy, context_precision,
                           context_recall, answer_correctness)
from datasets import Dataset
from config import settings

# metrics that require a human-written ground_truth reference
_NEEDS_GROUND_TRUTH = {"context_precision", "context_recall", "answer_correctness"}


def bedrock_judge():
    """Sonnet as judge + Titan as embedder — same models, same AWS bill."""
    from langchain_aws import ChatBedrockConverse, BedrockEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    llm = ChatBedrockConverse(model=settings.bedrock_sonnet,
                              region_name=settings.aws_region, temperature=0.0)
    emb = BedrockEmbeddings(model_id=settings.bedrock_embed_model,
                            region_name=settings.aws_region)
    return LangchainLLMWrapper(llm), LangchainEmbeddingsWrapper(emb)


class RAGASEvaluator:
    TARGETS = {
        "faithfulness": settings.ragas_faithfulness_target,
        "answer_relevancy": settings.ragas_relevancy_target,
        "context_precision": settings.ragas_precision_target,
        "context_recall": settings.ragas_recall_target,
        "answer_correctness": settings.ragas_correctness_target,
    }
    METRICS = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
        "answer_correctness": answer_correctness,
    }

    def evaluate_pipeline(self, test_cases, label="manual"):
        names = self._usable_metrics(test_cases)
        llm, emb = bedrock_judge()
        ds = Dataset.from_list(test_cases)
        start = time.time()
        results = evaluate(ds, metrics=[self.METRICS[n] for n in names],
                           llm=llm, embeddings=emb)
        scores = {n: round(float(results[n]), 4) for n in names}
        self._report(scores, time.time() - start, len(test_cases))
        return scores

    def _usable_metrics(self, cases):
        names = list(self.METRICS)
        if not all(c.get("ground_truth") for c in cases):
            names = [n for n in names if n not in _NEEDS_GROUND_TRUTH]
            print(f"note: cases lack ground_truth — scoring {', '.join(names)} only")
        return names

    def regression_check(self, new, baseline, tol=None):
        tol = tol or settings.ragas_regression_tolerance
        fails = [f"{m}: {s:.3f} < baseline {baseline[m]:.3f}"
                 for m, s in new.items() if m in baseline and s < baseline[m] - tol]
        if fails:
            print("\nREGRESSION — deployment blocked:")
            for f in fails:
                print(" ", f)
            return False
        print("Quality gates passed — safe to deploy")
        return True

    def _report(self, scores, elapsed, n):
        print(f"\nRAGAS REPORT ({n} cases, {elapsed:.1f}s, judge={settings.bedrock_sonnet})")
        for m, s in scores.items():
            flag = "OK " if s >= self.TARGETS[m] else "WARN"
            print(f"  [{flag}] {m:<22} {s:.3f} (target {self.TARGETS[m]})")


SMOKE_CASES = [
    {"question": "What are the main limitations of RAG for multi-hop reasoning?",
     "answer": "RAG struggles with multi-hop reasoning due to single-step retrieval insufficiency, cross-document coreference failure, and reasoning chain collapse during synthesis.",
     "contexts": ["We identify three primary failure modes in retrieval-augmented generation for compositional question answering: single-step retrieval insufficiency, cross-document coreference failure, and intermediate reasoning chain collapse."],
     "ground_truth": "RAG has three main limitations for multi-hop reasoning: single-step retrieval insufficiency, cross-document coreference failure, and reasoning chain collapse."},
    {"question": "How does HyDE improve retrieval?",
     "answer": "HyDE generates a hypothetical academic answer and embeds it instead of the raw query, bridging the vocabulary gap and improving recall.",
     "contexts": ["Hypothetical Document Embeddings (HyDE) address vocabulary mismatch by generating an idealized answer paragraph in the target domain's language before computing retrieval embeddings."],
     "ground_truth": "HyDE embeds a hypothetical academic answer rather than the raw query, bridging the vocabulary gap between user phrasing and paper language."},
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true", help="run built-in smoke cases")
    p.add_argument("--cases", type=str, help="JSON file of collected pipeline cases")
    p.add_argument("--baseline", type=str)
    p.add_argument("--output", type=str, default="ragas_scores.json")
    args = p.parse_args()

    if args.cases:
        with open(args.cases) as f:
            cases = json.load(f)
    else:
        cases = SMOKE_CASES

    ev = RAGASEvaluator()
    scores = ev.evaluate_pipeline(cases, label="cli")
    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        sys.exit(0 if ev.regression_check(scores, baseline) else 1)


if __name__ == "__main__":
    main()
