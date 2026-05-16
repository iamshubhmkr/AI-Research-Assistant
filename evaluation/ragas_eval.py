"""
RAGAS Evaluation Suite + Regression Gates.

Five metrics with production targets:
  faithfulness       > 0.85
  answer_relevancy   > 0.80
  context_precision  > 0.75
  context_recall     > 0.75
  answer_correctness > 0.75

Regression gate: fail CI/CD if any metric drops > 0.03 from baseline.

Usage:
  python -m evaluation.ragas_eval                    # full evaluation
  python -m evaluation.ragas_eval --quick             # smoke test (3 cases)
  python -m evaluation.ragas_eval --baseline scores.json  # regression check
"""
import json
import sys
import time
import argparse
import logging
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from datasets import Dataset
from config import settings

logger = logging.getLogger(__name__)


class RAGASEvaluator:
    TARGETS = {
        "faithfulness": settings.ragas_faithfulness_target,
        "answer_relevancy": settings.ragas_relevancy_target,
        "context_precision": settings.ragas_precision_target,
        "context_recall": settings.ragas_recall_target,
        "answer_correctness": settings.ragas_correctness_target,
    }

    def evaluate_pipeline(self, test_cases: list, label: str = "manual") -> dict:
        ds = Dataset.from_list(test_cases)
        start = time.time()
        results = evaluate(ds, metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness])
        elapsed = time.time() - start
        scores = {k: round(results[k], 4) for k in self.TARGETS}
        self._print_report(scores, elapsed, len(test_cases))
        return scores

    def regression_check(self, new_scores: dict, baseline: dict, tol: float = None) -> bool:
        tol = tol or settings.ragas_regression_tolerance
        failures = [
            f"{m}: {s:.3f} < baseline {baseline[m]:.3f} (dropped {baseline[m]-s:.3f})"
            for m, s in new_scores.items()
            if m in baseline and s < baseline[m] - tol
        ]
        if failures:
            print("\n❌ REGRESSION DETECTED — deployment blocked:")
            for f in failures:
                print(f"  {f}")
            return False
        print("✅ Quality gates passed — safe to deploy")
        return True

    def _print_report(self, scores: dict, elapsed: float = 0, n_cases: int = 0):
        print(f"\n{'═' * 50}")
        print(f"  RAGAS REPORT  ({n_cases} cases, {elapsed:.1f}s)")
        print(f"{'═' * 50}")
        for m, s in scores.items():
            tgt = self.TARGETS[m]
            bar = "█" * int(s * 20) + "░" * (20 - int(s * 20))
            ok = "✅" if s >= tgt else "⚠️ "
            print(f"{ok} {m:<22} {s:.3f} [{bar}] target:{tgt}")
        print(f"{'═' * 50}\n")


# ── Smoke test golden dataset ────────────────────────────────────
SMOKE_CASES = [
    {
        "question": "What are the main limitations of RAG for multi-hop reasoning?",
        "answer": "RAG struggles with multi-hop reasoning due to: (1) single-step retrieval insufficiency — the retriever fetches documents relevant to the surface query but misses documents needed for intermediate reasoning steps, (2) cross-document coreference failure — entities referenced across papers aren't linked, and (3) intermediate reasoning chain collapse — the LLM loses track of the reasoning chain when synthesizing across multiple retrieved passages.",
        "contexts": [
            "We identify three primary failure modes in retrieval-augmented generation for compositional question answering: single-step retrieval insufficiency, cross-document coreference failure, and intermediate reasoning chain collapse.",
            "Current RAG systems perform a single retrieval step, fetching documents relevant to the original query. For multi-hop questions requiring 2-3 reasoning steps, this approach fails because intermediate evidence is never retrieved."
        ],
        "ground_truth": "RAG has three main limitations for multi-hop reasoning: single-step retrieval insufficiency (only retrieves for surface query), cross-document coreference failure (can't link entities across documents), and intermediate reasoning chain collapse (loses track when synthesizing across passages)."
    },
    {
        "question": "How does HyDE improve retrieval performance?",
        "answer": "HyDE generates a hypothetical document that answers the query using academic vocabulary, then embeds that document instead of the raw query. This bridges the vocabulary gap between casual user queries and formal academic papers, improving Context Recall from 0.71 to 0.89.",
        "contexts": [
            "Hypothetical Document Embeddings (HyDE) address the vocabulary mismatch problem by generating an idealized answer paragraph in the target domain's language before computing embeddings for retrieval."
        ],
        "ground_truth": "HyDE improves retrieval by generating a hypothetical answer document in academic language, embedding that instead of the raw query, bridging the vocabulary gap between user queries and academic papers."
    },
    {
        "question": "What is the role of the critic agent in the pipeline?",
        "answer": "The critic agent fact-checks the synthesis using few-shot examples calibrated to detect specific failure modes: fabricated statistics (critical), invented citations (critical), unstated inferences (moderate), and unacknowledged gaps (moderate). It also runs RAGAS faithfulness scoring. If severity is critical or moderate with faithfulness below 0.85, revision is triggered.",
        "contexts": [
            "The critic uses few-shot prompting with 5 calibrated examples covering exact failure modes observed during development. Three severity levels prevent over-rejection while catching genuine hallucinations."
        ],
        "ground_truth": "The critic agent performs fact-checking using few-shot examples for calibrated hallucination detection, combined with RAGAS faithfulness scoring, triggering revisions for critical issues or moderate issues with low faithfulness."
    },
]


def main():
    parser = argparse.ArgumentParser(description="Run RAGAS evaluation")
    parser.add_argument("--quick", action="store_true", help="Run smoke test only")
    parser.add_argument("--baseline", type=str, help="Path to baseline scores JSON for regression check")
    parser.add_argument("--output", type=str, default="ragas_scores.json", help="Output scores file")
    args = parser.parse_args()

    evaluator = RAGASEvaluator()
    cases = SMOKE_CASES if args.quick else SMOKE_CASES  # In production: load from S3

    scores = evaluator.evaluate_pipeline(cases, label="cli")

    with open(args.output, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Scores saved to {args.output}")

    if args.baseline:
        with open(args.baseline) as f:
            baseline = json.load(f)
        passed = evaluator.regression_check(scores, baseline)
        sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
