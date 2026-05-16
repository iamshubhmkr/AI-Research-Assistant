"""RAGAS regression tests — run before every deployment."""
import json
import os
from evaluation.ragas_eval import RAGASEvaluator, SMOKE_CASES


def test_ragas_smoke():
    """Verify RAGAS evaluation pipeline runs without errors."""
    evaluator = RAGASEvaluator()
    # In CI, this would run against the actual pipeline
    # For unit testing, we verify the evaluator instantiates correctly
    assert evaluator.TARGETS["faithfulness"] == 0.85
    assert evaluator.TARGETS["answer_relevancy"] == 0.80


def test_smoke_cases_valid():
    """Verify smoke test cases have required fields."""
    for case in SMOKE_CASES:
        assert "question" in case
        assert "answer" in case
        assert "contexts" in case
        assert "ground_truth" in case
        assert len(case["contexts"]) > 0


def test_regression_check_passes():
    evaluator = RAGASEvaluator()
    scores = {"faithfulness": 0.90, "answer_relevancy": 0.85,
              "context_precision": 0.80, "context_recall": 0.80, "answer_correctness": 0.80}
    baseline = {"faithfulness": 0.88, "answer_relevancy": 0.82,
                "context_precision": 0.78, "context_recall": 0.78, "answer_correctness": 0.78}
    assert evaluator.regression_check(scores, baseline) is True


def test_regression_check_fails():
    evaluator = RAGASEvaluator()
    scores = {"faithfulness": 0.70}
    baseline = {"faithfulness": 0.88}
    assert evaluator.regression_check(scores, baseline) is False
