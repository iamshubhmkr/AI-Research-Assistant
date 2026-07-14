from evaluation.ragas_eval import RAGASEvaluator, SMOKE_CASES


def test_targets_loaded():
    ev = RAGASEvaluator()
    assert ev.TARGETS["faithfulness"] == 0.85


def test_smoke_cases_valid():
    for case in SMOKE_CASES:
        assert all(k in case for k in ["question", "answer", "contexts", "ground_truth"])


def test_regression_pass():
    ev = RAGASEvaluator()
    assert ev.regression_check({"faithfulness": 0.90}, {"faithfulness": 0.88}) is True


def test_regression_fail():
    ev = RAGASEvaluator()
    assert ev.regression_check({"faithfulness": 0.70}, {"faithfulness": 0.88}) is False
