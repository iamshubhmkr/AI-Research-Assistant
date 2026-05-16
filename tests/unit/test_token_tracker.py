"""Unit tests for token tracking."""
from llm_client import estimate_cost


def test_estimate_cost_haiku():
    usage = {"supervisor": {"input_tokens": 1000, "output_tokens": 100, "calls": 1, "total_ms": 100}}
    cost = estimate_cost(usage)
    # Haiku: 1000 * 0.25/1M + 100 * 1.25/1M = 0.000375
    assert 0.0001 < cost < 0.001


def test_estimate_cost_sonnet():
    usage = {"synthesizer": {"input_tokens": 5000, "output_tokens": 2000, "calls": 1, "total_ms": 3000}}
    cost = estimate_cost(usage)
    # Sonnet: 5000 * 3/1M + 2000 * 15/1M = 0.045
    assert 0.01 < cost < 0.1


def test_estimate_cost_mixed():
    usage = {
        "supervisor": {"input_tokens": 500, "output_tokens": 50, "calls": 5, "total_ms": 500},
        "synthesizer": {"input_tokens": 4000, "output_tokens": 1500, "calls": 1, "total_ms": 5000},
    }
    cost = estimate_cost(usage)
    assert cost > 0
