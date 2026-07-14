from llm_client import estimate_cost
from config import settings


def _rec(tokens_in, tokens_out, model):
    return {"input_tokens": tokens_in, "output_tokens": tokens_out,
            "calls": 1, "total_ms": 100.0, "model": model}


def test_cost_haiku():
    usage = {"supervisor": _rec(1000, 100, "us.anthropic.claude-haiku-4-5-20251001-v1:0")}
    expected = 1000 * settings.price_haiku_in / 1e6 + 100 * settings.price_haiku_out / 1e6
    assert abs(estimate_cost(usage) - expected) < 1e-9


def test_cost_sonnet():
    usage = {"synthesizer": _rec(5000, 2000, "us.anthropic.claude-sonnet-4-5-20250929-v1:0")}
    expected = 5000 * settings.price_sonnet_in / 1e6 + 2000 * settings.price_sonnet_out / 1e6
    assert abs(estimate_cost(usage) - expected) < 1e-9


def test_cost_mixed_models_sum():
    usage = {"supervisor": _rec(1000, 100, "claude-haiku-4-5-20251001"),
             "synthesizer": _rec(1000, 100, "claude-sonnet-4-5-20250929")}
    haiku = 1000 * settings.price_haiku_in / 1e6 + 100 * settings.price_haiku_out / 1e6
    sonnet = 1000 * settings.price_sonnet_in / 1e6 + 100 * settings.price_sonnet_out / 1e6
    assert abs(estimate_cost(usage) - (haiku + sonnet)) < 1e-9


def test_unknown_model_priced_as_sonnet():
    """Conservative default: unknown models billed at the higher tier."""
    usage = {"x": _rec(1_000_000, 0, "")}
    assert estimate_cost(usage) == settings.price_sonnet_in
