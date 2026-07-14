"""
LLM Client v3.1 — Bedrock-primary abstraction with production resilience.

Design decisions:
  - ONE choke point: every LLM call passes through call_llm(), which gives
    every agent retries, circuit breaking, optional fallback, and token/cost
    tracking for free.
  - Bedrock is the primary provider (pay through AWS, auth via IAM). The
    direct Anthropic API is an OPT-IN fallback (enable_provider_fallback)
    because it needs its own key and its own bill.
  - Circuit breaker state is per-provider (a Bedrock outage must not poison
    the Anthropic circuit and vice versa).
  - Cost is estimated from the MODEL actually used per agent, with prices in
    config.py — no magic numbers, no duplicated routing knowledge.
"""
import json
import time
import random
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

_token_usage: dict[str, dict] = {}

# ── Circuit breaker state (module-level, per-provider) ──────────
_circuits: dict[str, dict] = {
    "anthropic": {"failures": 0, "opened_at": 0.0},
    "bedrock": {"failures": 0, "opened_at": 0.0},
}


def _circuit_open(provider: str) -> bool:
    """Is this provider's circuit currently open (provider considered down)?"""
    c = _circuits[provider]
    if c["failures"] < settings.circuit_failure_threshold:
        return False
    if time.time() - c["opened_at"] > settings.circuit_cooldown_s:
        # Cooldown elapsed -> half-open: allow one trial call
        c["failures"] = settings.circuit_failure_threshold - 1
        return False
    return True


def _record_failure(provider: str):
    c = _circuits[provider]
    c["failures"] += 1
    if c["failures"] == settings.circuit_failure_threshold:
        c["opened_at"] = time.time()
        logger.error(f"Circuit breaker OPEN for {provider}")


def _record_success(provider: str):
    _circuits[provider]["failures"] = 0


def _get_anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=settings.anthropic_api_key)


def _get_bedrock_client():
    import boto3
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_profile:
        kwargs["profile_name"] = settings.aws_profile
    return boto3.Session(**kwargs).client("bedrock-runtime")


def _resolve_model(model_hint: str, provider: str) -> str:
    """Map a model hint to the provider-specific ID ('sonnet'/'haiku' aware)."""
    hint = model_hint.lower()
    if provider == "bedrock":
        if "sonnet" in hint:
            return settings.bedrock_sonnet
        if "haiku" in hint:
            return settings.bedrock_haiku
    else:
        if "sonnet" in hint:
            return settings.claude_sonnet
        if "haiku" in hint:
            return settings.claude_haiku
    return model_hint


def _to_plain(obj):
    """Recursively convert SDK blocks / SimpleNamespace to JSON-safe dicts.

    Needed because assistant turns from a previous response carry content
    blocks as objects; Bedrock's invoke_model body must be pure JSON.
    """
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_plain(v) for v in obj]
    if hasattr(obj, "model_dump"):           # anthropic SDK pydantic blocks
        return obj.model_dump()
    if hasattr(obj, "__dict__") and not isinstance(obj, (str, int, float, bool)):
        return {k: _to_plain(v) for k, v in vars(obj).items()}
    return obj


def call_llm(model: str, messages: list, system: str = "", max_tokens: int = 1024,
             temperature: float = 0.0, tools: Optional[list] = None,
             agent_name: str = "unknown") -> dict:
    """
    Unified call with retry -> circuit breaker -> optional fallback.
    Returns {content, stop_reason, usage}.
    """
    provider = settings.llm_provider
    if _circuit_open(provider):
        if settings.enable_provider_fallback:
            provider = "anthropic" if provider == "bedrock" else "bedrock"
            logger.warning(f"Circuit open: falling back to {provider}")
        else:
            raise RuntimeError(f"{settings.llm_provider} circuit open and fallback disabled")

    start_ms = time.time() * 1000
    last_err = None

    for attempt in range(settings.llm_max_retries):
        try:
            result = _dispatch(provider, model, messages, system, max_tokens, temperature, tools)
            _record_success(provider)
            _track(agent_name, result, _resolve_model(model, provider), start_ms)
            return result
        except Exception as e:
            last_err = e
            _record_failure(provider)
            if attempt < settings.llm_max_retries - 1:
                # exponential backoff with jitter: 1s, 2s (+/- 0-0.5s)
                wait = settings.llm_backoff_base_s * (2 ** attempt) + random.uniform(0, 0.5)
                logger.warning(f"[{agent_name}] attempt {attempt+1} failed ({e}); retrying in {wait:.1f}s")
                time.sleep(wait)
            else:
                logger.warning(f"[{agent_name}] attempt {attempt+1} failed ({e}); retries exhausted")

    # Retries exhausted: try the OTHER provider once before giving up (opt-in)
    if settings.enable_provider_fallback:
        alt = "anthropic" if provider == "bedrock" else "bedrock"
        logger.error(f"[{agent_name}] {provider} exhausted; final fallback to {alt}")
        result = _dispatch(alt, model, messages, system, max_tokens, temperature, tools)
        _track(agent_name, result, _resolve_model(model, alt), start_ms)
        return result
    raise last_err


def _dispatch(provider, model, messages, system, max_tokens, temperature, tools):
    resolved = _resolve_model(model, provider)
    if provider == "bedrock":
        return _call_bedrock(resolved, messages, system, max_tokens, temperature, tools)
    return _call_anthropic(resolved, messages, system, max_tokens, temperature, tools)


def _call_anthropic(model, messages, system, max_tokens, temperature, tools):
    client = _get_anthropic_client()
    kwargs = {"model": model, "messages": _to_plain(messages),
              "max_tokens": max_tokens, "temperature": temperature}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    resp = client.messages.create(**kwargs)
    return {"content": resp.content, "stop_reason": resp.stop_reason,
            "usage": {"input_tokens": resp.usage.input_tokens,
                      "output_tokens": resp.usage.output_tokens,
                      "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0) or 0}}


def _call_bedrock(model, messages, system, max_tokens, temperature, tools):
    client = _get_bedrock_client()
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
            "temperature": temperature, "messages": _to_plain(messages)}
    if system:
        body["system"] = system
    if tools:
        body["tools"] = tools
    response = client.invoke_model(modelId=model, contentType="application/json",
                                   accept="application/json", body=json.dumps(body))
    result = json.loads(response["body"].read())
    from types import SimpleNamespace
    content = [SimpleNamespace(**b) for b in result.get("content", [])]
    return {"content": content, "stop_reason": result.get("stop_reason", "end_turn"),
            "usage": result.get("usage", {})}


def _track(agent_name, result, model, start_ms):
    elapsed = time.time() * 1000 - start_ms
    u = result.get("usage", {})
    rec = _token_usage.setdefault(agent_name, {"input_tokens": 0, "output_tokens": 0,
                                               "calls": 0, "total_ms": 0.0, "model": model})
    rec["input_tokens"] += u.get("input_tokens", 0)
    rec["output_tokens"] += u.get("output_tokens", 0)
    rec["calls"] += 1
    rec["total_ms"] += elapsed
    rec["model"] = model
    logger.info(f"[{agent_name}] {model} in={u.get('input_tokens',0)} out={u.get('output_tokens',0)} {elapsed:.0f}ms")


def get_token_usage() -> dict:
    return dict(_token_usage)


def reset_token_usage():
    global _token_usage
    _token_usage = {}


def estimate_cost(usage: dict) -> float:
    """Price by the model each agent actually used (prices in config.py)."""
    total = 0.0
    for _agent, d in usage.items():
        if "haiku" in d.get("model", "").lower():
            p_in, p_out = settings.price_haiku_in, settings.price_haiku_out
        else:
            p_in, p_out = settings.price_sonnet_in, settings.price_sonnet_out
        total += d["input_tokens"] * p_in / 1e6 + d["output_tokens"] * p_out / 1e6
    return round(total, 6)
