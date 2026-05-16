"""
LLM Client Abstraction — Switch between Anthropic API and AWS Bedrock.

Design Decision: Why abstract the LLM client?
  - Development uses Anthropic API directly (simpler, faster iteration)
  - Production uses AWS Bedrock (IAM auth, VPC isolation, Guardrails)
  - Token tracking is centralized here, not scattered across agents

Interview talking point:
  "I abstracted the LLM client so agents don't know whether they're
   calling Anthropic directly or through Bedrock. In dev I use the
   direct API. In prod I flip one env var to Bedrock for IAM auth
   and Guardrails. Zero agent code changes."
"""
import json
import time
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)

_token_usage: dict[str, dict] = {}


def _get_anthropic_client():
    from anthropic import Anthropic
    return Anthropic(api_key=settings.anthropic_api_key)


def _get_bedrock_client():
    import boto3
    kwargs = {"region_name": settings.aws_region}
    if settings.aws_profile:
        kwargs["profile_name"] = settings.aws_profile
    return boto3.Session(**kwargs).client("bedrock-runtime")


def _resolve_model(model_hint: str) -> str:
    if settings.llm_provider == "bedrock":
        if "sonnet" in model_hint.lower():
            return settings.bedrock_sonnet
        elif "haiku" in model_hint.lower():
            return settings.bedrock_haiku
    return model_hint


def call_llm(
    model: str,
    messages: list,
    system: str = "",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    tools: Optional[list] = None,
    agent_name: str = "unknown",
) -> dict:
    """
    Unified LLM call. Returns: {content, stop_reason, usage}.
    Tracks tokens per agent_name for cost reporting.
    """
    resolved_model = _resolve_model(model)
    start_ms = time.time() * 1000

    if settings.llm_provider == "bedrock":
        result = _call_bedrock(resolved_model, messages, system, max_tokens, temperature, tools)
    else:
        result = _call_anthropic(resolved_model, messages, system, max_tokens, temperature, tools)

    elapsed_ms = time.time() * 1000 - start_ms
    usage = result.get("usage", {})

    if agent_name not in _token_usage:
        _token_usage[agent_name] = {"input_tokens": 0, "output_tokens": 0, "calls": 0, "total_ms": 0.0}
    _token_usage[agent_name]["input_tokens"] += usage.get("input_tokens", 0)
    _token_usage[agent_name]["output_tokens"] += usage.get("output_tokens", 0)
    _token_usage[agent_name]["calls"] += 1
    _token_usage[agent_name]["total_ms"] += elapsed_ms

    logger.info(f"[{agent_name}] {resolved_model} in={usage.get('input_tokens',0)} out={usage.get('output_tokens',0)} {elapsed_ms:.0f}ms")
    return result


def _call_anthropic(model, messages, system, max_tokens, temperature, tools):
    client = _get_anthropic_client()
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature}
    if system:
        kwargs["system"] = system
    if tools:
        kwargs["tools"] = tools
    resp = client.messages.create(**kwargs)
    return {
        "content": resp.content,
        "stop_reason": resp.stop_reason,
        "usage": {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens,
                  "cache_read_tokens": getattr(resp.usage, "cache_read_input_tokens", 0)},
    }


def _call_bedrock(model, messages, system, max_tokens, temperature, tools):
    client = _get_bedrock_client()
    body = {"anthropic_version": "bedrock-2023-05-31", "max_tokens": max_tokens,
            "temperature": temperature, "messages": messages}
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


def get_token_usage() -> dict:
    return dict(_token_usage)


def reset_token_usage():
    global _token_usage
    _token_usage = {}


def estimate_cost(usage: dict) -> float:
    """Estimate USD cost. Haiku: $0.25/$1.25 per M. Sonnet: $3/$15 per M."""
    total = 0.0
    haiku_agents = {"supervisor", "query_expansion", "compression", "raptor_summary"}
    for agent, data in usage.items():
        if agent in haiku_agents:
            total += data["input_tokens"] * 0.25 / 1e6 + data["output_tokens"] * 1.25 / 1e6
        else:
            total += data["input_tokens"] * 3.0 / 1e6 + data["output_tokens"] * 15.0 / 1e6
    return round(total, 6)
