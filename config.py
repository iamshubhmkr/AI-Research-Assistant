"""
Central configuration — every tunable lives here, loaded from environment.
v3.1: Bedrock is the PRIMARY provider (Sonnet/Haiku via inference profiles,
Titan v2 for embeddings) so all spend lands on the AWS bill. The direct
Anthropic API is an opt-in fallback only (requires ANTHROPIC_API_KEY).
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # ── LLM Provider ─────────────────────────────────────
    llm_provider: Literal["anthropic", "bedrock"] = "bedrock"
    # Fallback to the OTHER provider needs its credentials too — off by
    # default so a Bedrock-only deployment never silently bills Anthropic.
    enable_provider_fallback: bool = False

    # ── Anthropic (fallback only) ────────────────────────
    anthropic_api_key: str = ""
    claude_sonnet: str = "claude-sonnet-4-5-20250929"
    claude_haiku: str = "claude-haiku-4-5-20251001"

    # ── AWS / Bedrock ────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_profile: str = ""
    # Claude 4.5 on Bedrock is invoked via cross-region inference profiles.
    bedrock_sonnet: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    bedrock_haiku: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    bedrock_embed_model: str = "amazon.titan-embed-text-v2:0"
    embed_dimensions: int = 1024            # Titan v2 supports 256/512/1024
    s3_bucket: str = "research-assistant-papers"

    # ── Pricing (USD per 1M tokens) ──────────────────────
    # Same list price on Anthropic API and Bedrock on-demand.
    price_sonnet_in: float = 3.0
    price_sonnet_out: float = 15.0
    price_haiku_in: float = 1.0
    price_haiku_out: float = 5.0

    # ── Resilience (v3) ──────────────────────────────────
    llm_max_retries: int = 3
    llm_backoff_base_s: float = 1.0         # 1s, 2s, 4s + jitter
    circuit_failure_threshold: int = 5      # failures before circuit opens
    circuit_cooldown_s: int = 30            # how long circuit stays open

    # ── Redis ────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_ttl_query: int = 86400
    cache_ttl_embedding: int = 604800
    cache_ttl_llm: int = 3600
    semantic_cache_threshold: float = 0.92

    # ── PostgreSQL / DynamoDB ────────────────────────────
    postgres_url: str = "postgresql://user:pass@localhost/research_db"
    dynamo_sessions_table: str = "research-sessions"
    enable_dynamo: bool = False             # off for local testing; on in prod

    # ── LangSmith ────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_project: str = "research-assistant"
    langchain_api_key: str = ""

    # ── RAGAS targets ────────────────────────────────────
    ragas_faithfulness_target: float = 0.85
    ragas_relevancy_target: float = 0.80
    ragas_precision_target: float = 0.75
    ragas_recall_target: float = 0.75
    ragas_correctness_target: float = 0.75
    ragas_regression_tolerance: float = 0.03

    # ── Graph limits ─────────────────────────────────────
    max_revision_count: int = 2
    max_react_iterations: int = 8
    max_graph_iterations: int = 20

    # ── Retrieval ────────────────────────────────────────
    retrieval_top_k: int = 20
    retrieval_final_k: int = 5
    rrf_k: int = 60
    retrieval_mode: Literal["hybrid", "dense", "sparse", "vectorless"] = "hybrid"
    enable_cross_encoder: bool = False      # optional local reranker (needs sentence-transformers)

    # ── Token budget ─────────────────────────────────────
    max_input_tokens_per_query: int = 80000
    max_output_tokens_synthesis: int = 4096
    max_paper_chars: int = 12000

    # ── Document router (v3) ─────────────────────────────
    supported_formats: list[str] = ["pdf", "docx", "html", "txt", "md", "csv"]

    # ── Deployment ───────────────────────────────────────
    deployment_env: str = "development"
    log_level: str = "INFO"

settings = Settings()
