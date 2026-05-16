"""
Central configuration — all settings loaded from environment variables.
Uses pydantic-settings so .env files are automatically loaded.

Design Decision: Single Settings class is the ONLY place config lives.
  - No magic strings scattered across agents
  - Easy to audit every tunable parameter
  - Environment-specific values via .env files (dev vs prod)
"""
from pydantic_settings import BaseSettings
from typing import Literal


class Settings(BaseSettings):
    # ── LLM Provider ────────────────────────────────────────────
    llm_provider: Literal["anthropic", "bedrock"] = "anthropic"

    # ── Anthropic (direct API) ──────────────────────────────────
    anthropic_api_key: str = ""
    claude_sonnet: str = "claude-sonnet-4-5-20250514"
    claude_haiku: str = "claude-haiku-4-5-20251001"

    # ── AWS / Bedrock ───────────────────────────────────────────
    aws_region: str = "us-east-1"
    aws_profile: str = ""
    bedrock_sonnet: str = "anthropic.claude-sonnet-4-5-20251001"
    bedrock_haiku: str = "anthropic.claude-haiku-4-5-20251001"
    s3_bucket: str = "research-assistant-papers"
    bedrock_kb_id: str = ""

    # ── Redis ────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    cache_ttl_query: int = 86400
    cache_ttl_embedding: int = 604800
    cache_ttl_llm: int = 3600
    semantic_cache_threshold: float = 0.92

    # ── PostgreSQL ───────────────────────────────────────────────
    postgres_url: str = "postgresql://user:pass@localhost/research_db"

    # ── DynamoDB ─────────────────────────────────────────────────
    dynamo_sessions_table: str = "research-sessions"
    dynamo_ragas_table: str = "ragas-runs"

    # ── LangSmith ────────────────────────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_project: str = "research-assistant"
    langchain_api_key: str = ""

    # ── RAGAS targets ────────────────────────────────────────────
    ragas_faithfulness_target: float = 0.85
    ragas_relevancy_target: float = 0.80
    ragas_precision_target: float = 0.75
    ragas_recall_target: float = 0.75
    ragas_correctness_target: float = 0.75
    ragas_regression_tolerance: float = 0.03

    # ── Graph limits ─────────────────────────────────────────────
    max_revision_count: int = 2
    max_react_iterations: int = 8
    max_graph_iterations: int = 20

    # ── Retrieval ────────────────────────────────────────────────
    retrieval_top_k: int = 20
    retrieval_final_k: int = 5
    rrf_k: int = 60
    retrieval_mode: Literal["hybrid", "dense", "sparse", "vectorless"] = "hybrid"

    # ── Token Budget ─────────────────────────────────────────────
    max_input_tokens_per_query: int = 80000
    max_output_tokens_synthesis: int = 4096
    max_paper_chars: int = 12000
    target_context_tokens: int = 4000

    # ── Deployment ───────────────────────────────────────────────
    deployment_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
