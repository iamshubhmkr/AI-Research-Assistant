# AI Research Assistant v3 — Production-Grade Multi-Agent Pipeline

> LangGraph · Claude · AWS Bedrock · RAGAS · Redis · ChromaDB · RAPTOR · HyDE · MCP

Six agents (supervisor, fetcher, extractor, retriever, synthesizer, critic)
research any question across academic papers — with hybrid retrieval, two
human-in-the-loop checkpoints, 4-layer caching, RAGAS quality gates, and full
cost tracking.

## New in v3
- **Resilient LLM client**: retry w/ exponential backoff + jitter, circuit
  breaker, automatic Anthropic↔Bedrock fallback (llm_client.py)
- **Document Router**: PDF / DOCX / HTML / CSV / TXT ingestion — one parser
  file, zero downstream changes (rag/document_router.py)
- **Parallel Tree-of-Thoughts**: 3 branches concurrently (~8.4s → ~3s)
- **Parallel contextual compression** in the retriever
- **SSE streaming endpoint**: watch agents complete in real time
  (POST /research/start_stream)
- **Complete `.claude/` folder**: CLAUDE.md, settings.json, mcp.json, rules,
  skills, subagents, slash-commands — Claude Code works inside guardrails
- **CLAUDE_CREATE_BIBLE.md**: the exact prompts + model strategy to recreate
  this setup for ANY new project
- **Makefile**: `make up / test / eval / run / ui`

## Quickstart (Bedrock is the primary provider — all spend on your AWS bill)
```bash
make setup            # venv + deps
make up               # Redis + PostgreSQL via Docker (free, local)
cp .env.development .env
aws configure         # keys + us-east-1
# AWS Console → Bedrock → Model access → enable:
#   Claude Sonnet 4.5, Claude Haiku 4.5, Titan Text Embeddings V2
make test             # unit tests
make lint             # ruff
python scripts/bedrock_smoke.py   # 1 tiny call per model (<$0.01)
python scripts/e2e_local.py       # full pipeline E2E with stubbed Bedrock
make run              # API on :8000     (new terminal)
make ui               # Streamlit UI     (new terminal)
```

Embeddings run on **Bedrock Titan v2** (no local torch models); the direct
Anthropic API is an opt-in fallback (`ENABLE_PROVIDER_FALLBACK=true` +
`ANTHROPIC_API_KEY`).

## Structure
```
.claude/                 ← Claude Code constitution (memory, rules, skills,
                            subagents, commands, permissions, MCP config)
CLAUDE_CREATE_BIBLE.md   ← how to recreate .claude/ for any project
config.py                ← every tunable (pydantic-settings)
llm_client.py            ← LLM abstraction + retry/circuit-breaker/fallback
agents/                  ← state, supervisor, fetcher, extractor,
                            synthesizer, critic, graph
rag/                     ← document_router, chunker, hyde, raptor,
                            retriever (8-stage hybrid + vectorless),
                            vector_store, embedder
cache/                   ← L1 semantic, L2 embedding, L3 LLM (shared Redis)
persistence/             ← postgres (checkpoints), dynamo (sessions), s3
evaluation/              ← ragas_eval (+CLI), golden_dataset, token_tracker
mcp/                     ← arXiv MCP server
api/                     ← FastAPI: HITL endpoints + SSE streaming
ui/                      ← Streamlit: 2-checkpoint review flow
infrastructure/          ← AWS CDK stack
tests/                   ← unit / ragas / smoke
```

## Claude Code workflow
```bash
cd research-assistant-v3 && claude
# Claude reads .claude/CLAUDE.md automatically. Try:
> /eval
> Use the code-reviewer agent on my staged diff
> Add a citation-formatter agent        # triggers the add-agent skill
```
