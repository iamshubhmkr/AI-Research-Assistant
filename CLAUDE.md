# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate virtual environment (always required first)
source .venv/bin/activate

# Infrastructure (Redis + PostgreSQL — required before running the app)
docker-compose up -d
docker-compose ps   # verify both services show "Up"

# Run API server
uvicorn api.main:app --reload --port 8000

# Run UI
streamlit run ui/app.py

# Health check
curl http://localhost:8000/health

# Tests
python -m pytest tests/unit/ -v            # unit tests (run before any agent change)
python -m pytest tests/unit/test_chunker.py -v   # single test file
python -m pytest tests/smoke/ -v          # smoke tests (requires running API)

# RAGAS evaluation
python -m evaluation.ragas_eval --quick   # 3-case smoke test
python -m evaluation.ragas_eval           # full evaluation
python -m evaluation.ragas_eval --baseline scores.json  # regression check vs baseline
```

## Environment Setup

Copy `.env.production` as a reference — for local dev create `.env` with:
```
ANTHROPIC_API_KEY=sk-...
LLM_PROVIDER=anthropic          # use "bedrock" for AWS production
POSTGRES_URL=postgresql://user:pass@localhost/research_db
REDIS_URL=redis://localhost:6379
LANGCHAIN_API_KEY=...           # optional, for LangSmith tracing
```

All settings live in `config.py` (`Settings` class). Every tunable parameter is there — never hardcode values in agents.

## Architecture

### LangGraph Pipeline

```
supervisor → fetcher → extractor → retriever → [HITL-1] → synthesizer → critic → [HITL-2] → END
                ↑_______________________________________________|
```

`agents/graph.py` assembles the graph. The supervisor runs after **every** node and decides what runs next using `state["next"]`. Two human-in-the-loop interrupts:
- `interrupt_before=["synthesizer"]` — human reviews and optionally removes retrieved chunks
- `interrupt_after=["critic"]` — human can override the critic's revision request

Graph state persists in PostgreSQL (`PostgresSaver`), so any instance can resume any session by `thread_id`.

### API Flow (3-step HITL)

1. `POST /research/start` — runs pipeline until HITL-1, returns `retrieved_chunks` + `session_id`
2. `POST /research/approve_chunks` — human removes bad chunks, pipeline resumes until HITL-2
3. `POST /research/resolve_critique` — human approves or overrides critique, returns `final_answer`

### State Contract

`agents/state.py` is the shared contract between all agents. Fields using `Annotated[list, operator.add]` (e.g. `raw_papers`, `extracted_facts`) are append-only — multiple agents can write safely. **Never change `state.py` without updating every agent that reads or writes those fields.**

### LLM Client

All LLM calls must go through `llm_client.call_llm()` — never instantiate Anthropic/Bedrock clients directly in agents. This centralizes token tracking and allows flipping between providers via `LLM_PROVIDER` env var. The supervisor and cheap classification tasks use Haiku; synthesizer and critic use Sonnet.

### RAG Stack

- **Chunker** (`rag/chunker.py`): semantic chunking (level 0) feeds RAPTOR
- **RAPTOR** (`rag/raptor.py`): hierarchical summarization — levels 0 (leaf chunks) → 1 → 2
- **HyDE** (`rag/hyde.py`): generates a hypothetical answer to embed instead of the raw query
- **Retriever** (`rag/retriever.py`): dense (HyDE + ChromaDB) + sparse (BM25) → RRF fusion → cross-encoder rerank → contextual compression
- `retrieval_mode=vectorless` skips embeddings entirely and has the LLM score chunk relevance directly

### 4-Layer Cache

| Layer | File | What it caches |
|---|---|---|
| L1 Semantic | `cache/semantic_cache.py` | Similar queries (cosine > 0.92) → full answer |
| L2 Embedding | `cache/embedding_cache.py` | Text → embedding vectors |
| L3 LLM | `cache/llm_cache.py` | Prompt → LLM response |
| L4 Redis | `cache/redis_client.py` | Backing store for L1/L3 |

### Persistence

- **PostgreSQL**: LangGraph checkpoint state per `thread_id` (via `langgraph-checkpoint-postgres`)
- **DynamoDB** (`persistence/dynamo.py`): session metadata and query history
- **S3** (`persistence/s3.py`): raw paper PDFs

## Critical Rules (from RULES.md)

1. Any change to `agents/state.py` requires updating **all** agents that touch those fields.
2. Run `python -m pytest tests/unit/ -v` before committing agent changes.
3. All LLM calls in agents must use `llm_client.call_llm()` for token tracking.
4. New agents must be registered in `agents/graph.py`.
5. RAGAS scores must stay above the targets defined in `config.py` — run `ragas_eval` before merging.
