# AI Research Assistant — Production-Grade Multi-Agent Pipeline v2

> LangGraph · Claude Sonnet 4.5 · AWS Bedrock · RAGAS · Redis · ChromaDB · RAPTOR · HyDE

A multi-agent research pipeline that fetches academic papers from arXiv, extracts and indexes them hierarchically (RAPTOR), retrieves with hybrid dense+sparse+reranking, synthesizes using CoT/ToT/Reflexion, and fact-checks with RAGAS — with human-in-the-loop checkpoints, 4-layer caching, and full token tracking.

## v2 Improvements
- **LLM Client Abstraction**: Seamless switch between Anthropic API and AWS Bedrock
- **Token Tracking**: Per-agent token usage and cost reporting via `llm_client.py`
- **Vectorless RAG**: Optional LLM-based retrieval without vector DB dependency
- **Level 0 Chunking Integration**: Semantic chunker properly feeds RAPTOR leaves
- **MCP Server Stub**: Reference arXiv MCP server implementation
- **Full Test Suite**: Unit, integration, RAGAS regression, and smoke tests
- **RAGAS CLI**: `python -m evaluation.ragas_eval --quick` for smoke tests
- **Token Budget Controls**: Hard caps on input tokens and paper truncation

---

## Complete Setup Guide (From Zero to Running)

### Prerequisites
| Tool | Version | Why |
|------|---------|-----|
| Python | 3.11+ | Required for LangGraph + type hints |
| Docker & Docker Compose | Latest | Redis + PostgreSQL containers |
| AWS CLI | v2 | Bedrock access + S3/DynamoDB |

### Step 1: Unzip and Enter
```bash
unzip research-assistant.zip && cd research-assistant
```

### Step 2: Virtual Environment
```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3: Start Infrastructure
```bash
docker-compose up -d
docker-compose ps  # Should show redis + postgres "Up"
```

### Step 4: Configure
```bash
cp .env.development .env
# Edit .env: set ANTHROPIC_API_KEY and optionally LANGCHAIN_API_KEY
```

### Step 5: AWS Bedrock (Production)
```bash
aws configure  # Access Key, Secret, us-east-1, json
# AWS Console → Bedrock → Model access → Enable Claude models
aws bedrock list-foundation-models --query "modelSummaries[?providerName=='Anthropic']"
```

### Step 6: Run
```bash
# Terminal 1: API
uvicorn api.main:app --reload --port 8000

# Terminal 2: UI
streamlit run ui/app.py

# Terminal 3: RAGAS evaluation
python -m evaluation.ragas_eval --quick
```

### Step 7: Verify
```bash
curl http://localhost:8000/health
python -m pytest tests/unit/ -v
```

---

## Project Structure
```
research-assistant/
├── agents/                  # LangGraph agent nodes
│   ├── state.py             # ResearchState — the contract
│   ├── supervisor.py        # LLM-driven routing (Haiku)
│   ├── fetcher.py           # ReAct + arXiv tools + async fetch
│   ├── extractor.py         # CoT extraction + HyDE + RAPTOR + L0 chunking
│   ├── synthesizer.py       # CoT + ToT + Self-Consistency + Reflexion
│   ├── critic.py            # Few-shot critic + RAGAS faithfulness
│   └── graph.py             # LangGraph assembly + HITL checkpoints
├── rag/                     # Retrieval pipeline
│   ├── vector_store.py      # ChromaDB singleton
│   ├── embedder.py          # SPECTER2 embedder
│   ├── hyde.py              # Hypothetical Document Embeddings
│   ├── raptor.py            # 3-level hierarchical indexing
│   ├── retriever.py         # Hybrid dense+sparse+RRF+rerank+vectorless
│   └── chunker.py           # Semantic chunking for Level 0
├── cache/                   # 4-layer Redis caching
│   ├── redis_client.py      # Shared connection pool
│   ├── semantic_cache.py    # L1: query-level (24h)
│   ├── embedding_cache.py   # L2: embedding-level (7d)
│   └── llm_cache.py         # L3: LLM response-level (1h)
├── persistence/             # Durable storage
│   ├── postgres.py          # LangGraph checkpointer
│   ├── dynamo.py            # Session history
│   └── s3.py                # Papers + golden dataset
├── evaluation/              # Quality assurance
│   ├── ragas_eval.py        # 5-metric evaluation + regression gate + CLI
│   ├── golden_dataset.py    # 3 methods for test data generation
│   └── token_tracker.py     # Per-agent token + cost tracking
├── mcp/                     # MCP server
│   └── arxiv_server.py      # arXiv MCP server reference
├── llm_client.py            # LLM abstraction (Anthropic / Bedrock)
├── config.py                # Pydantic settings (all env vars)
├── api/main.py              # FastAPI gateway + HITL endpoints
├── ui/app.py                # Streamlit UI + RAGAS dashboard
├── infrastructure/          # AWS CDK deployment
├── tests/                   # Full test suite
├── docker-compose.yml       # Local Redis + PostgreSQL
├── Dockerfile               # Production container
└── requirements.txt         # Python dependencies
```

---

## Claude Code / VS Code Agent Workflow

Claude Code is a CLI tool for agentic coding from your terminal.

```bash
npm install -g @anthropic-ai/claude-code
cd research-assistant
claude

# Example commands:
> "Add error handling to the fetcher agent"
> "Write unit tests for the supervisor routing logic"
> "Explain the RAPTOR indexing flow"
> "Refactor retriever.py to support configurable reranker models"
```

Claude Code reads your project files, understands context, and makes changes directly.
