# Antigravity / Claude Code Agent Instructions

## Project Overview
Multi-agent research pipeline: LangGraph + Claude Sonnet 4.5 + AWS Bedrock.
5 agents: Fetcher (ReAct) → Extractor (CoT) → Retriever (Hybrid) → Synthesizer (CoT+ToT) → Critic (Few-shot).

## Development Rules
1. Never modify agents/state.py without updating ALL agents that read/write those fields.
2. Always run `python -m pytest tests/unit/ -v` before committing agent changes.
3. LLM calls in agents MUST go through llm_client.call_llm() for token tracking.
4. New agents must register in agents/graph.py and handle all state fields they write.
5. RAGAS scores must stay above targets in config.py.

## Key Files
- config.py: all settings — never hardcode values
- llm_client.py: LLM abstraction — Anthropic or Bedrock
- agents/state.py: the contract — change carefully
- agents/graph.py: graph topology + HITL checkpoints
- evaluation/ragas_eval.py: run before every merge

## Running Locally
```bash
docker-compose up -d
cp .env.development .env
uvicorn api.main:app --reload --port 8000
streamlit run ui/app.py
```
