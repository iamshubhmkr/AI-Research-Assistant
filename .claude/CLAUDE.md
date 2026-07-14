# AI Research Assistant v3 — Project Memory for Claude Code

## What This Project Is
A production-grade multi-agent research pipeline: 6 agents (supervisor, fetcher,
extractor, retriever, synthesizer, critic) orchestrated by LangGraph, grounded by
advanced RAG (HyDE + RAPTOR + hybrid retrieval), fact-checked by RAGAS, with two
human-in-the-loop checkpoints persisted in PostgreSQL.

## Architecture in One Paragraph
A query enters api/main.py → L1 semantic cache check → LangGraph state machine.
The supervisor (Haiku) routes after every node. Fetcher (ReAct + arXiv tools)
collects documents via rag/document_router.py (PDF/DOCX/HTML/CSV). Extractor
splits sections, extracts facts (CoT), chunks to Level 0, builds the RAPTOR tree
(L1+L2), generates a HyDE doc. Retriever runs the 8-stage hybrid pipeline.
HITL pause 1 (interrupt_before synthesizer). Synthesizer writes with CoT/parallel
ToT + self-consistency. Critic fact-checks with few-shot + RAGAS faithfulness.
HITL pause 2 (interrupt_after critic). Revision loop max 2× with Reflexion memory.
Answer cached (L1), session saved (DynamoDB), cost reported.

## Golden Rules (NEVER violate)
1. ALL LLM calls go through llm_client.call_llm() — never import anthropic/boto3
   in an agent. This is how we get retries, circuit breaking, fallback, and token
   tracking for free.
2. agents/state.py is THE contract. Changing a field means updating every agent
   that reads/writes it. Fields multiple agents append to MUST use
   Annotated[list, operator.add].
3. Model routing: Haiku for classification/summarization/compression; Sonnet for
   reasoning/synthesis/critique. Never use Sonnet where Haiku suffices.
4. New text formats go in rag/document_router.py ONLY. Everything downstream
   works on plain text.
5. Config values live in config.py (pydantic-settings). No magic numbers in code.
6. Run `make test` before any commit. Run `make eval` after any prompt change.

## Key Files
- config.py — all settings           - llm_client.py — LLM abstraction + resilience
- agents/state.py — the contract     - agents/graph.py — topology + HITL interrupts
- rag/retriever.py — 8-stage hybrid  - rag/document_router.py — multi-format parsing
- evaluation/ragas_eval.py — quality - api/main.py — HITL endpoints + SSE streaming

## Commands
make up      # start Redis + PostgreSQL
make test    # unit tests
make eval    # RAGAS smoke evaluation
make run     # API on :8000
make ui      # Streamlit UI

## Style
- Python 3.11, type hints on public functions, docstrings explain WHY not WHAT.
- Every module's docstring includes the design decision it embodies — these
  double as interview talking points.
- Tests live in tests/unit (fast, no network), tests/ragas (LLM-dependent),
  tests/smoke (post-deploy).
