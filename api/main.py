"""
FastAPI Gateway — HITL endpoints + streaming SSE + token tracking.

Scaling considerations:
  - uvicorn workers=2 for CPU-bound LLM calls (GIL limits benefit of more)
  - asyncio for I/O-bound PDF fetching
  - Redis shared across all worker processes
  - PostgreSQL connection pooling via LangGraph checkpointer
"""
import uuid
import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agents.graph import graph
from cache.semantic_cache import SemanticQueryCache
from persistence.dynamo import SessionStore
from llm_client import get_token_usage, reset_token_usage, estimate_cost
from config import settings

logging.basicConfig(level=settings.log_level)
app = FastAPI(title="AI Research Assistant")
semantic_cache = SemanticQueryCache()
session_store = SessionStore()


class StartRequest(BaseModel):
    query: str
    auto_search: bool = True
    paper_urls: list[str] = []
    user_id: str = "anonymous"


class ApproveChunksRequest(BaseModel):
    session_id: str
    remove_chunk_indices: list[int] = []
    feedback: str = ""


class ResolveCritiqueRequest(BaseModel):
    session_id: str
    override: bool = False
    revision_note: str = ""


@app.post("/research/start")
async def start_research(req: StartRequest):
    reset_token_usage()
    start_time = time.time()

    cached = semantic_cache.get(req.query)
    if cached:
        return {"status": "complete", "from_cache": True, **cached}

    session_id = str(uuid.uuid4())
    session_store.create(session_id, req.user_id)
    config = {"configurable": {"thread_id": session_id}}

    initial_state = {
        "query": req.query, "query_type": "specific_fact", "session_id": session_id,
        "paper_urls": req.paper_urls, "auto_search": req.auto_search,
        "raw_papers": [], "extracted_facts": [], "retrieved_chunks": [],
        "chunks_approved": False, "needs_revision": False, "revision_count": 0,
        "next": "", "sources": [], "token_usage": {}, "sc_verdicts": [],
        "faithfulness_score": 0.0, "total_latency_ms": 0.0, "estimated_cost_usd": 0.0,
    }

    state = await graph.ainvoke(initial_state, config=config)
    return {
        "status": "awaiting_chunk_approval", "session_id": session_id,
        "retrieved_chunks": state.get("retrieved_chunks", []),
        "token_usage": get_token_usage(),
    }


@app.post("/research/approve_chunks")
async def approve_chunks(req: ApproveChunksRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    current = graph.get_state(config)
    chunks = current.values.get("retrieved_chunks", [])
    filtered = [c for i, c in enumerate(chunks) if i not in req.remove_chunk_indices]
    graph.update_state(config, {
        "retrieved_chunks": filtered, "chunks_approved": True,
        "human_chunk_feedback": req.feedback or None,
    })
    state = await graph.ainvoke(None, config=config)
    return {
        "status": "awaiting_critique_review", "synthesis": state.get("synthesis"),
        "critique": state.get("critique"), "faithfulness_score": state.get("faithfulness_score"),
        "needs_revision": state.get("needs_revision"),
    }


@app.post("/research/resolve_critique")
async def resolve_critique(req: ResolveCritiqueRequest):
    config = {"configurable": {"thread_id": req.session_id}}
    updates = {"human_override": req.override}
    if req.override:
        updates["needs_revision"] = False
    if req.revision_note:
        updates["human_revision_note"] = req.revision_note
    graph.update_state(config, updates)
    state = await graph.ainvoke(None, config=config)

    usage = get_token_usage()
    cost = estimate_cost(usage)

    if state.get("final_answer"):
        semantic_cache.set(state.get("query", ""), state["final_answer"],
                          state.get("sources", []), state.get("ragas_scores", {}))
        session_store.add_query(req.session_id, state.get("query", ""),
                               state["final_answer"], state.get("ragas_scores", {}))

    return {
        "status": "complete", "final_answer": state.get("final_answer"),
        "sources": state.get("sources", []), "ragas_scores": state.get("ragas_scores"),
        "faithfulness_score": state.get("faithfulness_score"),
        "token_usage": usage, "estimated_cost_usd": cost,
    }


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.deployment_env}
