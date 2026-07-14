"""
FastAPI gateway v3.1 — HITL endpoints + SSE progress streaming + token/cost reporting.

HITL mechanics (fixed in v3.1):
  - Pause 1: graph interrupts BEFORE synthesizer; /research/start returns
    awaiting_chunk_approval; /research/approve_chunks updates state and resumes.
  - Pause 2: graph interrupts AFTER critic; /research/resolve_critique resumes,
    auto-stepping through the synthesizer interrupt during REVISION rounds
    (the human already approved the chunks) and returning either a fresh
    critique round or the final answer.
"""
import uuid
import json
import asyncio
import logging
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from agents.graph import get_graph
from cache.semantic_cache import SemanticQueryCache
from persistence.dynamo import SessionStore
from llm_client import get_token_usage, reset_token_usage, estimate_cost
from config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)
app = FastAPI(title="AI Research Assistant v3.1")
session_store = SessionStore()
_semantic_cache = None


def semantic_cache() -> SemanticQueryCache:
    global _semantic_cache
    if _semantic_cache is None:
        _semantic_cache = SemanticQueryCache()
    return _semantic_cache


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


def _initial_state(req, session_id):
    return {"query": req.query, "query_type": "specific_fact", "session_id": session_id,
            "paper_urls": req.paper_urls, "auto_search": req.auto_search,
            "raw_papers": [], "extracted_facts": [], "retrieved_chunks": [],
            "chunks_approved": False, "needs_revision": False, "revision_count": 0,
            "next": "", "sources": [], "token_usage": {}, "sc_verdicts": [],
            "faithfulness_score": 0.0, "total_latency_ms": 0.0, "estimated_cost_usd": 0.0}


def _config(session_id):
    return {"configurable": {"thread_id": session_id},
            "recursion_limit": settings.max_graph_iterations}


def _critique_payload(state):
    return {"synthesis": state.get("synthesis"),
            "critique": state.get("critique"),
            "faithfulness_score": state.get("faithfulness_score"),
            "needs_revision": state.get("needs_revision"),
            "revision_count": state.get("revision_count", 0)}


@app.post("/research/start")
async def start_research(req: StartRequest):
    graph = await get_graph()
    reset_token_usage()
    cached = semantic_cache().get(req.query)
    if cached:
        return {"status": "complete", "from_cache": True, **cached}
    session_id = str(uuid.uuid4())
    session_store.create(session_id, req.user_id)
    config = _config(session_id)
    state = await graph.ainvoke(_initial_state(req, session_id), config=config)
    snap = await graph.aget_state(config)
    if not snap.next:  # pipeline ended without pausing — nothing fetched/retrieved
        return {"status": "complete", "session_id": session_id,
                "final_answer": state.get("final_answer"),
                "sources": state.get("sources", []),
                "note": "pipeline finished without a chunk-review pause "
                        "(no documents were fetched or no chunks retrieved)",
                "token_usage": get_token_usage()}
    return {"status": "awaiting_chunk_approval", "session_id": session_id,
            "retrieved_chunks": state.get("retrieved_chunks", []),
            "token_usage": get_token_usage()}


@app.post("/research/start_stream")
async def start_research_stream(req: StartRequest):
    """v3: SSE streaming — emit an event as each agent completes."""
    graph = await get_graph()
    reset_token_usage()
    cached = semantic_cache().get(req.query)

    async def events():
        if cached:
            yield f"data: {json.dumps({'event': 'complete', 'from_cache': True, **cached})}\n\n"
            return
        session_id = str(uuid.uuid4())
        session_store.create(session_id, req.user_id)
        config = _config(session_id)
        async for chunk in graph.astream(_initial_state(req, session_id), config=config):
            node = list(chunk.keys())[0]
            yield f"data: {json.dumps({'event': 'node_complete', 'node': node, 'session_id': session_id})}\n\n"
            await asyncio.sleep(0)
        snap = await graph.aget_state(config)
        event = "paused_for_hitl" if snap.next else "complete"
        yield f"data: {json.dumps({'event': event, 'session_id': session_id})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/research/approve_chunks")
async def approve_chunks(req: ApproveChunksRequest):
    graph = await get_graph()
    config = _config(req.session_id)
    current = await graph.aget_state(config)
    chunks = current.values.get("retrieved_chunks", [])
    filtered = [c for i, c in enumerate(chunks) if i not in req.remove_chunk_indices]
    await graph.aupdate_state(config, {"retrieved_chunks": filtered, "chunks_approved": True,
                                       "human_chunk_feedback": req.feedback or None})
    state = await graph.ainvoke(None, config=config)
    snap = await graph.aget_state(config)
    status = "awaiting_critique_review" if snap.next else "complete"
    return {"status": status, **_critique_payload(state)}


@app.post("/research/resolve_critique")
async def resolve_critique(req: ResolveCritiqueRequest):
    graph = await get_graph()
    config = _config(req.session_id)
    updates = {"human_override": req.override}
    if req.override:
        updates["needs_revision"] = False
    if req.revision_note:
        updates["human_revision_note"] = req.revision_note
    await graph.aupdate_state(config, updates)

    state = await graph.ainvoke(None, config=config)
    snap = await graph.aget_state(config)
    # Revision round: the graph pauses before synthesizer again — the human
    # already approved the chunks, so step straight through that interrupt.
    while snap.next and snap.next[0] == "synthesizer":
        state = await graph.ainvoke(None, config=config)
        snap = await graph.aget_state(config)

    if snap.next:  # paused after critic — a fresh critique round for the human
        return {"status": "awaiting_critique_review", **_critique_payload(state)}

    usage = get_token_usage()
    final_answer = state.get("final_answer")
    if final_answer and not req.override:
        # never cache an answer the human had to force past the critic
        semantic_cache().set(state.get("query", ""), final_answer,
                             state.get("sources", []), state.get("ragas_scores") or {})
    if final_answer:
        session_store.add_query(req.session_id, state.get("query", ""),
                                final_answer, state.get("ragas_scores") or {})
        if not req.override:
            # high-faithfulness answers become golden-dataset candidates (best-effort)
            from evaluation.golden_dataset import maybe_collect_from_production
            maybe_collect_from_production(state.get("query", ""), final_answer,
                                          state.get("retrieved_chunks", []),
                                          state.get("faithfulness_score", 0.0))
    return {"status": "complete", "final_answer": final_answer,
            "sources": state.get("sources", []),
            "faithfulness_score": state.get("faithfulness_score"),
            "token_usage": usage, "estimated_cost_usd": estimate_cost(usage)}


@app.get("/health")
def health():
    return {"status": "ok", "env": settings.deployment_env,
            "provider": settings.llm_provider, "version": "v3.1"}
