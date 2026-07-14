"""
End-to-end pipeline test with STUBBED Bedrock (no AWS credentials needed).

Exercises the full graph against live Redis + Postgres + Chroma:
  fetch (real HTTP) -> extract -> chunk -> index -> retrieve (BM25+dense+RRF)
  -> HITL pause 1 (chunk approval) -> synthesize -> critic (forces ONE
  revision round) -> HITL pause 2 -> revision -> approval -> final answer.

Only llm_client.call_llm and the Titan embedder are stubbed — everything
else (graph mechanics, interrupts, checkpointing, caching, chunking,
retrieval math) runs for real.

Usage:  make up && .venv/bin/python scripts/e2e_local.py
"""
import asyncio
import http.server
import json
import re
import socketserver
import sys
import threading
import shutil
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# fresh vector store so reruns are deterministic
shutil.rmtree(ROOT / "data" / "chroma", ignore_errors=True)

# ── Stub the Titan embedder (deterministic, no boto3) ───────────────
import rag.embedder as embedder_mod  # noqa: E402


def _fake_init(self):
    self.client = None


def _fake_encode(self, text: str) -> list[float]:
    import hashlib
    h = hashlib.sha256(text.encode()).digest()
    return [(b - 128) / 128.0 for b in (h * 2)[:64]]


embedder_mod.TitanEmbedder.__init__ = _fake_init
embedder_mod.TitanEmbedder.encode = _fake_encode

# ── Stub call_llm with agent-aware canned responses ─────────────────
import llm_client  # noqa: E402

_critic_calls = {"n": 0}


def _route(s: dict) -> str:
    if s["raw_papers_count"] == 0 and (s["auto_search"] or s["paper_urls_count"] > 0):
        return "fetcher"
    if s["raw_papers_count"] > 0 and not s["sections_populated"]:
        return "extractor"
    if s["sections_populated"] and s["retrieved_chunks_count"] == 0:
        return "retriever"
    if s["retrieved_chunks_count"] > 0 and not s["synthesis_exists"]:
        return "synthesizer"
    if s["synthesis_exists"] and not s["critique_exists"]:
        return "critic"
    if s["needs_revision"] and s["revision_count"] < 2:
        return "synthesizer"
    return "END"


def _text(t):
    return {"content": [SimpleNamespace(type="text", text=t)],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 100, "output_tokens": 50}}


def fake_call_llm(model, messages, system="", max_tokens=1024, temperature=0.0,
                  tools=None, agent_name="unknown"):
    if agent_name == "supervisor":
        summary = json.loads(re.search(r"\{[^{}]*raw_papers_count.*?\}",
                                       messages[0]["content"], re.DOTALL).group())
        return _text(json.dumps({"next": _route(summary), "reason": "stub rules"}))
    if agent_name == "extractor":
        return _text(json.dumps({"contribution": "stub contribution",
                                 "methodology": "stub method", "key_results": [],
                                 "limitations": [], "key_claims": []}))
    if agent_name == "query_expansion":
        return _text("alternative query one\nalternative query two")
    if agent_name in ("hyde", "raptor_summary", "compression", "reflexion"):
        return _text(f"stub {agent_name} text about retrieval augmented generation.")
    if agent_name in ("synthesizer", "synthesizer_tot"):
        return _text("Stubbed synthesis: RAG limits multi-hop reasoning [p1: intro]. "
                     "Gaps: none stated.")
    if agent_name == "synthesizer_vote":
        return _text(json.dumps({"winner": "A", "reason": "stub"}))
    if agent_name == "self_consistency":
        return _text("YES")
    if agent_name == "faithfulness_judge":
        score = 0.5 if _critic_calls["n"] <= 1 else 0.95
        return _text(json.dumps({"score": score}))
    if agent_name == "critic":
        _critic_calls["n"] += 1
        if _critic_calls["n"] == 1:   # force exactly one revision round
            return _text(json.dumps({"approved": False,
                                     "issues": ["stub: unsupported claim"],
                                     "severity": "moderate",
                                     "reflexion": "be precise"}))
        return _text(json.dumps({"approved": True, "issues": [],
                                 "severity": "none", "reflexion": ""}))
    return _text("stub")


llm_client.call_llm = fake_call_llm
# agents imported call_llm by name — patch their references too
for mod_name in ("agents.supervisor", "agents.fetcher", "agents.extractor",
                 "agents.synthesizer", "agents.critic", "rag.hyde", "rag.raptor",
                 "rag.retriever"):
    __import__(mod_name)
    setattr(sys.modules[mod_name], "call_llm", fake_call_llm)

# ── Serve a sample document over real HTTP ──────────────────────────
DOC = """# Abstract

Retrieval augmented generation grounds language models in retrieved documents
and reduces hallucination across knowledge intensive tasks substantially.

# Introduction

We identify three failure modes in retrieval augmented generation for
compositional question answering: single-step retrieval insufficiency,
cross-document coreference failure, and reasoning chain collapse during
synthesis. These failures compound on multi-hop questions.

# Methods

We evaluate hybrid retrieval combining dense embeddings with lexical BM25
scoring fused through reciprocal rank fusion across three benchmarks.

# Results

Hybrid retrieval improves context recall from 0.71 to 0.89 on our benchmark,
a twenty five percent relative improvement over dense-only retrieval.
"""

docroot = ROOT / "data" / "e2e_docs"
docroot.mkdir(parents=True, exist_ok=True)
(docroot / "sample.md").write_text(DOC)


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(docroot), **kw)

    def log_message(self, *a):
        pass


httpd = socketserver.TCPServer(("127.0.0.1", 0), _Quiet)
port = httpd.server_address[1]
threading.Thread(target=httpd.serve_forever, daemon=True).start()
DOC_URL = f"http://127.0.0.1:{port}/sample.md"


# ── Drive the graph exactly like the API does ───────────────────────
async def main():
    from agents.graph import get_graph
    ok = True

    def check(cond, label):
        nonlocal ok
        print(("  PASS  " if cond else "  FAIL  ") + label)
        ok = ok and cond

    import uuid
    thread_id = f"e2e-local-{uuid.uuid4().hex[:8]}"  # fresh thread: old checkpoints
    graph = await get_graph()                        # in Postgres outlive the wiped Chroma
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    initial = {"query": "What are the limitations of RAG for multi-hop reasoning?",
               "query_type": "specific_fact", "session_id": thread_id,
               "paper_urls": [DOC_URL], "auto_search": False,
               "raw_papers": [], "extracted_facts": [], "retrieved_chunks": [],
               "chunks_approved": False, "needs_revision": False, "revision_count": 0,
               "next": "", "sources": [], "token_usage": {}, "sc_verdicts": [],
               "faithfulness_score": 0.0, "total_latency_ms": 0.0,
               "estimated_cost_usd": 0.0}

    print("\n[1] start -> fetch/extract/index/retrieve, pause before synthesizer")
    state = await graph.ainvoke(initial, config=config)
    snap = await graph.aget_state(config)
    check(snap.next == ("synthesizer",), f"paused before synthesizer (next={snap.next})")
    check(len(state.get("raw_papers", [])) == 1, "fetched 1 document over HTTP")
    check(bool(state.get("sections")), "sections extracted")
    chunks = state.get("retrieved_chunks", [])
    check(len(chunks) > 0, f"retrieved {len(chunks)} chunks (hybrid BM25+dense+RRF)")

    print("\n[2] approve chunks -> synthesize -> critic flags revision, pause after critic")
    await graph.aupdate_state(config, {"retrieved_chunks": chunks, "chunks_approved": True,
                                       "human_chunk_feedback": "focus on multi-hop"})
    state = await graph.ainvoke(None, config=config)
    snap = await graph.aget_state(config)
    check(state.get("synthesis") is not None, "synthesis produced")
    check(state.get("needs_revision") is True, "critic requested revision (forced by stub)")
    check(bool(snap.next), f"paused after critic (next={snap.next})")

    print("\n[3] resolve -> auto-resume through revision -> critic approves, pause again")
    await graph.aupdate_state(config, {"human_override": False})
    state = await graph.ainvoke(None, config=config)
    snap = await graph.aget_state(config)
    while snap.next and snap.next[0] == "synthesizer":
        state = await graph.ainvoke(None, config=config)
        snap = await graph.aget_state(config)
    check(state.get("revision_count", 0) == 2, f"revision ran (count={state.get('revision_count')})")
    check(state.get("needs_revision") is False, "critic approved after revision")
    check(bool(snap.next), "paused after critic for round-2 human review")

    print("\n[4] resolve round 2 -> END with final answer")
    await graph.aupdate_state(config, {"human_override": False})
    state = await graph.ainvoke(None, config=config)
    snap = await graph.aget_state(config)
    while snap.next and snap.next[0] == "synthesizer":
        state = await graph.ainvoke(None, config=config)
        snap = await graph.aget_state(config)
    check(snap.next == (), f"graph ended (next={snap.next})")
    check(bool(state.get("final_answer")), "final_answer set")
    check(state.get("sources") == [DOC_URL], "sources recorded")

    print("\n" + ("E2E PIPELINE TEST: ALL CHECKS PASSED" if ok else "E2E PIPELINE TEST: FAILURES ABOVE"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
