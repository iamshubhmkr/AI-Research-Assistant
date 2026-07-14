"""
Run a REAL query through the live pipeline (Bedrock) with auto-approved HITL
pauses, and append the result as a RAGAS test case.

This is how the eval set gets real pipeline output instead of static
fixtures. Add a human-written ground_truth to each case afterwards to unlock
the precision/recall/correctness metrics.

Usage:
  .venv/bin/python scripts/collect_eval_case.py "your research question"
  .venv/bin/python scripts/collect_eval_case.py "question" --url https://... [--out evaluation/cases.json]
"""
import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


async def run(query: str, urls: list[str], out_path: Path) -> int:
    from agents.graph import get_graph
    from config import settings

    graph = await get_graph()
    thread = f"collect-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread},
              "recursion_limit": settings.max_graph_iterations}
    initial = {"query": query, "query_type": "specific_fact", "session_id": thread,
               "paper_urls": urls, "auto_search": not urls,
               "raw_papers": [], "extracted_facts": [], "retrieved_chunks": [],
               "chunks_approved": False, "needs_revision": False, "revision_count": 0,
               "next": "", "sources": [], "token_usage": {}, "sc_verdicts": [],
               "faithfulness_score": 0.0, "total_latency_ms": 0.0,
               "estimated_cost_usd": 0.0}

    print(f"[collect] running pipeline for: {query!r}")
    state = await graph.ainvoke(initial, config=config)
    snap = await graph.aget_state(config)
    if snap.next != ("synthesizer",):
        print(f"[collect] pipeline did not reach chunk review (next={snap.next}) — aborting")
        return 1

    chunks = state.get("retrieved_chunks", [])
    print(f"[collect] auto-approving {len(chunks)} chunks")
    await graph.aupdate_state(config, {"chunks_approved": True})
    state = await graph.ainvoke(None, config=config)
    snap = await graph.aget_state(config)

    # auto-accept critic verdicts (incl. stepping through revision rounds)
    while snap.next:
        if snap.next[0] != "synthesizer":
            await graph.aupdate_state(config, {"human_override": False})
        state = await graph.ainvoke(None, config=config)
        snap = await graph.aget_state(config)

    answer = state.get("final_answer")
    if not answer:
        print("[collect] pipeline ended without a final answer — nothing to save")
        return 1

    case = {"question": query, "answer": answer,
            "contexts": [c.get("text", "") for c in chunks],
            "ground_truth": "",  # fill in by hand to unlock all 5 metrics
            "faithfulness_at_capture": state.get("faithfulness_score"),
            "sources": state.get("sources", [])}
    cases = json.loads(out_path.read_text()) if out_path.exists() else []
    cases.append(case)
    out_path.write_text(json.dumps(cases, indent=2))
    print(f"[collect] saved case #{len(cases)} -> {out_path}")
    print(f"[collect] evaluate with: python -m evaluation.ragas_eval --cases {out_path}")
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query")
    p.add_argument("--url", action="append", default=[], help="document URL (repeatable); skips arXiv auto-search")
    p.add_argument("--out", default="evaluation/cases.json")
    args = p.parse_args()
    raise SystemExit(asyncio.run(run(args.query, args.url, ROOT / args.out)))


if __name__ == "__main__":
    main()
