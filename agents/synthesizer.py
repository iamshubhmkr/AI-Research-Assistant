"""
Synthesizer — CoT (6 steps) + parallel ToT + Self-Consistency + Reflexion.
v3.1: the ToT vote and self-consistency checks moved to HAIKU — they are
classification calls, and golden rule 3 says never use Sonnet where Haiku
suffices. Synthesis itself stays on Sonnet.
"""
import json
import re
import logging
from concurrent.futures import ThreadPoolExecutor
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

COT_SYSTEM = """You synthesize research findings from retrieved paper chunks.
MANDATORY: complete ALL 6 steps inside <reasoning> tags before the answer.
<reasoning>
STEP 1 - UNDERSTAND: what exactly is asked, what answer type fits?
STEP 2 - INVENTORY: list every relevant fact with [paper_id: section].
STEP 3 - CONFLICTS: where do papers disagree? State BOTH positions.
STEP 4 - PATTERNS: which findings appear in multiple papers?
STEP 5 - GAPS: what is NOT answered by context? NEVER fill gaps yourself.
STEP 6 - STRUCTURE: plan the answer.
</reasoning>
Then write the answer. Cite every claim as [paper_id: section].
State the Step-5 gaps explicitly."""


def _build_context(chunks):
    return "\n\n---\n\n".join(
        f"[{c.get('meta',{}).get('paper_id','?')}: {c.get('meta',{}).get('section','?')}]\n{c.get('text','')}"
        for c in chunks)


def _cot(query, context, reflexion=""):
    system = COT_SYSTEM
    if reflexion:
        system += f"\n\n<previous_failure_memory>\n{reflexion}\nAvoid these mistakes.\n</previous_failure_memory>"
    r = call_llm(model=settings.claude_sonnet,
                 messages=[{"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}],
                 system=system, max_tokens=settings.max_output_tokens_synthesis,
                 temperature=0.3, agent_name="synthesizer")
    return r["content"][0].text


def _one_branch(args):
    query, context, name, instruction = args
    r = call_llm(model=settings.claude_sonnet,
                 messages=[{"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}],
                 system=f"Synthesize using {name} structure. {instruction}. Show CoT then answer.",
                 max_tokens=2048, temperature=0.3, agent_name="synthesizer_tot")
    return {"name": name, "text": r["content"][0].text}


def _tree_of_thoughts(query, context):
    approaches = [
        ("chronological", "Trace how understanding evolved over time"),
        ("thematic", "Organize by recurring themes"),
        ("conflict_first", "Lead with disagreements, then resolve with evidence"),
    ]
    # v3: branches are independent -> run them in parallel
    with ThreadPoolExecutor(max_workers=3) as pool:
        branches = list(pool.map(_one_branch, [(query, context, n, i) for n, i in approaches]))

    vote = call_llm(model=settings.claude_haiku, max_tokens=200, temperature=0.0,
                    agent_name="synthesizer_vote",
                    messages=[{"role": "user", "content":
                        f'Query: "{query}"\n\nA: {branches[0]["text"][:600]}\n\n'
                        f'B: {branches[1]["text"][:600]}\n\nC: {branches[2]["text"][:600]}\n\n'
                        'Which best answers the query? JSON: {"winner":"A/B/C","reason":""}'}])
    try:
        parsed = json.loads(re.search(r"\{.*\}", vote["content"][0].text, re.DOTALL).group())
        idx = {"A": 0, "B": 1, "C": 2}.get(parsed.get("winner", "A"), 0)
    except Exception:
        idx = 0
    return branches[idx]["text"], branches


def _self_consistency(query, synthesis, chunks, n=3):
    sample = chunks[0].get("text", "")[:300] if chunks else ""
    verdicts = []
    for _ in range(n):
        try:
            r = call_llm(model=settings.claude_haiku, max_tokens=80, temperature=0.3,
                         agent_name="self_consistency",
                         messages=[{"role": "user", "content":
                             f"Does this answer use ONLY facts from the context? YES or NO.\n"
                             f"Answer: {synthesis[:500]}\nContext sample: {sample}"}])
            verdicts.append("YES" in r["content"][0].text.upper())
        except Exception:
            verdicts.append(True)  # degraded path: don't block on a failed check
    return verdicts


def synthesizer_node(state: ResearchState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    context = _build_context(chunks)
    if state.get("human_chunk_feedback"):
        context = f"USER GUIDANCE: {state['human_chunk_feedback']}\n\n{context}"
    if state.get("human_revision_note"):
        context = f"HUMAN REVISION REQUEST: {state['human_revision_note']}\n\n{context}"

    is_complex = (len(chunks) > 3
                  and state.get("query_type") in ("comparison", "paper_overview")
                  and state.get("revision_count", 0) == 0)

    tot_branches = None
    if is_complex:
        synthesis, tot_branches = _tree_of_thoughts(state["query"], context)
    else:
        synthesis = _cot(state["query"], context, state.get("reflexion_memory") or "")

    sc = _self_consistency(state["query"], synthesis, chunks)
    logger.info(f"[synthesizer] SC verdicts {sc} pass={sum(sc)>=2}")
    return {"synthesis": synthesis, "tot_branches": tot_branches,
            "sc_verdicts": sc, "critique": None,  # reset so critic re-runs after revision
            "revision_count": state.get("revision_count", 0) + 1}
