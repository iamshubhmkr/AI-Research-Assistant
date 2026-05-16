"""
Synthesizer Agent — CoT + Tree-of-Thoughts + Self-Consistency + Reflexion.

Token optimization:
  - Context is compressed before reaching synthesizer (~500 tokens vs ~3000)
  - ToT only fires for complex queries on first attempt (3x Sonnet cost)
  - Self-consistency uses temperature=0.3 for independent samples
  - Reflexion injects specific failure memory, not generic "try harder"

Scaling consideration:
  ToT generates 3 full synthesis branches — biggest latency bottleneck.
  Optimization: run 3 branches with asyncio.gather (8s → ~3s).
  Further: skip ToT for simple factual queries (query_type detection).
"""
import json
import re
import logging
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

COT_SYSTEM = """You synthesize academic research findings from retrieved paper chunks.

MANDATORY: Complete ALL 6 reasoning steps inside <reasoning> tags before writing your answer.

<reasoning>
STEP 1 — UNDERSTAND THE QUERY
What exactly is being asked? What type of answer is needed?

STEP 2 — INVENTORY ALL EVIDENCE
List every relevant fact from the context with its source.
Format: [paper_id: section] → exact claim or number

STEP 3 — IDENTIFY CONFLICTS
Where do papers disagree? State BOTH positions explicitly.

STEP 4 — FIND CONSISTENT PATTERNS
What findings appear in multiple papers? These warrant stronger claims.

STEP 5 — IDENTIFY GAPS
What parts of the query are NOT answered by the context?
NEVER fill gaps with outside knowledge.

STEP 6 — PLAN STRUCTURE
Given steps 1-5, how should the answer be organized?
</reasoning>

After completing ALL 6 steps, write the final answer.
Cite every claim as [paper_id: section].
If gaps were identified in Step 5, state them explicitly."""


def _build_context(chunks: list) -> str:
    return "\n\n---\n\n".join([
        f"[{c.get('meta', {}).get('paper_id', '?')}: {c.get('meta', {}).get('section', '?')}]\n{c.get('text', '')}"
        for c in chunks
    ])


def _cot_synthesis(query: str, context: str, reflexion: str = "") -> str:
    system = COT_SYSTEM
    if reflexion:
        system += f"\n\n<previous_failure_memory>\n{reflexion}\nAvoid these mistakes.\n</previous_failure_memory>"
    result = call_llm(
        model=settings.claude_sonnet,
        messages=[{"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}],
        system=system,
        max_tokens=settings.max_output_tokens_synthesis,
        temperature=0.3,
        agent_name="synthesizer",
    )
    return result["content"][0].text


def _tree_of_thoughts(query: str, context: str) -> tuple[str, list]:
    """Generate 3 synthesis approaches, vote on the best."""
    approaches = [
        ("chronological", "Trace how understanding evolved over time across papers"),
        ("thematic", "Organize by recurring themes regardless of paper order"),
        ("conflict_first", "Lead with where papers disagree, then resolve with evidence"),
    ]
    branches = []
    for name, instruction in approaches:
        result = call_llm(
            model=settings.claude_sonnet,
            messages=[{"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}],
            system=f"Synthesize using {name} structure. {instruction}. Show CoT reasoning then write answer.",
            max_tokens=2048,
            temperature=0.3,
            agent_name="synthesizer_tot",
        )
        branches.append({"name": name, "text": result["content"][0].text})

    # Vote on best branch
    vote_result = call_llm(
        model=settings.claude_sonnet,
        messages=[{"role": "user", "content":
            f'Query: "{query}"\n\n'
            f'A: {branches[0]["text"][:600]}\n\n'
            f'B: {branches[1]["text"][:600]}\n\n'
            f'C: {branches[2]["text"][:600]}\n\n'
            f'Which best answers the query? Score each 0-10 on: answers query / uses context / handles conflicts. '
            f'Return JSON: {{"winner":"A/B/C","reason":""}}'
        }],
        max_tokens=200,
        temperature=0.0,
        agent_name="synthesizer_vote",
    )
    raw = vote_result["content"][0].text
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        idx = {"A": 0, "B": 1, "C": 2}.get(parsed.get("winner", "A"), 0)
    except Exception:
        idx = 0
    return branches[idx]["text"], branches


def _self_consistency(query: str, synthesis: str, chunks: list, n: int = 3) -> list[bool]:
    """3 independent grounding checks via temperature=0.3."""
    verdicts = []
    sample = chunks[0].get("text", "")[:300] if chunks else ""
    for _ in range(n):
        result = call_llm(
            model=settings.claude_sonnet,
            messages=[{"role": "user", "content":
                f"Does this answer use ONLY facts from the context below? YES or NO.\n"
                f"Answer excerpt: {synthesis[:500]}\n"
                f"Context sample: {sample}"
            }],
            max_tokens=80,
            temperature=0.3,
            agent_name="self_consistency",
        )
        verdicts.append("YES" in result["content"][0].text.upper())
    return verdicts


def synthesizer_node(state: ResearchState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    context = _build_context(chunks)
    reflexion = state.get("reflexion_memory") or ""
    human_feedback = state.get("human_chunk_feedback") or ""
    if human_feedback:
        context = f"USER GUIDANCE: {human_feedback}\n\n{context}"

    is_complex = (
        len(chunks) > 3
        and state.get("query_type") in ("comparison", "paper_overview")
        and state.get("revision_count", 0) == 0
    )

    tot_branches = None
    if is_complex:
        synthesis, tot_branches = _tree_of_thoughts(state["query"], context)
    else:
        synthesis = _cot_synthesis(state["query"], context, reflexion)

    sc_verdicts = _self_consistency(state["query"], synthesis, chunks)
    logger.info(f"[synthesizer] SC verdicts: {sc_verdicts} (pass={sum(sc_verdicts)>=2})")

    return {
        "synthesis": synthesis,
        "tot_branches": tot_branches,
        "sc_verdicts": sc_verdicts,
        "revision_count": state.get("revision_count", 0) + 1,
    }
