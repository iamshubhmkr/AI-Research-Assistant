"""
Supervisor Agent — LLM-driven dynamic routing.

Uses Haiku for routing (not Sonnet):
- Simple classification: 8 state fields -> 1 of 7 targets
- Haiku is 10x cheaper and 3x faster
- Supervisor runs after EVERY node — cost adds up with Sonnet
- Temperature=0.0 for deterministic routing

Interview talking points:
  "Haiku for supervisor because it's a classification task, not reasoning.
   10x cheaper, 3x faster. Runs 6-8 times per query."
  "LLM routing handles unexpected states gracefully. Static if/else breaks
   on edge cases. LLM reasons: 'sections populated but chunks empty, so retriever'."
"""
import json
import re
import logging
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = """You are the supervisor of a research pipeline.
Inspect the current state and decide which agent should run next.

AGENTS AVAILABLE:
- fetcher:     Searches arXiv and fetches paper content. Run when raw_papers is empty.
- extractor:   Parses PDFs into sections and generates HyDE docs. Run after fetch.
- retriever:   Runs hybrid RAG retrieval. Run after extraction.
- synthesizer: Synthesizes the answer from retrieved chunks. ONLY if chunks_approved=True.
- critic:      Fact-checks the synthesis. Run after synthesizer.
- END:         Return final answer. Run when:
               (a) critic approves (needs_revision=False), OR
               (b) revision_count >= {max_revisions}, OR
               (c) chunks_approved=False (HITL pause — awaiting human)

DECISION RULES (in priority order):
1. raw_papers=[] AND auto_search=True → fetcher
2. raw_papers=[] AND paper_urls not empty → fetcher
3. raw_papers populated AND sections={{}} → extractor
4. sections populated AND retrieved_chunks=[] → retriever
5. retrieved_chunks populated AND chunks_approved=False → END  (HITL pause)
6. chunks_approved=True AND synthesis=None → synthesizer
7. synthesis exists AND critique=None → critic
8. needs_revision=True AND revision_count < {max_revisions} → synthesizer
9. needs_revision=False OR revision_count >= {max_revisions} → END

Current state:
{state_summary}

Return JSON only, no explanation:
{{"next": "<agent_name>", "reason": "<one sentence why>"}}"""


def supervisor_node(state: ResearchState) -> dict:
    summary = {
        "raw_papers_count":       len(state.get("raw_papers", [])),
        "paper_urls_count":       len(state.get("paper_urls", [])),
        "auto_search":            state.get("auto_search", True),
        "sections_populated":     bool(state.get("sections")),
        "retrieved_chunks_count": len(state.get("retrieved_chunks", [])),
        "chunks_approved":        state.get("chunks_approved", False),
        "synthesis_exists":       state.get("synthesis") is not None,
        "critique_exists":        state.get("critique") is not None,
        "needs_revision":         state.get("needs_revision", False),
        "revision_count":         state.get("revision_count", 0),
        "human_override":         state.get("human_override"),
    }

    prompt = SUPERVISOR_PROMPT.format(
        max_revisions=settings.max_revision_count,
        state_summary=json.dumps(summary, indent=2)
    )

    try:
        result = call_llm(
            model=settings.claude_haiku,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150,
            temperature=0.0,
            agent_name="supervisor",
        )
        raw = result["content"][0].text
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
        next_node = parsed.get("next", "END")
        logger.info(f"[supervisor] → {next_node}: {parsed.get('reason', '')}")
    except Exception as e:
        logger.error(f"[supervisor] routing error: {e} — defaulting to END")
        next_node = "END"

    return {"next": next_node}
