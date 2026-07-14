"""
Supervisor — LLM-driven routing on Haiku (cheap, fast, deterministic).
Runs after every node; same state -> same decision (temp=0).
Defaults to END on any error so a glitch can never loop forever.

HITL note: when chunks exist but aren't approved, the supervisor routes to
SYNTHESIZER — the graph's interrupt_before=["synthesizer"] is what pauses
for human review. (Routing to END here would TERMINATE the thread and make
it impossible to resume; that was a v3 bug.)
"""
import json
import re
import logging
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

VALID_ROUTES = {"fetcher", "extractor", "retriever", "synthesizer", "critic", "END"}

SUPERVISOR_PROMPT = """You are the supervisor of a research pipeline.
Decide which agent runs next based on the state.

AGENTS:
- fetcher: search arXiv / fetch documents. Run when raw_papers empty.
- extractor: parse docs, build index. Run after fetch.
- retriever: hybrid retrieval. Run after extraction.
- synthesizer: write the answer. The graph PAUSES before this node for human
  chunk approval, so route here as soon as chunks are retrieved.
- critic: fact-check the synthesis. Run after synthesizer.
- END: finish. When the critic approved, OR revision_count >= {max_revisions},
       OR nothing was fetched/retrieved and no agent can make progress.

RULES (priority order):
1. raw_papers empty AND (auto_search OR paper_urls) -> fetcher
2. raw_papers present AND sections empty -> extractor
3. sections present AND retrieved_chunks empty -> retriever
4. retrieved_chunks present AND synthesis is None -> synthesizer
   (graph interrupts before synthesizer until a human approves the chunks)
5. synthesis present AND critique is None -> critic
6. needs_revision AND revision_count < {max_revisions} -> synthesizer
7. otherwise -> END

State:
{state_summary}

Return JSON only: {{"next": "<agent>", "reason": "<one sentence>"}}"""


def supervisor_node(state: ResearchState) -> dict:
    summary = {
        "raw_papers_count": len(state.get("raw_papers", [])),
        "paper_urls_count": len(state.get("paper_urls", [])),
        "auto_search": state.get("auto_search", True),
        "sections_populated": bool(state.get("sections")),
        "retrieved_chunks_count": len(state.get("retrieved_chunks", [])),
        "chunks_approved": state.get("chunks_approved", False),
        "synthesis_exists": state.get("synthesis") is not None,
        "critique_exists": state.get("critique") is not None,
        "needs_revision": state.get("needs_revision", False),
        "revision_count": state.get("revision_count", 0),
        "human_override": state.get("human_override"),
    }
    prompt = SUPERVISOR_PROMPT.format(max_revisions=settings.max_revision_count,
                                      state_summary=json.dumps(summary, indent=2))
    try:
        result = call_llm(model=settings.claude_haiku,
                          messages=[{"role": "user", "content": prompt}],
                          max_tokens=150, temperature=0.0, agent_name="supervisor")
        parsed = json.loads(re.search(r"\{.*\}", result["content"][0].text, re.DOTALL).group())
        next_node = parsed.get("next", "END")
        if next_node not in VALID_ROUTES:
            logger.warning(f"[supervisor] invalid route '{next_node}' — defaulting to END")
            next_node = "END"
        logger.info(f"[supervisor] -> {next_node}: {parsed.get('reason','')}")
    except Exception as e:
        logger.error(f"[supervisor] error: {e} — defaulting to END")
        next_node = "END"
    return {"next": next_node}
