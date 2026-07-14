"""Golden dataset — generation + production harvesting.

Design decision: the eval set should GROW from real traffic. Answers that
ship with high runtime faithfulness become candidate golden cases (marked
human_reviewed=False until someone vets them). Harvesting is best-effort —
an S3 hiccup must never fail a user request.
"""
import json
import logging
from llm_client import call_llm
from config import settings

logger = logging.getLogger(__name__)

QUESTION_TYPES = {
    "factual": "a specific verifiable number, name, date, or measurement",
    "methodological": "how they implemented or measured something",
    "comparative": "how this paper differs from prior work",
    "limitation": "a stated weakness or boundary condition",
    "implication": "what the findings mean for the field",
}

HARVEST_THRESHOLD = 0.92   # runtime faithfulness needed to become a candidate


def generate_per_paper(paper_text, paper_id, n_per_type=2):
    pairs = []
    for qtype, desc in QUESTION_TYPES.items():
        r = call_llm(model=settings.claude_sonnet, max_tokens=1500, temperature=0.1,
                     agent_name="golden_dataset",
                     messages=[{"role": "user", "content":
                         f"Paper:\n{paper_text[:5000]}\n\nGenerate {n_per_type} {qtype} "
                         f"questions ({desc}). JSON array with question, ground_truth, supporting_text."}])
        try:
            for p in json.loads(r["content"][0].text):
                p.update({"paper_id": paper_id, "type": qtype, "human_reviewed": False})
                pairs.append(p)
        except Exception:
            pass
    return pairs


def maybe_collect_from_production(query: str, final_answer: str,
                                  retrieved_chunks: list, faithfulness_score: float) -> bool:
    """Harvest a high-faithfulness production answer into the golden set (S3).

    Called from the API after a final answer ships. Best-effort: returns
    False (with a warning) on any failure instead of raising.
    """
    if not final_answer or faithfulness_score < HARVEST_THRESHOLD:
        return False
    try:
        from persistence.s3 import PaperStore
        PaperStore().append_golden({
            "question": query,
            "ground_truth": final_answer,
            "contexts": [c.get("text", "") for c in retrieved_chunks],
            "faithfulness_at_capture": faithfulness_score,
            "human_reviewed": False,
            "source": "production"})
        logger.info(f"golden dataset: harvested production case (faithfulness={faithfulness_score:.2f})")
        return True
    except Exception as e:
        logger.warning(f"golden dataset harvest failed ({e}); continuing")
        return False
