"""
Golden Dataset Generation — 3 complementary methods.

Method 1: RAGAS TestsetGenerator (automated, 3 difficulty levels)
Method 2: Structured per-paper Q&A via Claude (5 question types)
Method 3: Production auto-collection (faithfulness > 0.92)
"""
import json
from llm_client import call_llm
from config import settings

QUESTION_TYPES = {
    "factual": "a specific verifiable number, name, date, or measurement",
    "methodological": "how they implemented or measured something",
    "comparative": "how this paper differs from prior work",
    "limitation": "a stated weakness or boundary condition",
    "implication": "what the findings mean for the broader field",
}


def generate_per_paper(paper_text: str, paper_id: str, n_per_type: int = 2) -> list:
    all_pairs = []
    for qtype, description in QUESTION_TYPES.items():
        result = call_llm(
            model=settings.claude_sonnet, max_tokens=1500, temperature=0.1,
            messages=[{"role": "user", "content":
                f"Paper text:\n{paper_text[:5000]}\n\n"
                f"Generate {n_per_type} {qtype} questions ({description}).\n"
                f"For each: question, ground_truth, supporting_text.\n"
                f'Return JSON array: [{{"question":"...","ground_truth":"...","supporting_text":"..."}}]'
            }],
            agent_name="golden_dataset",
        )
        try:
            pairs = json.loads(result["content"][0].text)
            for p in pairs:
                p.update({"paper_id": paper_id, "type": qtype, "human_reviewed": False})
            all_pairs.extend(pairs)
        except Exception:
            pass
    return all_pairs


def maybe_collect_from_production(answer: dict, threshold: float = 0.92) -> bool:
    if answer.get("ragas_scores", {}).get("faithfulness", 0) >= threshold:
        from persistence.s3 import PaperStore
        PaperStore().append_golden({
            "question": answer["query"], "ground_truth": answer["final_answer"],
            "contexts": [c.get("text", "") for c in answer.get("retrieved_chunks", [])],
            "source": "production", "faithfulness": answer["ragas_scores"]["faithfulness"]
        })
        return True
    return False
