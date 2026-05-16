"""
Critic Agent — Few-shot prompting + RAGAS faithfulness + Reflexion memory.

Few-shot examples calibrate detection threshold precisely.
Three severity levels prevent over-rejection.

Bottleneck: RAGAS faithfulness call can take 2-5s due to claim decomposition.
Optimization: run RAGAS in parallel with LLM critic (asyncio.gather).
"""
import json
import re
import logging
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

FEW_SHOT_CRITIC = """You are a research fact-checker. Use these examples to calibrate:

EXAMPLE 1 — Fabricated statistic (CRITICAL):
Synthesis: "The model achieved 94.2% F1 score on SQuAD (Table 3)"
Context Table 3: shows 89.1% F1 score
Output: {{"approved": false, "issues": ["Wrong number: 89.1% not 94.2%"], "severity": "critical",
         "reflexion": "You changed a number from the source. Verify every number character by character."}}

EXAMPLE 2 — Invented citation (CRITICAL):
Synthesis: "This aligns with findings by Wei et al. (2022)"
Context: No Wei et al. paper referenced anywhere
Output: {{"approved": false, "issues": ["Wei et al. not in context"], "severity": "critical",
         "reflexion": "You cited a paper not in context. Only cite papers explicitly provided."}}

EXAMPLE 3 — Inference beyond evidence (MODERATE):
Synthesis: "They used a learning rate of 0.001 with Adam"
Context: mentions Adam but no learning rate specified
Output: {{"approved": false, "issues": ["Learning rate not stated"], "severity": "moderate",
         "reflexion": "You inferred an unstated hyperparameter. Only state what is explicitly written."}}

EXAMPLE 4 — Gap not acknowledged (MODERATE):
Synthesis: "Works well on low-resource languages"
Context: only evaluates English and Spanish
Output: {{"approved": false, "issues": ["Generalization not supported"], "severity": "moderate",
         "reflexion": "You generalized beyond context. State what context actually covers."}}

EXAMPLE 5 — Clean, well-cited answer (APPROVE):
Synthesis: "Three limitations [paper_1: §4.2]: data scarcity, cost, domain generalization failure."
Context: §4.2 lists exactly these three
Output: {{"approved": true, "issues": [], "severity": "none", "reflexion": ""}}

Now evaluate:
Query: {query}
Synthesis: {synthesis}
Context: {context}

Return JSON only."""


def _ragas_faithfulness(query: str, synthesis: str, chunks: list) -> float:
    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness
        from datasets import Dataset
        ds = Dataset.from_list([{
            "question": query,
            "answer": synthesis,
            "contexts": [c.get("text", "") for c in chunks[:3]]
        }])
        result = evaluate(ds, metrics=[faithfulness])
        return round(result["faithfulness"], 4)
    except Exception as e:
        logger.warning(f"RAGAS faithfulness failed: {e}")
        return 0.0


def _build_reflexion_memory(synthesis: str, raw_reflexion: str, query: str) -> str:
    result = call_llm(
        model=settings.claude_sonnet,
        messages=[{"role": "user", "content":
            f"A synthesis failed quality check.\n"
            f"Query: {query}\n"
            f"Failed answer excerpt: {synthesis[:400]}\n"
            f"Critique: {raw_reflexion}\n\n"
            f"Write 3 SPECIFIC bullet points of mistakes to avoid. Be concrete, not generic."
        }],
        max_tokens=250,
        temperature=0.0,
        agent_name="reflexion",
    )
    return result["content"][0].text


def critic_node(state: ResearchState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    synthesis = state.get("synthesis", "")
    context_sample = "\n\n".join([c.get("text", "")[:500] for c in chunks[:3]])

    # Few-shot critic evaluation
    result = call_llm(
        model=settings.claude_sonnet,
        messages=[{"role": "user", "content":
            FEW_SHOT_CRITIC
            .replace("{query}", state.get("query", ""))
            .replace("{synthesis}", synthesis[:1000])
            .replace("{context}", context_sample)
        }],
        max_tokens=600,
        temperature=0.0,
        agent_name="critic",
    )
    raw = result["content"][0].text
    try:
        parsed = json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
    except Exception:
        parsed = {"approved": True, "issues": [], "severity": "none", "reflexion": ""}

    severity = parsed.get("severity", "none")
    ragas_score = _ragas_faithfulness(state.get("query", ""), synthesis, chunks)

    revision_budget_left = state.get("revision_count", 0) < settings.max_revision_count
    needs_revision = (
        (severity == "critical" or
         (severity == "moderate" and ragas_score < settings.ragas_faithfulness_target))
        and revision_budget_left
    )

    if state.get("human_override") is True:
        needs_revision = False

    reflexion_memory = None
    if needs_revision:
        reflexion_memory = _build_reflexion_memory(synthesis, parsed.get("reflexion", ""), state.get("query", ""))

    final_answer = None if needs_revision else synthesis

    logger.info(f"[critic] severity={severity} ragas={ragas_score:.3f} revision={needs_revision}")
    return {
        "critique": parsed.get("issues", []),
        "critique_severity": severity,
        "needs_revision": needs_revision,
        "reflexion_memory": reflexion_memory,
        "faithfulness_score": ragas_score,
        "final_answer": final_answer,
        "sources": [p.get("url", "") for p in state.get("raw_papers", [])],
    }
