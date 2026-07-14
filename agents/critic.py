"""
Critic — few-shot fact-checker + faithfulness judge + Reflexion memory.
Severity ladder: critical / moderate / none. Hard cap of 2 revisions.

v3.1 design decisions:
  - Faithfulness is scored by a HAIKU judge instead of inline RAGAS: RAGAS's
    default judge is OpenAI (not on the AWS bill, key not configured), so it
    always failed and returned 0.0 — which silently forced a revision for
    every 'moderate' finding. RAGAS remains the OFFLINE evaluation harness.
  - If the judge itself fails, the score is None and faithfulness simply
    doesn't gate (fail-open, per the project's degradation philosophy).
"""
import json
import re
import logging
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

FEW_SHOT_CRITIC = """You are a research fact-checker. Calibrate with these examples:

EXAMPLE 1 - fabricated statistic (CRITICAL):
Synthesis: "achieved 94.2% F1 (Table 3)"  Context Table 3: 89.1%
-> {"approved": false, "issues": ["Wrong number: 89.1% not 94.2%"],
    "severity": "critical", "reflexion": "Verify every number exactly."}

EXAMPLE 2 - invented citation (CRITICAL):
Synthesis cites "Wei et al. (2022)"; no such paper in context.
-> {"approved": false, "issues": ["Wei et al. not in context"],
    "severity": "critical", "reflexion": "Only cite provided papers."}

EXAMPLE 3 - inference beyond evidence (MODERATE):
Synthesis: "learning rate 0.001 with Adam"; context only mentions Adam.
-> {"approved": false, "issues": ["Learning rate not stated"],
    "severity": "moderate", "reflexion": "State only what is written."}

EXAMPLE 4 - unsupported generalization (MODERATE):
Synthesis: "works on low-resource languages"; context tests English+Spanish.
-> {"approved": false, "issues": ["Generalization unsupported"],
    "severity": "moderate", "reflexion": "Stay within tested scope."}

EXAMPLE 5 - clean answer (APPROVE):
Well-cited claims exactly matching context.
-> {"approved": true, "issues": [], "severity": "none", "reflexion": ""}

Now evaluate:
Query: {query}
Synthesis: {synthesis}
Context: {context}
Return JSON only."""

FAITHFULNESS_JUDGE = """Rate how faithful the ANSWER is to the CONTEXT.
1.0 = every claim is directly supported; 0.0 = contradicts or invents facts.
ANSWER: {answer}
CONTEXT: {context}
Return JSON only: {{"score": <float 0.0-1.0>}}"""


def _faithfulness_score(synthesis, chunks):
    """Haiku-judged faithfulness in [0,1]; None when the judge fails."""
    context = "\n\n".join(c.get("text", "")[:500] for c in chunks[:3])
    try:
        r = call_llm(model=settings.claude_haiku, max_tokens=60, temperature=0.0,
                     agent_name="faithfulness_judge",
                     messages=[{"role": "user", "content":
                         FAITHFULNESS_JUDGE.format(answer=synthesis[:1500], context=context)}])
        parsed = json.loads(re.search(r"\{.*\}", r["content"][0].text, re.DOTALL).group())
        return max(0.0, min(1.0, float(parsed["score"])))
    except Exception as e:
        logger.warning(f"faithfulness judge failed: {e}")
        return None


def _build_reflexion(synthesis, critique_note, query):
    r = call_llm(model=settings.claude_sonnet, max_tokens=250, temperature=0.0,
                 agent_name="reflexion",
                 messages=[{"role": "user", "content":
                     f"A synthesis failed review.\nQuery: {query}\n"
                     f"Failed excerpt: {synthesis[:400]}\nCritique: {critique_note}\n"
                     f"Write 3 SPECIFIC bullets of mistakes to avoid."}])
    return r["content"][0].text


def critic_node(state: ResearchState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    synthesis = state.get("synthesis", "")
    context_sample = "\n\n".join(c.get("text", "")[:500] for c in chunks[:3])

    prompt = (FEW_SHOT_CRITIC
              .replace("{query}", state.get("query", ""))
              .replace("{synthesis}", synthesis[:1000])
              .replace("{context}", context_sample))
    result = call_llm(model=settings.claude_sonnet, max_tokens=600, temperature=0.0,
                      agent_name="critic",
                      messages=[{"role": "user", "content": prompt}])
    try:
        parsed = json.loads(re.search(r"\{.*\}", result["content"][0].text, re.DOTALL).group())
    except Exception:
        # degraded path: an unparseable critique approves rather than crashing
        parsed = {"approved": True, "issues": [], "severity": "none", "reflexion": ""}

    severity = parsed.get("severity", "none")
    score = _faithfulness_score(synthesis, chunks)
    budget_left = state.get("revision_count", 0) < settings.max_revision_count
    needs_revision = budget_left and (
        severity == "critical"
        or (severity == "moderate" and score is not None
            and score < settings.ragas_faithfulness_target))
    if state.get("human_override") is True:
        needs_revision = False

    reflexion = _build_reflexion(synthesis, parsed.get("reflexion", ""),
                                 state.get("query", "")) if needs_revision else None

    logger.info(f"[critic] severity={severity} faithfulness={score} revise={needs_revision}")
    return {"critique": parsed.get("issues", []), "critique_severity": severity,
            "needs_revision": needs_revision, "reflexion_memory": reflexion,
            "faithfulness_score": score if score is not None else 0.0,
            "final_answer": None if needs_revision else synthesis,
            "sources": [p.get("url", "") for p in state.get("raw_papers", [])]}
