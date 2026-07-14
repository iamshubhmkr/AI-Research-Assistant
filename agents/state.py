"""
ResearchState — the shared notebook every agent reads and writes.
Rule of thumb: fields multiple agents append to use operator.add (merge);
fields where only the latest value matters overwrite normally.
"""
from typing import TypedDict, Annotated, Optional
import operator


class ResearchState(TypedDict):
    # Input
    query: str
    query_type: str            # specific_fact | comparison | section_overview | paper_overview
    session_id: str
    paper_urls: list[str]
    auto_search: bool

    # Fetcher
    raw_papers: Annotated[list[dict], operator.add]
    paper_ids: list[str]

    # Extractor
    extracted_facts: Annotated[list[dict], operator.add]
    sections: dict
    hyde_docs: list[str]

    # Retriever
    retrieved_chunks: list[dict]
    raptor_level: int
    retrieval_scores: list[float]

    # HITL 1
    human_chunk_feedback: Optional[str]
    chunks_approved: bool

    # Synthesizer
    synthesis: Optional[str]
    cot_reasoning: Optional[str]
    tot_branches: Optional[list]
    sc_verdicts: list[bool]

    # Critic
    critique: Optional[list]
    critique_severity: str
    needs_revision: bool
    reflexion_memory: Optional[str]
    faithfulness_score: float

    # HITL 2
    human_override: Optional[bool]
    human_revision_note: Optional[str]

    # Loop control
    revision_count: int
    next: str

    # Output
    final_answer: Optional[str]
    sources: list[str]
    ragas_scores: Optional[dict]
    token_usage: dict
    total_latency_ms: float
    estimated_cost_usd: float
