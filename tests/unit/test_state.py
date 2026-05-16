"""Unit tests for ResearchState design."""
import operator
from typing import get_type_hints, Annotated
from agents.state import ResearchState


def test_state_has_required_fields():
    hints = get_type_hints(ResearchState, include_extras=True)
    required = ["query", "session_id", "raw_papers", "retrieved_chunks",
                 "synthesis", "final_answer", "revision_count", "next"]
    for field in required:
        assert field in hints, f"Missing required field: {field}"


def test_raw_papers_uses_add_reducer():
    hints = get_type_hints(ResearchState, include_extras=True)
    raw_papers_hint = hints["raw_papers"]
    assert hasattr(raw_papers_hint, "__metadata__"), "raw_papers should use Annotated"
    assert raw_papers_hint.__metadata__[0] is operator.add, "raw_papers should use operator.add reducer"


def test_extracted_facts_uses_add_reducer():
    hints = get_type_hints(ResearchState, include_extras=True)
    facts_hint = hints["extracted_facts"]
    assert hasattr(facts_hint, "__metadata__"), "extracted_facts should use Annotated"
