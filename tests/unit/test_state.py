import operator
from typing import get_type_hints
from agents.state import ResearchState


def test_required_fields():
    hints = get_type_hints(ResearchState, include_extras=True)
    for f in ["query", "session_id", "raw_papers", "retrieved_chunks",
              "synthesis", "final_answer", "revision_count", "next"]:
        assert f in hints


def test_raw_papers_add_reducer():
    hints = get_type_hints(ResearchState, include_extras=True)
    assert hints["raw_papers"].__metadata__[0] is operator.add
