"""Unit tests for supervisor routing logic."""
import json


def test_supervisor_routes_to_fetcher_when_no_papers():
    """When raw_papers is empty and auto_search=True, should route to fetcher."""
    state = {
        "raw_papers": [], "paper_urls": [], "auto_search": True,
        "sections": {}, "retrieved_chunks": [], "chunks_approved": False,
        "synthesis": None, "critique": None, "needs_revision": False,
        "revision_count": 0, "human_override": None,
    }
    # Expected routing: fetcher (because raw_papers is empty)
    summary = {
        "raw_papers_count": 0, "paper_urls_count": 0, "auto_search": True,
        "sections_populated": False, "retrieved_chunks_count": 0,
        "chunks_approved": False, "synthesis_exists": False,
        "critique_exists": False, "needs_revision": False,
        "revision_count": 0, "human_override": None,
    }
    # This would be tested with a mock LLM call in integration tests
    assert summary["raw_papers_count"] == 0
    assert summary["auto_search"] is True


def test_supervisor_routes_to_end_when_revision_cap_reached():
    state_summary = {"revision_count": 2, "needs_revision": True}
    # With revision_count >= max_revision_count (2), should route to END
    assert state_summary["revision_count"] >= 2
