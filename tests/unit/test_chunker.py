"""Unit tests for semantic chunker."""
from rag.chunker import semantic_chunk, estimate_tokens, chunk_paper_sections


def test_estimate_tokens():
    assert estimate_tokens("hello world") == 2  # 11 chars / 4 ≈ 2


def test_semantic_chunk_basic():
    text = "First paragraph about methods.\n\nSecond paragraph about results.\n\nThird paragraph about conclusions."
    chunks = semantic_chunk(text, target_tokens=20, max_tokens=50)
    assert len(chunks) >= 1
    assert all("text" in c for c in chunks)
    assert all("chunk_index" in c for c in chunks)


def test_semantic_chunk_overlap():
    text = ("A " * 200 + "\n\n" + "B " * 200)
    chunks = semantic_chunk(text, target_tokens=100, max_tokens=150, overlap_tokens=20)
    assert len(chunks) >= 2


def test_chunk_paper_sections_skips_references():
    sections = {"introduction": "Some intro text " * 50, "references": "[1] Some reference"}
    chunks = chunk_paper_sections(sections, "test_paper")
    section_names = [c["meta"]["section"] for c in chunks]
    assert "references" not in section_names


def test_chunk_paper_sections_metadata():
    sections = {"methods": "Method description " * 50}
    chunks = chunk_paper_sections(sections, "paper_123")
    assert all(c["meta"]["paper_id"] == "paper_123" for c in chunks)
    assert all(c["meta"]["level"] == 0 for c in chunks)
    assert all(c["meta"]["type"] == "raw_chunk" for c in chunks)
