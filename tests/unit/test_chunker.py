from rag.chunker import semantic_chunk, estimate_tokens, chunk_paper_sections


def test_estimate_tokens():
    assert estimate_tokens("hello world") == 2


def test_semantic_chunk_basic():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = semantic_chunk(text, target_tokens=20, max_tokens=50)
    assert len(chunks) >= 1
    assert all("text" in c and "chunk_index" in c for c in chunks)


def test_skips_references():
    sections = {"introduction": "Intro text " * 50, "references": "[1] Ref"}
    chunks = chunk_paper_sections(sections, "p1")
    assert "references" not in [c["meta"]["section"] for c in chunks]


def test_metadata():
    chunks = chunk_paper_sections({"methods": "Method text " * 50}, "p123")
    assert all(c["meta"]["paper_id"] == "p123" and c["meta"]["level"] == 0 for c in chunks)
