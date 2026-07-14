from rag.retriever import rrf_fuse


def _m(paper, section, idx):
    return {"paper_id": paper, "section": section, "chunk_index": idx, "level": 0}


def test_rrf_overlap_prefix_does_not_merge_chunks():
    """Chunks sharing an overlap prefix must stay distinct (v3 doc[:80] bug)."""
    shared = "OVERLAP " * 30
    dense = [(shared + "chunk one body", _m("p1", "methods", 0), 0.9),
             (shared + "chunk two body", _m("p1", "methods", 1), 0.8)]
    fused = rrf_fuse(dense, [])
    assert len(fused) == 2


def test_rrf_same_chunk_from_both_retrievers_merges():
    doc, meta = "some chunk text", _m("p1", "intro", 0)
    fused = rrf_fuse([(doc, meta, 0.9)], [(doc, meta, 7.0)])
    assert len(fused) == 1
    # appearing in both lists must score higher than appearing in one
    solo = rrf_fuse([(doc, meta, 0.9)], [])
    assert fused[0]["s"] > solo[0]["s"]


def test_rrf_orders_by_fused_score():
    a, b = _m("p1", "s", 0), _m("p1", "s", 1)
    dense = [("doc a", a, 0.9), ("doc b", b, 0.8)]
    sparse = [("doc a", a, 5.0)]
    fused = rrf_fuse(dense, sparse)
    assert fused[0]["doc"] == "doc a"
