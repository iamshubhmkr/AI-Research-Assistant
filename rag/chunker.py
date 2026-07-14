"""
Semantic chunker — Level-0 RAPTOR leaves.
Targets ~400 tokens, max 600, 50-token overlap; cuts at paragraph boundaries.
"""
import re


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def semantic_chunk(text, target_tokens=400, max_tokens=600, overlap_tokens=50):
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, current, current_tokens, overlap = [], [], 0, ""
    for para in paragraphs:
        pt = estimate_tokens(para)
        if pt > max_tokens:
            if current:
                chunks.append(_mk(overlap, current, len(chunks)))
                overlap = _tail(current, overlap_tokens)
                current, current_tokens = [], 0
            for piece in _split_sentences(para, target_tokens):
                chunks.append(_mk(overlap, [piece], len(chunks)))
                overlap = piece[-overlap_tokens * 4:]
            continue
        if current_tokens + pt > target_tokens and current:
            chunks.append(_mk(overlap, current, len(chunks)))
            overlap = _tail(current, overlap_tokens)
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += pt
    if current:
        chunks.append(_mk(overlap, current, len(chunks)))
    return chunks


def _split_sentences(text, target):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    groups, cur, t = [], [], 0
    for s in sentences:
        st = estimate_tokens(s)
        if t + st > target and cur:
            groups.append(" ".join(cur))
            cur, t = [], 0
        cur.append(s)
        t += st
    if cur:
        groups.append(" ".join(cur))
    return groups


def _tail(lines, overlap_tokens):
    combined = " ".join(lines)
    n = overlap_tokens * 4
    return combined[-n:] if len(combined) > n else ""


def _mk(overlap, lines, idx):
    text = ((overlap + " ") if overlap else "") + " ".join(lines)
    return {"text": text.strip(), "token_estimate": estimate_tokens(text), "chunk_index": idx}


def chunk_paper_sections(sections, paper_id):
    skip = {"references", "acknowledgements", "acknowledgments", "bibliography"}
    out = []
    for name, body in sections.items():
        if name in skip or len(body.strip()) < 50:
            continue
        for ch in semantic_chunk(body):
            ch["meta"] = {"paper_id": paper_id, "section": name, "level": 0,
                          "type": "raw_chunk", "chunk_index": ch["chunk_index"]}
            ch["id"] = f"{paper_id}_L0_{name[:15]}_{ch['chunk_index']}"
            out.append(ch)
    return out
