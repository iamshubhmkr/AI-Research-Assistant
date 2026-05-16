"""
Semantic Chunker — Token-aware chunking for Level 0 RAPTOR leaves.

Why semantic chunking over fixed-size:
  Fixed 512 chunks split mid-sentence, mid-argument.
  Semantic chunking finds natural break points.

Token budget:
  Target: 400 tokens (leaves room for metadata)
  Max:    600 tokens (hard cap — avoids context bloat in reranking)
  Overlap: 50 tokens (prevents losing context at boundaries)
"""
import re


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def semantic_chunk(text: str, target_tokens: int = 400, max_tokens: int = 600, overlap_tokens: int = 50) -> list[dict]:
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current_lines, current_tokens = [], 0
    overlap_text = ""

    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if para_tokens > max_tokens:
            if current_lines:
                chunks.append(_make_chunk(overlap_text, current_lines, len(chunks)))
                overlap_text = _get_overlap(current_lines, overlap_tokens)
                current_lines, current_tokens = [], 0
            for sent_chunk in _split_sentences(para, target_tokens, max_tokens):
                chunks.append(_make_chunk(overlap_text, [sent_chunk], len(chunks)))
                overlap_text = sent_chunk[-overlap_tokens * 4:]
            continue
        if current_tokens + para_tokens > target_tokens and current_lines:
            chunks.append(_make_chunk(overlap_text, current_lines, len(chunks)))
            overlap_text = _get_overlap(current_lines, overlap_tokens)
            current_lines, current_tokens = [], 0
        current_lines.append(para)
        current_tokens += para_tokens

    if current_lines:
        chunks.append(_make_chunk(overlap_text, current_lines, len(chunks)))
    return chunks


def _split_sentences(text, target, max_t):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    groups, current, current_t = [], [], 0
    for s in sentences:
        st = estimate_tokens(s)
        if current_t + st > target and current:
            groups.append(" ".join(current))
            current, current_t = [], 0
        current.append(s)
        current_t += st
    if current:
        groups.append(" ".join(current))
    return groups


def _get_overlap(lines, overlap_tokens):
    combined = " ".join(lines)
    char_count = overlap_tokens * 4
    return combined[-char_count:] if len(combined) > char_count else ""


def _make_chunk(overlap, lines, index):
    text = " ".join(lines)
    if overlap:
        text = overlap + " " + text
    return {"text": text.strip(), "token_estimate": estimate_tokens(text), "chunk_index": index}


def chunk_paper_sections(sections: dict[str, str], paper_id: str) -> list[dict]:
    all_chunks = []
    skip = {"references", "acknowledgements", "acknowledgments", "bibliography"}
    for section_name, section_text in sections.items():
        if section_name in skip or len(section_text.strip()) < 50:
            continue
        chunks = semantic_chunk(section_text)
        for chunk in chunks:
            chunk["meta"] = {
                "paper_id": paper_id, "section": section_name,
                "level": 0, "type": "raw_chunk", "chunk_index": chunk["chunk_index"],
            }
            chunk["id"] = f"{paper_id}_L0_{section_name[:15]}_{chunk['chunk_index']}"
            all_chunks.append(chunk)
    return all_chunks
