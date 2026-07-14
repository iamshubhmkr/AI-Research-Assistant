"""
Extractor — five jobs per document:
1) section split  2) CoT fact extraction  3) Level-0 chunking
4) RAPTOR tree (L1+L2)  5) HyDE document generation
v3.1: chunk embeddings are computed explicitly via Titan (through the L2
cache) so Chroma never instantiates a local default embedding model.
"""
import json
import re
import logging
from llm_client import call_llm
from utils import stable_id
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

COT_EXTRACT_SYSTEM = """Extract structured info from a research paper.
Reason through: 1) abstract/intro for contribution  2) headings for structure
3) results for metrics with context  4) limitations  5) structure as JSON.
Return ONLY JSON:
{"contribution": "...", "methodology": "...",
 "key_results": [{"metric": "...", "value": "...", "context": "..."}],
 "limitations": ["..."], "key_claims": ["..."]}"""


def split_into_sections(markdown_text: str) -> dict:
    sections, current, lines = {}, "abstract", []
    for line in markdown_text.split("\n"):
        if re.match(r"^#{1,3}\s+", line):
            if lines:
                sections[current] = "\n".join(lines).strip()
            current = re.sub(r"^#{1,3}\s+", "", line).strip().lower().replace(" ", "_")[:30]
            lines = []
        else:
            lines.append(line)
    if lines:
        sections[current] = "\n".join(lines).strip()
    return sections


def extract_facts_cot(paper_text: str) -> dict:
    result = call_llm(model=settings.claude_sonnet,
                      messages=[{"role": "user", "content": f"Paper text:\n\n{paper_text[:8000]}"}],
                      system=COT_EXTRACT_SYSTEM, max_tokens=2000, temperature=0.0,
                      agent_name="extractor")
    raw = result["content"][0].text
    try:
        return json.loads(re.search(r"\{.*\}", raw, re.DOTALL).group())
    except Exception:
        return {"contribution": raw[:200], "key_results": [], "limitations": []}


def extractor_node(state: ResearchState) -> dict:
    from rag.vector_store import get_collection
    from rag.embedder import get_embedder, embed_texts
    from rag.raptor import RAPTORIndexer
    from rag.chunker import chunk_paper_sections
    from rag.hyde import generate_hypothetical_document
    from cache.embedding_cache import EmbeddingCache

    collection = get_collection()
    embedder = get_embedder()
    raptor = RAPTORIndexer(collection, embedder)
    emb_cache = EmbeddingCache()
    all_sections, all_facts = {}, []

    for paper in state.get("raw_papers", []):
        text = paper.get("text", "")
        if not text.strip():
            continue
        paper_id = stable_id(paper.get("url") or text[:200])

        sections = split_into_sections(text)
        all_sections[paper_id] = sections

        facts = extract_facts_cot(text)
        all_facts.append({"paper_id": paper_id, "facts": facts, "url": paper.get("url", "")})

        chunks = chunk_paper_sections(sections, paper_id)
        if chunks:
            collection.upsert(documents=[c["text"] for c in chunks],
                              embeddings=embed_texts([c["text"] for c in chunks], emb_cache),
                              metadatas=[c["meta"] for c in chunks],
                              ids=[c["id"] for c in chunks])
            logger.info(f"[extractor] indexed {len(chunks)} L0 chunks for {paper_id[:10]}")

        raptor.build_tree(sections, paper_id)

    hyde_doc = generate_hypothetical_document(state["query"])
    return {"sections": all_sections, "extracted_facts": all_facts, "hyde_docs": [hyde_doc]}
