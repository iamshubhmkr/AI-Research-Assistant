"""
Extractor Agent — CoT extraction + HyDE + RAPTOR tree + Level 0 chunking.

Four jobs:
1. Parse PDF markdown into named sections
2. Extract structured facts using CoT
3. Chunk sections into Level 0 RAPTOR leaves (via semantic chunker)
4. Build RAPTOR tree (Level 1 + Level 2 summaries)
5. Generate HyDE hypothetical document for retrieval
"""
import json
import re
import logging
from llm_client import call_llm
from .state import ResearchState
from config import settings

logger = logging.getLogger(__name__)

COT_EXTRACT_SYSTEM = """You are extracting structured information from a research paper.

Use this reasoning chain before responding:
Step 1: Read the abstract and introduction for the main contribution
Step 2: Scan section headings for paper structure
Step 3: Find results/evaluation — extract numerical metrics with context
Step 4: Find limitations/conclusion — list stated weaknesses
Step 5: Structure into JSON

Return ONLY valid JSON:
{
  "contribution": "one sentence summary",
  "methodology": "2-3 sentences",
  "key_results": [{"metric": "name", "value": "X", "context": "on dataset Y"}],
  "limitations": ["limitation 1", "limitation 2"],
  "key_claims": ["claim 1 with citation location"]
}"""


def split_into_sections(markdown_text: str) -> dict:
    sections = {}
    current_section = "abstract"
    current_lines = []
    for line in markdown_text.split("\n"):
        if re.match(r'^#{1,3}\s+', line):
            if current_lines:
                sections[current_section] = "\n".join(current_lines).strip()
            section_name = re.sub(r'^#{1,3}\s+', '', line).strip()
            current_section = section_name.lower().replace(" ", "_")[:30]
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections[current_section] = "\n".join(current_lines).strip()
    return sections


def extract_facts_cot(paper_text: str) -> dict:
    result = call_llm(
        model=settings.claude_sonnet,
        messages=[{"role": "user", "content": f"Paper text:\n\n{paper_text[:8000]}"}],
        system=COT_EXTRACT_SYSTEM,
        max_tokens=2000,
        temperature=0.0,
        agent_name="extractor",
    )
    raw = result["content"][0].text
    try:
        return json.loads(re.search(r'\{.*\}', raw, re.DOTALL).group())
    except Exception:
        return {"contribution": raw[:200], "key_results": [], "limitations": []}


def extractor_node(state: ResearchState) -> dict:
    all_sections = {}
    all_facts = []

    from rag.vector_store import get_collection
    from rag.embedder import get_embedder
    from rag.raptor import RAPTORIndexer
    from rag.chunker import chunk_paper_sections
    from rag.hyde import generate_hypothetical_document

    collection = get_collection()
    embedder = get_embedder()
    raptor = RAPTORIndexer(collection, embedder)

    for paper in state.get("raw_papers", []):
        text = paper.get("text", "")
        paper_id = str(hash(paper.get("url", text[:50])))

        # 1. Section splitting
        sections = split_into_sections(text)
        all_sections[paper_id] = sections

        # 2. CoT fact extraction
        facts = extract_facts_cot(text)
        all_facts.append({"paper_id": paper_id, "facts": facts, "url": paper.get("url", "")})

        # 3. Level 0: semantic chunking → ChromaDB
        chunks = chunk_paper_sections(sections, paper_id)
        if chunks:
            collection.upsert(
                documents=[c["text"] for c in chunks],
                metadatas=[c["meta"] for c in chunks],
                ids=[c["id"] for c in chunks],
            )
            logger.info(f"[extractor] indexed {len(chunks)} L0 chunks for paper {paper_id[:12]}")

        # 4. Level 1+2: RAPTOR summaries
        raptor.build_tree(sections, paper_id)

    # 5. HyDE document
    hyde_doc = generate_hypothetical_document(state["query"])

    return {
        "sections": all_sections,
        "extracted_facts": all_facts,
        "hyde_docs": [hyde_doc],
    }
