"""
RAPTOR — Recursive Abstractive Processing for Tree-Organized Retrieval.

3-level summary tree:
  Level 0: raw 512-token chunks (added during ingestion)
  Level 1: per-section summaries via Haiku
  Level 2: full paper summary via Haiku
"""
from llm_client import call_llm
from config import settings


class RAPTORIndexer:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder

    def build_tree(self, sections: dict, paper_id: str):
        skip = {"references", "acknowledgements", "acknowledgments", "bibliography"}
        section_summaries = {}
        for section_name, section_text in sections.items():
            if section_name in skip or len(section_text.strip()) < 50:
                continue
            summary = self._summarize_section(section_name, section_text)
            section_summaries[section_name] = summary
            self.collection.upsert(
                documents=[summary],
                metadatas=[{"paper_id": paper_id, "section": section_name, "level": 1, "type": "section_summary"}],
                ids=[f"{paper_id}_sec_{section_name[:20]}"]
            )
        if section_summaries:
            paper_summary = self._summarize_paper(section_summaries)
            self.collection.upsert(
                documents=[paper_summary],
                metadatas=[{"paper_id": paper_id, "level": 2, "type": "paper_summary"}],
                ids=[f"{paper_id}_summary"]
            )

    def retrieve(self, query: str, query_type: str, n_results: int = 10) -> dict:
        level_map = {"specific_fact": 0, "method_question": 0, "section_overview": 1, "paper_overview": 2, "comparison": 1}
        level = level_map.get(query_type, 0)
        return self.collection.query(query_texts=[query], n_results=n_results, where={"level": level})

    def _summarize_section(self, name: str, text: str) -> str:
        result = call_llm(
            model=settings.claude_haiku, max_tokens=250, temperature=0.0,
            messages=[{"role": "user", "content": f"Summarize this {name} section in 2-3 sentences:\n\n{text[:3000]}"}],
            agent_name="raptor_summary",
        )
        return result["content"][0].text

    def _summarize_paper(self, section_summaries: dict) -> str:
        combined = "\n\n".join(f"{k.upper()}: {v}" for k, v in section_summaries.items())
        result = call_llm(
            model=settings.claude_haiku, max_tokens=400, temperature=0.0,
            messages=[{"role": "user", "content": f"Write 4-sentence summary (contribution, methodology, results, limitations):\n\n{combined}"}],
            agent_name="raptor_summary",
        )
        return result["content"][0].text
