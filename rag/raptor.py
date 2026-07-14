"""
RAPTOR — 3-level summary tree (Haiku-built, temp=0 so no invention).
L0 raw chunks | L1 section summaries | L2 paper summary.
Embeddings are passed EXPLICITLY (Titan) so Chroma never falls back to its
local default embedding function.
"""
from llm_client import call_llm
from config import settings


class RAPTORIndexer:
    def __init__(self, collection, embedder):
        self.collection = collection
        self.embedder = embedder

    def build_tree(self, sections, paper_id):
        skip = {"references", "acknowledgements", "acknowledgments", "bibliography"}
        summaries = {}
        for name, body in sections.items():
            if name in skip or len(body.strip()) < 50:
                continue
            s = self._summarize_section(name, body)
            summaries[name] = s
            self.collection.upsert(documents=[s],
                embeddings=[self.embedder.encode(s)],
                metadatas=[{"paper_id": paper_id, "section": name,
                            "level": 1, "type": "section_summary"}],
                ids=[f"{paper_id}_sec_{name[:20]}"])
        if summaries:
            ps = self._summarize_paper(summaries)
            self.collection.upsert(documents=[ps],
                embeddings=[self.embedder.encode(ps)],
                metadatas=[{"paper_id": paper_id, "level": 2, "type": "paper_summary"}],
                ids=[f"{paper_id}_summary"])

    def retrieve(self, query, query_type, n_results=10):
        level = {"specific_fact": 0, "method_question": 0, "section_overview": 1,
                 "paper_overview": 2, "comparison": 1}.get(query_type, 0)
        return self.collection.query(query_embeddings=[self.embedder.encode(query)],
                                     n_results=n_results, where={"level": level})

    def _summarize_section(self, name, text):
        r = call_llm(model=settings.claude_haiku, max_tokens=250, temperature=0.0,
                     agent_name="raptor_summary",
                     messages=[{"role": "user", "content":
                         f"Summarize this {name} section in 2-3 sentences:\n\n{text[:3000]}"}])
        return r["content"][0].text

    def _summarize_paper(self, summaries):
        combined = "\n\n".join(f"{k.upper()}: {v}" for k, v in summaries.items())
        r = call_llm(model=settings.claude_haiku, max_tokens=400, temperature=0.0,
                     agent_name="raptor_summary",
                     messages=[{"role": "user", "content":
                         f"Write a 4-sentence summary (contribution, methodology, "
                         f"results, limitations):\n\n{combined}"}])
        return r["content"][0].text
