"""
HyDE — Hypothetical Document Embeddings.

Problem: User queries use casual vocabulary. Papers use formal vocabulary.
Solution: Generate what the ideal answer paragraph looks like in academic language.
Typical improvement: Context Recall 0.71 → 0.89 (+25%).
"""
from llm_client import call_llm
from config import settings

HYDE_PROMPT = """You are an expert academic researcher writing a paragraph
that would appear in a peer-reviewed paper.

Write a 150-200 word paragraph that directly answers the research question below,
using formal academic vocabulary and technical terminology.
This paragraph will be used for document retrieval, not shown to users."""


def generate_hypothetical_document(query: str) -> str:
    result = call_llm(
        model=settings.claude_sonnet,
        messages=[{"role": "user", "content": f"{HYDE_PROMPT}\n\nResearch question: {query}"}],
        max_tokens=350,
        temperature=0.3,
        agent_name="hyde",
    )
    return result["content"][0].text


def hyde_embed(query: str, embedder, cache) -> list:
    cache_key = f"hyde:{hash(query)}"
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    hypo_doc = generate_hypothetical_document(query)
    embedding = embedder.encode(hypo_doc).tolist()
    cache.set(cache_key, embedding, ttl=settings.cache_ttl_embedding)
    return embedding
