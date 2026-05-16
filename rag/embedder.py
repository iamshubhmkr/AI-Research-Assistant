"""Sentence transformer embedder singleton (SPECTER2 for academic papers)."""
from sentence_transformers import SentenceTransformer

_embedder = None

def get_embedder(model_name: str = "allenai/specter2_base"):
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder
