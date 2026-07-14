"""ChromaDB singleton — one persistent client/collection per process."""
import chromadb

_collection = None

def get_collection(path: str = "./data/chroma"):
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=path)
        _collection = client.get_or_create_collection(
            name="research_papers", metadata={"hnsw:space": "cosine"})
    return _collection
