"""ChromaDB vector store singleton."""
import chromadb

_client = None
_collection = None

def get_collection(path: str = "./data/chroma"):
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=path)
        _collection = _client.get_or_create_collection(
            name="research_papers",
            metadata={"hnsw:space": "cosine"}
        )
    return _collection
