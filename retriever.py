"""
retriever.py — Embed a query with sentence-transformers and retrieve top-k chunks from ChromaDB.
"""

import os
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

_embed_model = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def retrieve(query: str, top_k: int = TOP_K) -> list[dict]:
    """
    Embed the query and return the top-k most relevant chunks.

    Returns a list of dicts with keys: text, source, chunk_index, distance.
    """
    model = get_embed_model()
    query_embedding = model.encode([query])[0].tolist()

    collection = get_collection()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for text, meta, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        chunks.append({
            "text": text,
            "source": meta.get("source", ""),
            "chunk_index": meta.get("chunk_index", -1),
            "distance": distance,
        })
    return chunks
