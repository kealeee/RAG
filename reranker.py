"""
reranker.py — Rerank retrieved chunks using Cohere rerank-v3.
"""

import os
from dotenv import load_dotenv
import cohere

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
RERANK_MODEL = os.getenv("RERANK_MODEL", "rerank-v3.5")


def rerank(query: str, chunks: list[dict], top_n: int | None = None) -> list[dict]:
    """
    Rerank chunks using Cohere. Returns chunks sorted by relevance score (descending).
    Each chunk gets a 'rerank_score' field added.
    """
    if not chunks:
        return chunks

    top_n = top_n or len(chunks)
    client = cohere.ClientV2(api_key=COHERE_API_KEY)

    response = client.rerank(
        model=RERANK_MODEL,
        query=query,
        documents=[c["text"] for c in chunks],
        top_n=top_n,
    )

    reranked = []
    for result in response.results:
        chunk = dict(chunks[result.index])
        chunk["rerank_score"] = result.relevance_score
        reranked.append(chunk)

    return reranked
