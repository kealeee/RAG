"""
rag.py — RAG chain with query rewriting, Cohere reranking, and trace logging.

Pipeline:
  1. Retrieve top-k chunks
  2. If top similarity < REWRITE_THRESHOLD → rewrite query + retry (max 1 retry)
  3. Rerank with Cohere rerank-v3
  4. Generate answer with flan-t5
  5. Log trace
"""

import os
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from retriever import retrieve
from reranker import rerank
import tracer

load_dotenv()

LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-small")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))
TOP_K = int(os.getenv("TOP_K", "3"))
REWRITE_THRESHOLD = float(os.getenv("REWRITE_THRESHOLD", "0.5"))

_tokenizer = None
_model = None


def get_model():
    global _tokenizer, _model
    if _model is None:
        print(f"Loading LLM: {LLM_MODEL} ...")
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        _model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL)
    return _tokenizer, _model


def rewrite_query(question: str) -> str:
    """Use the LLM to rewrite the query for better retrieval."""
    tokenizer, model = get_model()
    prompt = (
        f"Rewrite the following question to be more specific and detailed "
        f"for searching a document knowledge base. "
        f"Return only the rewritten question.\n\n"
        f"Question: {question}\n\nRewritten question:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    outputs = model.generate(**inputs, max_new_tokens=64)
    rewritten = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    # Fall back to original if rewrite is empty or too short
    return rewritten if len(rewritten) > 5 else question


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {c['text']}" for i, c in enumerate(chunks)
    )
    return (
        f"Answer the question based only on the context below. "
        f"If the context doesn't contain the answer, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def query(question: str, top_k: int = TOP_K) -> dict:
    """
    Full RAG pipeline with query rewriting, reranking, and tracing.

    Returns:
        {
            "answer": str,
            "sources": list[str],
            "chunks": list[dict],
            "trace": dict,
        }
    """
    # Step 1: Initial retrieval
    chunks = retrieve(question, top_k=top_k)
    was_rewritten = False
    rewritten_query = None

    if not chunks:
        return {
            "answer": "No relevant documents found in the knowledge base.",
            "sources": [],
            "chunks": [],
            "trace": None,
        }

    top_similarity = 1 - chunks[0]["distance"]  # cosine: distance → similarity

    # Step 2: Query rewrite if similarity is low
    if top_similarity < REWRITE_THRESHOLD:
        rewritten_query = rewrite_query(question)
        print(f"[rewrite] '{question}' → '{rewritten_query}'")
        retry_chunks = retrieve(rewritten_query, top_k=top_k)
        if retry_chunks:
            chunks = retry_chunks
            was_rewritten = True

    retrieval_scores = [round(1 - c["distance"], 4) for c in chunks]

    # Step 3: Rerank with Cohere
    effective_query = rewritten_query if was_rewritten else question
    chunks = rerank(effective_query, chunks, top_n=top_k)
    rerank_scores = [c.get("rerank_score", 0.0) for c in chunks]

    # Step 4: Generate answer
    prompt = build_prompt(effective_query, chunks)
    tokenizer, model = get_model()
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    sources = list({c["source"] for c in chunks})

    # Step 5: Trace
    trace_entry = tracer.log(
        query=question,
        rewritten_query=rewritten_query,
        was_rewritten=was_rewritten,
        top_similarity=top_similarity,
        retrieval_scores=retrieval_scores,
        rerank_scores=rerank_scores,
        sources=sources,
        answer=answer,
    )

    return {
        "answer": answer,
        "sources": sources,
        "chunks": chunks,
        "trace": trace_entry,
    }
