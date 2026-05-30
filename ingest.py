"""
ingest.py — Load .txt/.md docs, chunk, embed with sentence-transformers, store in ChromaDB.

Usage:
    python ingest.py --docs_dir ./docs
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "rag_docs")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
EMBED_BATCH_SIZE = 64


def load_documents(docs_dir: str) -> list[dict]:
    """Recursively load all .txt and .md files from a directory."""
    docs = []
    for path in Path(docs_dir).rglob("*"):
        if path.suffix in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8")
            docs.append({"source": str(path), "text": text})
    print(f"Loaded {len(docs)} documents from '{docs_dir}'")
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    """Split documents into chunks using recursive character splitting."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in docs:
        splits = splitter.split_text(doc["text"])
        for i, split in enumerate(splits):
            chunks.append({
                "id": f"{doc['source']}::chunk{i}",
                "text": split,
                "source": doc["source"],
                "chunk_index": i,
            })
    print(f"Created {len(chunks)} chunks")
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Embed chunk texts using sentence-transformers (runs locally)."""
    model = SentenceTransformer(EMBED_MODEL)
    texts = [c["text"] for c in chunks]

    embeddings = model.encode(
        texts,
        batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=True,
    ).tolist()

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    return chunks


def store_chunks(chunks: list[dict]) -> None:
    """Store chunks and embeddings in ChromaDB."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [c["id"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [{"source": c["source"], "chunk_index": c["chunk_index"]} for c in chunks]

    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
    print(f"Stored {len(chunks)} chunks in ChromaDB collection '{COLLECTION_NAME}'")


def ingest(docs_dir: str) -> None:
    docs = load_documents(docs_dir)
    if not docs:
        print("No .txt or .md files found. Add files to the docs directory and retry.")
        return
    chunks = chunk_documents(docs)
    chunks = embed_chunks(chunks)
    store_chunks(chunks)
    print("Ingestion complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB.")
    parser.add_argument("--docs_dir", default="./docs", help="Directory containing .txt/.md files")
    args = parser.parse_args()
    ingest(args.docs_dir)
