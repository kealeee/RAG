"""
api.py — FastAPI app exposing /upload, /ingest, /query, and /trace endpoints.

Run:
    uvicorn api:app --reload
"""

import os
import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest
from rag import query as rag_query
import tracer

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

app = FastAPI(title="Classic RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class IngestRequest(BaseModel):
    docs_dir: str = DOCS_DIR


class IngestResponse(BaseModel):
    message: str


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


class ChunkResult(BaseModel):
    text: str
    source: str
    chunk_index: int
    distance: float
    rerank_score: float = 0.0


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    chunks: list[ChunkResult]
    trace: dict | None = None


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    """Upload .txt or .md files, save to docs/, then ingest them."""
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".txt", ".md"}:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}. Only .txt and .md allowed."
            )
        dest = docs_path / file.filename
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)

    ingest(DOCS_DIR)
    return {"message": f"Uploaded and ingested: {', '.join(saved)}"}


@app.post("/ingest", response_model=IngestResponse)
def ingest_documents(req: IngestRequest):
    try:
        ingest(req.docs_dir)
        return IngestResponse(message=f"Ingestion complete for '{req.docs_dir}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty")
    try:
        result = rag_query(req.question, top_k=req.top_k)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/trace")
def get_trace():
    """Return all trace entries, newest first."""
    return tracer.read_all()
