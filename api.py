"""
api.py — FastAPI app exposing /upload, /ingest, /query, /trace, and /evaluate endpoints.

Run:
    uvicorn api:app --reload
"""

import os
import sys
import shutil
import subprocess
import threading
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ingest import ingest
from rag import query as rag_query
import tracer

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")

app = FastAPI(title="Agentic RAG API", version="3.0.0")

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
    docs_path = Path(DOCS_DIR)
    docs_path.mkdir(parents=True, exist_ok=True)
    saved = []
    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in {".txt", ".md"}:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.filename}")
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
    return tracer.read_all()


# ── Evaluate (subprocess — avoids asyncio conflicts with Ragas) ────────────────

RESULTS_FILE  = os.getenv("EVAL_RESULTS_FILE", "./eval_results.json")
_eval_process = None


def _run_eval_subprocess():
    global _eval_process
    _eval_process = subprocess.Popen(
        [sys.executable, "-u", "evaluate.py"],  # -u = unbuffered stdout
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in _eval_process.stdout:
        print(line, end="", flush=True)
    _eval_process.wait()


@app.post("/evaluate/start")
def start_evaluation():
    global _eval_process
    if _eval_process and _eval_process.poll() is None:
        return {"job_id": "running", "message": "already running"}
    Path(RESULTS_FILE).unlink(missing_ok=True)
    t = threading.Thread(target=_run_eval_subprocess, daemon=True)
    t.start()
    return {"job_id": "running"}


@app.get("/evaluate/status/{job_id}")
def eval_status(job_id: str):
    results_path = Path(RESULTS_FILE)
    if not results_path.exists():
        return {"status": "running", "progress": [], "result": None, "error": None}
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return {"status": "running", "progress": [], "result": None, "error": None}
