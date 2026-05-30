# Classic RAG

A fully local Retrieval-Augmented Generation system — no cloud LLM required.

**Stage 1** — ingest, retrieve, generate  
**Stage 2** — query rewriting, Cohere reranking, decision trace

## Stack

| Layer | Technology |
|---|---|
| Embeddings | `sentence-transformers` (all-MiniLM-L6-v2) — local |
| Vector store | ChromaDB (local, persistent) |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Reranker | Cohere `rerank-v3.5` (free tier: 1000 calls/month) |
| LLM | `google/flan-t5-small` via HuggingFace — local |
| API | FastAPI |
| UI | Single-page HTML (no framework) |

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd classic-rag
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set your Cohere API key (free at [cohere.com](https://cohere.com)):

```env
COHERE_API_KEY=your_cohere_api_key_here
```

All other values have sensible defaults and can be left as-is.

### 3. Add documents

Drop `.txt` or `.md` files into the `docs/` folder, or upload them via the UI.

### 4. Run

```bash
uvicorn api:app --reload
```

Open **http://127.0.0.1:8000** in your browser.

## Usage

The UI has three tabs:

**📄 Upload** — drag & drop `.txt` / `.md` files. They are saved to `docs/` and ingested into ChromaDB automatically.

**💬 Query** — ask a question. The pipeline:
1. Retrieves top-K chunks from ChromaDB
2. If top similarity < 0.5, rewrites the query and retries
3. Reranks with Cohere rerank-v3.5
4. Generates an answer with flan-t5
5. Shows the rewrite badge and source files

**🔍 Trace** — visual timeline of every decision: similarity scores, rewrite detection, rerank scores, and the final answer.

You can also run ingestion from the command line:

```bash
python ingest.py --docs_dir ./docs
```

## Project structure

```
├── ingest.py        # Load → chunk → embed → store in ChromaDB
├── retriever.py     # Query ChromaDB, return top-k chunks
├── reranker.py      # Cohere rerank-v3.5
├── rag.py           # Full pipeline: retrieve → rewrite → rerank → generate → trace
├── tracer.py        # Append/read trace_log.jsonl
├── api.py           # FastAPI: /upload /ingest /query /trace
├── index.html       # Single-page UI (Upload · Query · Trace tabs)
├── requirements.txt
├── .env.example     # Environment variable template
└── docs/            # Drop your documents here
```

## Configuration

All settings are in `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `COHERE_API_KEY` | — | Required for reranking |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model |
| `LLM_MODEL` | `google/flan-t5-small` | HuggingFace generative model |
| `CHROMA_PATH` | `./chroma_db` | ChromaDB storage path |
| `COLLECTION_NAME` | `rag_docs` | ChromaDB collection name |
| `TOP_K` | `3` | Chunks retrieved per query |
| `REWRITE_THRESHOLD` | `0.5` | Similarity below this triggers query rewrite |
| `MAX_NEW_TOKENS` | `256` | Max LLM output tokens |

## Upgrading the LLM

`flan-t5-small` is fast but produces short answers. For better quality, update `LLM_MODEL` in `.env`:

```env
LLM_MODEL=google/flan-t5-base   # better answers, ~250 MB
LLM_MODEL=google/flan-t5-large  # best answers, ~800 MB
```

Models download automatically from HuggingFace on first run.
