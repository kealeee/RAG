"""
tracer.py — Append one trace row per query decision to trace_log.jsonl.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

TRACE_FILE = os.getenv("TRACE_FILE", "./trace_log.jsonl")


def log(
    query: str,
    rewritten_query: str | None,
    was_rewritten: bool,
    top_similarity: float,
    retrieval_scores: list[float],
    rerank_scores: list[float],
    sources: list[str],
    answer: str,
) -> dict:
    """Append a trace entry and return it."""
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "was_rewritten": was_rewritten,
        "rewritten_query": rewritten_query,
        "top_similarity": round(top_similarity, 4),
        "retrieval_scores": [round(s, 4) for s in retrieval_scores],
        "rerank_scores": [round(s, 4) for s in rerank_scores],
        "sources": sources,
        "answer": answer,
    }
    Path(TRACE_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(TRACE_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_all() -> list[dict]:
    """Return all trace entries, newest first."""
    path = Path(TRACE_FILE)
    if not path.exists():
        return []
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return list(reversed(entries))
