"""
evaluate.py — Run the golden set through the RAG pipeline.
Judge: Gemini (direct API, model auto-detected from your key).
Metrics: faithfulness, answer_relevancy, context_precision, context_recall
"""

import os, re, json, time
from pathlib import Path
from dotenv import load_dotenv
import anthropic

from rag import query as rag_query

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
CLAUDE_MODEL      = os.getenv("JUDGE_MODEL", "claude-haiku-4-5")
GOLDEN_SET_FILE   = os.getenv("GOLDEN_SET_FILE", "./golden_set.json")
RESULTS_FILE      = os.getenv("EVAL_RESULTS_FILE", "./eval_results.json")
TOP_K             = int(os.getenv("TOP_K", "3"))
CALL_DELAY        = 1

_client = None

def get_client():
    global _client
    if _client is None:
        print(f"  Using judge model: {CLAUDE_MODEL}", flush=True)
        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    return _client

def judge(prompt: str) -> float:
    try:
        time.sleep(CALL_DELAY)
        client = get_client()
        message = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=16,
            messages=[{"role": "user", "content": prompt}],
        )
        text = message.content[0].text.strip()
        print(f"    [judge] {repr(text[:80])}", flush=True)
        # decimal like 0.85
        for m in re.findall(r"\d+\.\d+", text):
            s = float(m)
            if 0.0 <= s <= 1.0:
                return round(s, 4)
        # percentage like 85%
        p = re.search(r"(\d{1,3})\s*%", text)
        if p:
            return round(float(p.group(1)) / 100, 4)
        # x/10
        f = re.search(r"(\d)\s*/\s*10", text)
        if f:
            return round(float(f.group(1)) / 10, 4)
        # bare integer
        if text.strip() in ("0","1"):
            return float(text.strip())
        print(f"    [judge] unparseable, returning 0.0", flush=True)
        return 0.0
    except Exception as e:
        print(f"    [judge error] {e}", flush=True)
        return 0.0


def score_all(question, answer, contexts, ground_truth):
    ctx = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return {
        "faithfulness": judge(
            f"Rate 0.0-1.0: how faithful is the answer to the context?\n"
            f"1.0=fully supported, 0.0=contradicts context.\n"
            f"Reply with ONE decimal number only.\n\nContext:\n{ctx}\n\nAnswer:\n{answer}"
        ),
        "answer_relevancy": judge(
            f"Rate 0.0-1.0: how well does the answer address the question?\n"
            f"1.0=directly answers it, 0.0=off-topic.\n"
            f"Reply with ONE decimal number only.\n\nQuestion: {question}\nAnswer: {answer}"
        ),
        "context_precision": judge(
            f"Rate 0.0-1.0: how relevant are the retrieved chunks to the question?\n"
            f"1.0=all relevant, 0.0=none relevant.\n"
            f"Reply with ONE decimal number only.\n\nQuestion: {question}\n\nChunks:\n{ctx}"
        ),
        "context_recall": judge(
            f"Rate 0.0-1.0: do the chunks contain enough info to produce the correct answer?\n"
            f"1.0=fully covers ground truth, 0.0=missing key info.\n"
            f"Reply with ONE decimal number only.\n\nGround truth: {ground_truth}\n\nChunks:\n{ctx}"
        ),
    }


def _save(data):
    Path(RESULTS_FILE).write_text(json.dumps(data), encoding="utf-8")


def run_evaluation():
    with open(GOLDEN_SET_FILE, encoding="utf-8") as f:
        golden_set = json.load(f)

    _save({"status": "running", "progress": [], "result": None, "error": None})
    progress, per_question = [], []

    for i, item in enumerate(golden_set):
        question, ground_truth = item["question"], item["ground_truth"]

        result   = rag_query(question, top_k=TOP_K)
        answer   = result["answer"]
        contexts = [c["text"] for c in result["chunks"]]
        print(f"  ✓ RAG [{i+1}/10] {question[:55]}…")

        progress.append({"stage": "rag", "question": question, "index": i})
        _save({"status": "running", "progress": progress, "result": None, "error": None})
        time.sleep(7)  # Cohere rate limit

        print(f"    scoring…")
        progress.append({"stage": "scoring", "question": question, "index": i})
        _save({"status": "running", "progress": progress, "result": None, "error": None})

        scores = score_all(question, answer, contexts, ground_truth)
        print(f"    {scores}")

        per_question.append({
            "question": question, "answer": answer,
            "ground_truth": ground_truth, "contexts": contexts, "scores": scores,
        })

    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    aggregate = {
        m: round(sum(q["scores"][m] for q in per_question) / len(per_question), 4)
        for m in metric_names
    }

    print("\n── Scorecard ──")
    for m, s in aggregate.items():
        print(f"  {m:<22} {s:.4f}  {'█'*int(s*20)}")

    final = {"scores": aggregate, "per_question": per_question}
    _save({"status": "done", "progress": progress, "result": final, "error": None})
    return final


if __name__ == "__main__":
    try:
        run_evaluation()
    except Exception as e:
        _save({"status": "error", "progress": [], "result": None, "error": str(e)})
        raise
