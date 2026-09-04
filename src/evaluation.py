import json
import re
import time
from langchain_core.documents import Document
from config import DATA_DIR, TOP_K, TOP_K_VALUES


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[*#]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def is_relevant(doc: Document, evidence_list: list[dict]) -> bool:
    doc_source = doc.metadata.get("source", "")
    doc_text = normalize(doc.page_content)
    for ev in evidence_list:
        if ev["source"] == doc_source and normalize(ev["text"]) in doc_text:
            return True
    return False


def load_questions() -> list[dict]:
    path = DATA_DIR / "benchmark_questions.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _hit_at_k(ranked: list[bool], k: int) -> int:
    return int(any(ranked[:k]))


def _rr_at_k(ranked: list[bool], k: int) -> float:
    for i, hit in enumerate(ranked[:k]):
        if hit:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(
    strategy: str,
    method: str,
    questions: list[dict],
    retriever,
) -> tuple[list[dict], dict]:
    rows = []
    for q in questions:
        t0 = time.perf_counter()
        docs = retriever.invoke(q["question"])
        elapsed_ms = (time.perf_counter() - t0) * 1000
        docs = docs[:TOP_K]

        relevance = [is_relevant(d, q["evidence"]) for d in docs]
        first_rank = next((i + 1 for i, r in enumerate(relevance) if r), None)
        retrieved_chars = [len(d.page_content) for d in docs]

        row = {
            "strategy": strategy,
            "method": method,
            "qid": q["qid"],
            "query_style": q["query_style"],
            "question": q["question"],
            "first_relevant_rank": first_rank,
            "query_ms": round(elapsed_ms, 1),
            "retrieved_chars": sum(retrieved_chars) / len(retrieved_chars) if retrieved_chars else 0,
            "retrieved_ids": "|".join(d.metadata.get("chunk_id", "") for d in docs),
        }
        for k in TOP_K_VALUES:
            row[f"hit@{k}"] = _hit_at_k(relevance, k)
            row[f"rr@{k}"] = round(_rr_at_k(relevance, k), 4)
        rows.append(row)

    n = len(questions)
    aggregates = {}
    for k in TOP_K_VALUES:
        aggregates[f"hit_rate@{k}"] = round(sum(r[f"hit@{k}"] for r in rows) / n, 4)
        aggregates[f"mrr@{k}"] = round(sum(r[f"rr@{k}"] for r in rows) / n, 4)
    aggregates["avg_query_ms"] = round(sum(r["query_ms"] for r in rows) / n, 1)
    aggregates["avg_retrieved_chars"] = round(
        sum(r["retrieved_chars"] for r in rows) / n
    )

    return rows, aggregates
