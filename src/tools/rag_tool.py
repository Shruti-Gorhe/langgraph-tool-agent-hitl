"""Adapter that exposes the Week 3 retrieval benchmark as an agent tool."""

import json
import os

from langchain.tools import tool

from src.embeddings import get_embeddings
from src.retrievers import build_retriever, retrieve


_DEFAULT_STRATEGY = os.getenv("AGENT_RAG_STRATEGY", "hierarchical")
_DEFAULT_METHOD = os.getenv("AGENT_RAG_METHOD", "hybrid")


@tool
def enterprise_search(query: str) -> str:
    """Search the company's internal policy knowledge base.

    Use this for questions about company policies, benefits, leave, expenses,
    security, privacy, travel, procurement, onboarding, remote work, or conduct.
    The returned snippets are evidence; do not invent policy details not present
    in the retrieved evidence.
    """
    query = query.strip()
    if not query:
        return json.dumps({"error": "Query cannot be empty."})

    strategy = os.getenv("AGENT_RAG_STRATEGY", _DEFAULT_STRATEGY)
    method = os.getenv("AGENT_RAG_METHOD", _DEFAULT_METHOD)
    if strategy not in {"fixed", "semantic", "hierarchical"}:
        strategy = "hierarchical"
    if method not in {"semantic", "hybrid"}:
        method = "hybrid"

    embeddings = get_embeddings()
    retriever = build_retriever(strategy, method, embeddings)
    docs, latency_ms = retrieve(retriever, query)

    results = []
    for rank, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        results.append(
            {
                "rank": rank,
                "source": meta.get("source", "unknown"),
                "section": meta.get("section") or meta.get("subsection"),
                "content": doc.page_content[:1800],
            }
        )

    return json.dumps(
        {
            "query": query,
            "strategy": strategy,
            "method": method,
            "latency_ms": round(latency_ms, 2),
            "results": results,
        },
        indent=2,
    )
