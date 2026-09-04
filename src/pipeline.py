import time
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from config import KB_DIR, STRATEGIES
from src.loaders import load_corpus
from src.chunkers import chunk_fixed, chunk_semantic, chunk_hierarchical
from src.vectorstore import reset_collection

_CHUNKER = {
    "fixed": chunk_fixed,
    "semantic": chunk_semantic,
    "hierarchical": chunk_hierarchical,
}


def ingest(
    strategies: list[str] | None = None,
    progress: object = None,
) -> dict:
    strategies = strategies or list(STRATEGIES)
    docs = load_corpus(KB_DIR)
    stats = {}

    for strategy in strategies:
        t0 = time.perf_counter()
        chunker = _CHUNKER[strategy]
        embeddings_ref = _get_embeddings_ref()
        all_chunks = []
        for doc in docs:
            all_chunks.extend(chunker(doc, embeddings_ref))

        store = reset_collection(strategy, embeddings_ref)
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            store.add_documents(batch)

        elapsed = time.perf_counter() - t0
        char_lens = [c.metadata.get("char_len", len(c.page_content)) for c in all_chunks]
        sections = {c.metadata.get("section", "") for c in all_chunks if c.metadata.get("section")}
        stats[strategy] = {
            "n_chunks": len(all_chunks),
            "avg_chunk_chars": round(sum(char_lens) / len(char_lens)) if char_lens else 0,
            "n_sections": len(sections),
            "ingest_secs": round(elapsed, 2),
        }

        if progress is not None:
            progress(strategy)

    return stats


_embeddings_cache: HuggingFaceEmbeddings | None = None


def _get_embeddings_ref() -> HuggingFaceEmbeddings:
    global _embeddings_cache
    if _embeddings_cache is None:
        from src.embeddings import get_embeddings
        _embeddings_cache = get_embeddings()
    return _embeddings_cache


def set_embeddings_ref(embeddings: HuggingFaceEmbeddings) -> None:
    global _embeddings_cache
    _embeddings_cache = embeddings
