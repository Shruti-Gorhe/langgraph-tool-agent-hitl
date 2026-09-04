import time
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from config import TOP_K, ENSEMBLE_WEIGHTS, RRF_C
from src.vectorstore import get_store, read_all_chunks


def build_retriever(strategy: str, method: str, embeddings: HuggingFaceEmbeddings):
    store = get_store(strategy, embeddings)
    vector_retriever = store.as_retriever(search_kwargs={"k": TOP_K})

    if method == "semantic":
        return vector_retriever

    chunks = read_all_chunks(store)
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = TOP_K
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=list(ENSEMBLE_WEIGHTS),
        c=RRF_C,
    )


def retrieve(retriever, query: str) -> tuple[list[Document], float]:
    t0 = time.perf_counter()
    docs = retriever.invoke(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return docs[:TOP_K], elapsed_ms
